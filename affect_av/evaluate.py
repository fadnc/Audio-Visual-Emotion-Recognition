"""
Evaluate trained checkpoints on the validation or test split.

Checkpoints are discovered recursively (``*.pt`` with a ``*.json`` sidecar) and
rebuilt from their sidecar, so modalities, face cropping, landmarks, audio
encoders, and mel normalization are never guessed. Writes an ablation table and
a confusion matrix for the best macro-F1 checkpoint.

Optional statistics (all off by default):
  --bootstrap N         stratified bootstrap confidence intervals
  --calibration         expected calibration error and reliability diagram
  --temperature         fit temperature scaling on --calibration_split
  --compare_checkpoint  McNemar test against a second checkpoint
  --max_clips N         evaluate only the first N clips

Usage:
    affect-eval --data_dir data/processed --checkpoint_dir checkpoints
    affect-eval --split val --checkpoint checkpoints/fusion_attention.pt
    affect-eval --bootstrap 1000 --calibration --temperature
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from .checkpoints import (CheckpointConfigError, checkpoint_tag, discover_checkpoints,
                          load_checkpoint, resolve_mel_stats)
from .dataset import AVEmotionDataset
from .metrics import (apply_temperature, expected_calibration_error, fit_temperature,
                      macro_ovr_auc, mcnemar, stratified_bootstrap)

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def collect_outputs(model, loader, device):
    """Return labels, predictions, probabilities, and logits in loader order."""
    model.eval()
    all_labels, all_preds, all_probs, all_logits = [], [], [], []
    for batch in loader:
        mel = batch["mel"].to(device)
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        landmarks = batch.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device)
        waveform = batch.get("waveform")
        if waveform is not None:
            waveform = waveform.to(device)
        logits = model(mel=mel, frames=frames, landmarks=landmarks, waveform=waveform)
        all_logits.extend(logits.cpu().numpy())
        all_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return (np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probs),
            np.asarray(all_logits))


def evaluate_model(model, loader, device, n_classes):
    labels, preds, probs, logits = collect_outputs(model, loader, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, labels=list(range(n_classes)), average="macro",
                  zero_division=0)
    auc = macro_ovr_auc(labels, probs)
    return acc, f1, auc, labels, preds, probs, logits


def build_loader(args, config, split=None):
    dataset = AVEmotionDataset(
        args.data_dir, split or args.split,
        include_landmarks=config.use_landmarks,
        landmark_dir=args.landmark_dir,
        landmark_feature_mode=config.landmark_feature_mode,
        face_frame_dir=args.face_frame_dir if config.face_crop else None,
        mel_norm=config.mel_norm,
        mel_stats_path=resolve_mel_stats(config, ROOT),
        include_waveform=config.audio_encoder == "wavlm",
    )
    if len(dataset) == 0:
        raise SystemExit(f"No clips for split '{split or args.split}' under {args.data_dir}.")
    if args.max_clips and len(dataset) > args.max_clips:
        dataset = Subset(dataset, range(args.max_clips))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)


def resolve_tag(path) -> str:
    try:
        return checkpoint_tag(Path(path).resolve(), ROOT)
    except ValueError:
        return Path(path).stem


def save_reliability_diagram(bins: list[dict], output_path: Path, title: str) -> None:
    plotted = [entry for entry in bins if entry["count"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    if plotted:
        width = (plotted[0]["upper"] - plotted[0]["lower"]) * 0.9
        centers = [(entry["lower"] + entry["upper"]) / 2 for entry in plotted]
        ax.bar(centers, [entry["accuracy"] for entry in plotted], width=width,
               alpha=0.65, label="Accuracy")
        ax.plot(centers, [entry["confidence"] for entry in plotted], "o-",
                label="Mean confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--checkpoint", default=None,
                    help="Evaluate a single .pt file instead of discovering every checkpoint.")
    ap.add_argument("--landmark_dir", default="data/landmarks")
    ap.add_argument("--face_frame_dir", default="data/face_frames")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="Use validation while tuning; reserve test for final reporting.")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Number of stratified bootstrap resamples (0 disables).")
    ap.add_argument("--bootstrap_alpha", type=float, default=0.05)
    ap.add_argument("--calibration", action="store_true",
                    help="Compute ECE/MCE and write a reliability diagram.")
    ap.add_argument("--calibration_bins", type=int, default=15)
    ap.add_argument("--temperature", action="store_true",
                    help="Fit temperature scaling on --calibration_split and report ECE after.")
    ap.add_argument("--calibration_split", choices=["val", "test"], default="val")
    ap.add_argument("--compare_checkpoint", default=None,
                    help="Second .pt checkpoint; adds an exact McNemar test on the same clips.")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="Evaluate only the first N clips (0 evaluates all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0")
    if not 0.0 < args.bootstrap_alpha < 1.0:
        ap.error("--bootstrap_alpha must be in (0, 1)")
    if args.calibration_bins <= 0:
        ap.error("--calibration_bins must be > 0")
    if args.max_clips < 0:
        ap.error("--max_clips must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        paths = [Path(args.checkpoint)]
    else:
        paths = discover_checkpoints(args.checkpoint_dir)
    if not paths:
        raise SystemExit(
            f"No checkpoints with sidecars found under {args.checkpoint_dir}. "
            "Train with affect-train, which writes <name>.json next to <name>.pt."
        )

    results = []
    best = None
    details = {}

    for path in paths:
        tag = resolve_tag(path)
        try:
            model, config = load_checkpoint(path, device)
            loader = build_loader(args, config)
        except (CheckpointConfigError, FileNotFoundError) as exc:
            print(f"skip {tag}: {exc}")
            continue
        acc, f1, auc, labels, preds, probs, logits = evaluate_model(
            model, loader, device, config.n_classes
        )
        results.append({"model": tag, "accuracy": round(acc, 3),
                        "macro_f1": round(f1, 3), "macro_ovr_auc": round(auc, 3)})
        print(f"{tag:32s} | acc {acc:.3f} | macro-F1 {f1:.3f} | macro-AUROC {auc:.3f}")

        model_details = {}
        if args.bootstrap > 0:
            model_details["bootstrap"] = stratified_bootstrap(
                labels, preds, probs, n_bootstrap=args.bootstrap,
                alpha=args.bootstrap_alpha, seed=args.seed,
            )
        if args.calibration or args.temperature:
            model_details["calibration"] = expected_calibration_error(
                labels, probs, n_bins=args.calibration_bins
            )
        if args.temperature:
            if args.calibration_split == args.split:
                calibration_labels, calibration_logits = labels, logits
            else:
                calibration_loader = build_loader(args, config, split=args.calibration_split)
                if len(calibration_loader.dataset) == 0:
                    raise SystemExit(
                        f"No clips in the calibration split '{args.calibration_split}'."
                    )
                _, _, _, calibration_labels, _, _, calibration_logits = evaluate_model(
                    model, calibration_loader, device, config.n_classes
                )
            temperature = fit_temperature(calibration_logits, calibration_labels)
            scaled = apply_temperature(logits, temperature)
            model_details["temperature"] = {
                "value": temperature,
                "calibration_split": args.calibration_split,
                "ece_before": expected_calibration_error(
                    labels, probs, n_bins=args.calibration_bins)["ece"],
                "ece_after": expected_calibration_error(
                    labels, scaled, n_bins=args.calibration_bins)["ece"],
                "accuracy_after": float(accuracy_score(labels, scaled.argmax(axis=1))),
                "macro_f1_after": float(f1_score(
                    labels, scaled.argmax(axis=1), labels=list(range(config.n_classes)),
                    average="macro", zero_division=0)),
            }
        if model_details:
            details[tag] = model_details

        if best is None or f1 > best["f1"]:
            best = {"tag": tag, "f1": f1, "labels": labels, "preds": preds,
                    "probs": probs, "class_names": config.class_names,
                    "n_classes": config.n_classes}

    if not results:
        raise SystemExit("No checkpoint could be evaluated.")

    if args.compare_checkpoint:
        comparison_path = Path(args.compare_checkpoint)
        comparison_tag = resolve_tag(comparison_path)
        if compare_path_matches(comparison_path, paths):
            print(f"skip comparison: {comparison_tag} was already evaluated.")
        else:
            try:
                comparison_model, comparison_config = load_checkpoint(comparison_path, device)
            except (CheckpointConfigError, FileNotFoundError) as exc:
                raise SystemExit(f"Could not load comparison checkpoint: {exc}") from exc
            if comparison_config.class_names != best["class_names"]:
                raise SystemExit(
                    "Comparison checkpoint classes differ from the best checkpoint classes."
                )
            comparison_loader = build_loader(args, comparison_config)
            c_acc, c_f1, c_auc, c_labels, c_preds, c_probs, _ = evaluate_model(
                comparison_model, comparison_loader, device, comparison_config.n_classes
            )
            if not np.array_equal(c_labels, best["labels"]):
                raise SystemExit("Comparison checkpoint was evaluated on different clips.")
            results.append({"model": comparison_tag, "accuracy": round(c_acc, 3),
                            "macro_f1": round(c_f1, 3), "macro_ovr_auc": round(c_auc, 3)})
            comparison_details = {
                "checkpoint": str(comparison_path),
                "tag": comparison_tag,
                "accuracy": float(c_acc),
                "macro_f1": float(c_f1),
                "macro_ovr_auc": float(c_auc),
                "mcnemar": mcnemar(best["labels"], best["preds"], c_preds),
            }
            if args.bootstrap > 0:
                comparison_details["bootstrap"] = stratified_bootstrap(
                    c_labels, c_preds, c_probs, n_bootstrap=args.bootstrap,
                    alpha=args.bootstrap_alpha, seed=args.seed,
                )
            details["comparison"] = comparison_details
            print(f"{comparison_tag:32s} | acc {c_acc:.3f} | macro-F1 {c_f1:.3f} "
                  f"| macro-AUROC {c_auc:.3f} (comparison)")

    df = pd.DataFrame(results)
    out_csv = Path(args.checkpoint_dir) / "ablation_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ablation table -> {out_csv}")
    print(df.to_string(index=False))

    if details:
        details_out = {
            "split": args.split,
            "max_clips": args.max_clips or None,
            "bootstrap": ({"n_bootstrap": args.bootstrap, "alpha": args.bootstrap_alpha,
                           "seed": args.seed} if args.bootstrap else None),
            "models": {tag: value for tag, value in details.items() if tag != "comparison"},
            "comparison": details.get("comparison"),
        }
        details_path = Path(args.checkpoint_dir) / "evaluation_details.json"
        details_path.write_text(json.dumps(details_out, indent=2), encoding="utf-8")
        print(f"Saved statistical evaluation -> {details_path}")

    if args.calibration or args.temperature:
        reliability = expected_calibration_error(
            best["labels"], best["probs"], n_bins=args.calibration_bins
        )
        reliability_path = Path(args.checkpoint_dir) / "reliability_diagram.png"
        save_reliability_diagram(
            reliability["bins"], reliability_path,
            f"Reliability — best model: {best['tag']} (ECE {reliability['ece']:.3f})",
        )
        print(f"Saved reliability diagram -> {reliability_path}")

    class_names = best["class_names"]
    cm = confusion_matrix(best["labels"], best["preds"],
                          labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — best model: {best['tag']}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_png = Path(args.checkpoint_dir) / "confusion_matrix.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_png}")


def compare_path_matches(comparison_path: Path, paths: list[Path]) -> bool:
    """True when the comparison checkpoint is already in the evaluated list."""
    try:
        comparison = comparison_path.resolve()
    except OSError:
        return False
    return any(Path(path).resolve() == comparison for path in paths)


if __name__ == "__main__":
    main()
