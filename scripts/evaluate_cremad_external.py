"""Download and evaluate a balanced 50-clip CREMA-D external test sample.

The sample uses the official CREMA-D audiovisual majority vote and one clip per
actor. This is an external test only; no CREMA-D samples are used for training.
The checkpoint sidecar determines face cropping, landmarks, the audio encoder,
and mel normalization.
"""
import argparse
import io
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score, top_k_accuracy_score)

from affect_av.checkpoints import (CheckpointConfigError, load_checkpoint,
                                   resolve_mel_stats)
from affect_av.evaluate import save_reliability_diagram
from affect_av.metrics import (expected_calibration_error, mcnemar,
                               stratified_bootstrap)
from affect_av.dataset import (FRAME_SIZE, center_crop_frames, decode_clip,
                               extract_audio_features, extract_audio_waveform,
                               load_mel_stats, normalize_mel)
from affect_av.face_geometry import select_landmark_features
from affect_av.face_landmarks import (create_landmarker, face_crops_from_frames,
                                      landmarks_from_frames)


DATASET_URL = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv"
VIDEO_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/VideoFlash/{filename}"
CODE_TO_EMOTION = {"A": "angry", "D": "disgust", "F": "fearful",
                   "H": "happy", "N": "neutral", "S": "sad"}
FILENAME_CODE_TO_EMOTION = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
                            "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CLASS_ORDER = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
QUOTAS = {"A": 9, "D": 9, "F": 8, "H": 8, "N": 8, "S": 8}
USER_AGENT = "AV-Emotion-Recognition external evaluation"


def fetch(url, target=None, retries=4):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            return payload
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download failed: {url}: {error}")


def select_sample(table):
    rng = random.Random(20261016)
    candidates = {code: [] for code in QUOTAS}
    for row in table.to_dict("records"):
        name = str(row.get("FileName", ""))
        vote = str(row.get("MultiModalVote", ""))
        if vote not in candidates or not name or name.lower() == "nan":
            continue
        if not name.endswith(".flv"):
            name += ".flv"
        candidates[vote].append({
            "filename": name,
            "actor": name.split("_")[0],
            "vote_code": vote,
            "emotion": CODE_TO_EMOTION[vote],
            "intended_emotion": FILENAME_CODE_TO_EMOTION[name.split("_")[2]],
        })
    for rows in candidates.values():
        rng.shuffle(rows)
    chosen, used_actors = [], set()
    for code, quota in QUOTAS.items():
        selected = []
        for row in candidates[code]:
            if row["actor"] in used_actors:
                continue
            selected.append(row)
            used_actors.add(row["actor"])
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise RuntimeError(f"Only selected {len(selected)}/{quota} clips for label {code}")
        chosen.extend(selected)
    rng.shuffle(chosen)
    return chosen


def prepare_data(data_dir, prepare_only=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir = data_dir / "VideoFlash"
    manifest_path = data_dir / "external_manifest.csv"
    if not manifest_path.is_file():
        metadata = fetch(DATASET_URL)
        table = pd.read_csv(io.BytesIO(metadata))
        chosen = select_sample(table)
        pd.DataFrame(chosen).to_csv(manifest_path, index=False)
    else:
        chosen = pd.read_csv(manifest_path).to_dict("records")
        actors = [row["actor"] for row in chosen]
        if len(set(actors)) != len(actors):
            print("Refreshing sample: prior manifest repeated an actor.", flush=True)
            table = pd.read_csv(io.BytesIO(fetch(DATASET_URL)))
            chosen = select_sample(table)
            pd.DataFrame(chosen).to_csv(manifest_path, index=False)
    missing = [row for row in chosen if not (video_dir / row["filename"]).is_file()]
    for index, row in enumerate(missing, start=1):
        destination = video_dir / row["filename"]
        payload = fetch(VIDEO_URL.format(filename=row["filename"]), destination)
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Git LFS pointer received instead of video: {row['filename']}")
        print(f"Downloaded {index}/{len(missing)}: {row['filename']} ({len(payload):,} bytes)",
              flush=True)
    if prepare_only:
        print(f"Prepared {len(chosen)} CREMA-D clips under {data_dir}")
    return chosen, video_dir


@torch.no_grad()
def predict(model, config, rows, video_dir, device, mel_stats=None, face_crop=True):
    probabilities, truth, filenames, failures = [], [], [], []
    use_mediapipe = config.use_landmarks or face_crop
    landmarker = create_landmarker() if use_mediapipe else None
    try:
        for index, row in enumerate(rows):
            path = video_dir / row["filename"]
            try:
                decoded = None
                if config.modality in ("video", "fusion"):
                    decoded = decode_clip(str(path))
                    if decoded is None:
                        raise ValueError("video frames could not be decoded")
                mel_tensor = waveform = None
                if config.modality in ("audio", "fusion"):
                    if config.audio_encoder == "wavlm":
                        samples = extract_audio_waveform(str(path))
                        if samples is None:
                            raise ValueError("audio stream could not be decoded")
                        waveform = torch.from_numpy(samples).unsqueeze(0).to(device)
                    else:
                        audio = extract_audio_features(str(path))
                        if audio is None:
                            raise ValueError("audio stream could not be decoded")
                        audio = normalize_mel(audio, config.mel_norm, mel_stats)
                        mel_tensor = (torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
                                      .to(device))
                frame_tensor = landmark_tensor = None
                if config.modality in ("video", "fusion"):
                    if use_mediapipe:
                        if face_crop:
                            frames = face_crops_from_frames(
                                decoded, landmarker, size=FRAME_SIZE,
                                timestamp_offset_ms=index * 20000,
                            )
                        else:
                            frames = center_crop_frames(decoded, size=FRAME_SIZE)
                        if config.use_landmarks:
                            features, valid = landmarks_from_frames(
                                decoded, landmarker, timestamp_offset_ms=index * 20000 + 10000
                            )
                            features = select_landmark_features(
                                features, config.landmark_feature_mode
                            )
                            features[:, -1] = valid.astype(np.float32)
                            landmark_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                    else:
                        frames = center_crop_frames(decoded, size=FRAME_SIZE)
                    frame_tensor = (torch.from_numpy(frames).permute(0, 3, 1, 2)
                                    .float().div(255.0).unsqueeze(0).to(device))
                logits = model(mel=mel_tensor, frames=frame_tensor,
                               landmarks=landmark_tensor, waveform=waveform)
                probabilities.append(torch.softmax(logits[0], dim=-1).cpu().numpy())
                truth.append(config.class_names.index(row["emotion"]))
                filenames.append(row["filename"])
                print(f"Evaluated {len(filenames)}/{len(rows)}: {row['filename']}", flush=True)
            except Exception as exc:
                failures.append({"filename": row["filename"], "reason": str(exc)})
                print(f"Skipped {row['filename']}: {exc}", flush=True)
    finally:
        if landmarker is not None:
            landmarker.close()
    return np.asarray(probabilities), np.asarray(truth), filenames, failures


def resolve_checkpoint(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/external/crema_d")
    parser.add_argument("--checkpoint", default="checkpoints/fusion_attention_face.pt")
    parser.add_argument("--report_dir", default="reports/crema_d_external")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--full_frame", action="store_true",
                        help="Force center-cropped frames even for a face-crop checkpoint.")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Number of stratified bootstrap resamples (0 disables).")
    parser.add_argument("--bootstrap_alpha", type=float, default=0.05)
    parser.add_argument("--calibration", action="store_true",
                        help="Compute ECE/MCE and write a reliability diagram.")
    parser.add_argument("--calibration_bins", type=int, default=15)
    parser.add_argument("--compare_checkpoint", default=None,
                        help="Second checkpoint; adds an exact McNemar test on common clips.")
    parser.add_argument("--max_clips", type=int, default=0,
                        help="Evaluate only the first N manifest clips (0 evaluates all).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0")
    if not 0.0 < args.bootstrap_alpha < 1.0:
        parser.error("--bootstrap_alpha must be in (0, 1)")
    if args.calibration_bins <= 0:
        parser.error("--calibration_bins must be > 0")
    if args.max_clips < 0:
        parser.error("--max_clips must be >= 0")
    rows, video_dir = prepare_data(args.data_dir, args.prepare_only)
    if args.prepare_only:
        return
    if args.max_clips:
        rows = rows[:args.max_clips]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    try:
        model, config = load_checkpoint(checkpoint_path, device)
    except CheckpointConfigError as exc:
        raise SystemExit(str(exc)) from exc
    missing_classes = [name for name in CLASS_ORDER if name not in config.class_names]
    if missing_classes:
        raise SystemExit(f"Checkpoint classes are missing CREMA-D labels: {missing_classes}")

    mel_stats = None
    if config.audio_encoder == "cnn" and config.mel_norm == "global":
        mel_stats = load_mel_stats(resolve_mel_stats(config, ROOT))
    face_crop = config.face_crop and not args.full_frame
    probs, y_true, filenames, failures = predict(
        model, config, rows, video_dir, device, mel_stats=mel_stats, face_crop=face_crop
    )
    if len(y_true) < 2:
        raise RuntimeError("Fewer than two clips were successfully evaluated.")

    class_indices = [config.class_names.index(name) for name in CLASS_ORDER]
    y_pred_all = probs.argmax(axis=1)
    known_prediction = np.isin(y_pred_all, class_indices)
    y_pred_6 = np.where(known_prediction, y_pred_all, -1)
    y_prob_6 = probs[:, class_indices]
    accuracy = accuracy_score(y_true, y_pred_6)
    macro_f1 = f1_score(y_true, y_pred_6, labels=class_indices,
                        average="macro", zero_division=0)
    y_true_binary = np.stack([y_true == class_index for class_index in class_indices], axis=1)
    macro_auc = roc_auc_score(y_true_binary, y_prob_6, average="macro")
    balanced_acc = balanced_accuracy_score(y_true, y_pred_6)
    top2 = top_k_accuracy_score(y_true, probs, k=2, labels=np.arange(config.n_classes))
    full_labels = list(range(config.n_classes))
    cm = confusion_matrix(y_true, y_pred_all, labels=full_labels)
    report = classification_report(y_true, y_pred_6, labels=class_indices,
                                   target_names=CLASS_ORDER, output_dict=True,
                                   zero_division=0)
    output_dir = ROOT / args.report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    intended_by_file = {row["filename"]: row["intended_emotion"] for row in rows}
    intended_names = [intended_by_file[name] for name in filenames]
    intended_y = np.asarray([config.class_names.index(name) for name in intended_names])
    intended_auc = None
    intended_report = None
    intended_cm = confusion_matrix(intended_y, y_pred_all, labels=full_labels)
    if all(np.any(intended_y == class_index) for class_index in class_indices):
        intended_binary = np.stack(
            [intended_y == class_index for class_index in class_indices], axis=1
        )
        intended_auc = float(roc_auc_score(intended_binary, y_prob_6, average="macro"))
        intended_report = classification_report(
            intended_y, y_pred_6, labels=class_indices, target_names=CLASS_ORDER,
            output_dict=True, zero_division=0
        )
    intended_accuracy = float(accuracy_score(intended_y, y_pred_6))
    intended_balanced_acc = float(balanced_accuracy_score(intended_y, y_pred_6))
    intended_macro_f1 = float(f1_score(intended_y, y_pred_6, labels=class_indices,
                                       average="macro", zero_division=0))
    intended_top2 = float(top_k_accuracy_score(
        intended_y, probs, k=2, labels=np.arange(config.n_classes)
    ))
    metrics = {
        "dataset": "CREMA-D",
        "n_requested": len(rows),
        "n_evaluated": len(y_true),
        "n_failed": len(failures),
        "sample_selection": "50 clips, balanced by CREMA-D MultiModalVote; one clip per actor; seed 20261016",
        "max_clips": args.max_clips or None,
        "model": str(args.checkpoint),
        "model_config": {
            "modality": config.modality,
            "fusion_type": config.fusion_type,
            "audio_encoder": config.audio_encoder,
            "face_crop": face_crop,
            "use_landmarks": config.use_landmarks,
            "mel_norm": config.mel_norm,
        },
        "n_classes": config.n_classes,
        "accuracy": float(accuracy),
        "multimodal_vote_accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "macro_ovr_auroc_6class": float(macro_auc),
        "intended_label_accuracy": intended_accuracy,
        "intended_label_balanced_accuracy": intended_balanced_acc,
        "intended_label_macro_f1": intended_macro_f1,
        "intended_label_macro_ovr_auroc_6class": intended_auc,
        "intended_label_top2_accuracy_8outputs": intended_top2,
        "intended_label_confusion_matrix_8outputs": intended_cm.tolist(),
        "top2_accuracy_8outputs": float(top2),
        "unsupported_output_predictions": int((~known_prediction).sum()),
        "class_order": CLASS_ORDER,
        "confusion_matrix_8outputs": cm.tolist(),
        "per_class": report,
        "intended_label_per_class": intended_report,
        "failures": failures,
    }
    if args.bootstrap > 0:
        metrics["bootstrap_8outputs"] = stratified_bootstrap(
            y_true, y_pred_all, probs, n_bootstrap=args.bootstrap,
            alpha=args.bootstrap_alpha, seed=args.seed,
        )
    if args.calibration:
        metrics["calibration_8outputs"] = expected_calibration_error(
            y_true, probs, n_bins=args.calibration_bins
        )
        save_reliability_diagram(
            metrics["calibration_8outputs"]["bins"],
            output_dir / "reliability_diagram.png",
            f"CREMA-D external reliability (ECE {metrics['calibration_8outputs']['ece']:.3f})",
        )
    if args.compare_checkpoint:
        comparison_path = resolve_checkpoint(args.compare_checkpoint)
        if comparison_path.resolve() == checkpoint_path.resolve():
            print("Skipped comparison: the comparison checkpoint is the evaluated model.")
        else:
            comparison_model, comparison_config = load_checkpoint(comparison_path, device)
            missing = [name for name in CLASS_ORDER
                       if name not in comparison_config.class_names]
            if missing:
                raise SystemExit(
                    f"Comparison checkpoint is missing CREMA-D labels: {missing}"
                )
            comparison_stats = None
            if (comparison_config.audio_encoder == "cnn"
                    and comparison_config.mel_norm == "global"):
                comparison_stats = load_mel_stats(
                    resolve_mel_stats(comparison_config, ROOT)
                )
            comparison_probs, comparison_truth, comparison_files, comparison_failures = predict(
                comparison_model, comparison_config, rows, video_dir, device,
                mel_stats=comparison_stats,
                face_crop=comparison_config.face_crop and not args.full_frame,
            )
            primary_index = {name: index for index, name in enumerate(filenames)}
            comparison_index = {name: index for index, name in enumerate(comparison_files)}
            common = [name for name in filenames if name in comparison_index]
            if len(common) < 2:
                raise RuntimeError("Fewer than two clips were evaluated by both checkpoints.")
            primary_rows = np.asarray([primary_index[name] for name in common])
            comparison_rows = np.asarray([comparison_index[name] for name in common])
            common_truth = y_true[primary_rows]
            if not np.array_equal(comparison_truth[comparison_rows], common_truth):
                raise RuntimeError("Checkpoints disagree on the common clip labels.")
            primary_pred = probs[primary_rows].argmax(axis=1)
            comparison_pred = comparison_probs[comparison_rows].argmax(axis=1)
            metrics["comparison"] = {
                "checkpoint": str(args.compare_checkpoint),
                "n_common": len(common),
                "n_failed": len(comparison_failures),
                "accuracy": float(accuracy_score(common_truth, comparison_pred)),
                "mcnemar": mcnemar(common_truth, primary_pred, comparison_pred),
            }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    sample_out = pd.DataFrame({
        "filename": filenames,
        "true_emotion": [config.class_names[label] for label in y_true],
        "intended_emotion": intended_names,
        "predicted_emotion": [config.class_names[pred] for pred in y_pred_all],
        "confidence": probs.max(axis=1),
    })
    sample_out.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(cm, index=config.class_names, columns=config.class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    print(json.dumps({k: metrics[k] for k in (
        "n_evaluated", "n_failed", "accuracy", "balanced_accuracy", "macro_f1",
        "macro_ovr_auroc_6class", "top2_accuracy_8outputs",
        "unsupported_output_predictions", "intended_label_accuracy",
        "intended_label_balanced_accuracy", "intended_label_macro_f1",
        "intended_label_macro_ovr_auroc_6class", "intended_label_top2_accuracy_8outputs")}, indent=2))
    if args.bootstrap > 0:
        interval = metrics["bootstrap_8outputs"]["accuracy"]
        print(f"Accuracy {100 * (1 - args.bootstrap_alpha):.0f}% CI (8 outputs): "
              f"{interval['low']:.3f}..{interval['high']:.3f}")
    if args.calibration:
        print(f"ECE (8 outputs): {metrics['calibration_8outputs']['ece']:.3f}")
    if "comparison" in metrics:
        print(f"McNemar p={metrics['comparison']['mcnemar']['p_value']:.3f} "
              f"over {metrics['comparison']['n_common']} common clips")
    print(f"Saved external evaluation report to {output_dir}")


if __name__ == "__main__":
    main()
