"""Leave-actors-out cross-validation for RAVDESS.

Actors are partitioned into contiguous, speaker-disjoint folds. For each fold
and seed, the test actors are held out, a small validation group is sampled from
the remaining actors for early stopping, and the rest train the model. The fold
manifest is written next to the fold checkpoint so the evaluated test clips are
exactly the held-out actors.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, resolve_mel_stats
from .dataset import AVEmotionDataset
from .evaluate import evaluate_model
from .train import build_parser as build_train_parser
from .train import train, validate_args

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ("accuracy", "macro_f1", "macro_ovr_auc")


def make_folds(actors: list[int], n_folds: int = 6) -> list[list[int]]:
    """Partition sorted actors into ``n_folds`` contiguous folds."""
    if n_folds <= 0:
        raise ValueError("n_folds must be positive.")
    unique = sorted({int(actor) for actor in actors})
    if len(unique) < n_folds:
        raise ValueError(f"Cannot make {n_folds} folds from {len(unique)} actors.")
    return [list(map(int, chunk)) for chunk in np.array_split(np.asarray(unique), n_folds)]


def split_actors(actors: list[int], test_actors: list[int], n_val_actors: int = 2,
                 seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Return disjoint (train, val, test) actors for one fold."""
    if n_val_actors < 1:
        raise ValueError("n_val_actors must be at least 1.")
    test = sorted({int(actor) for actor in test_actors})
    pool = sorted({int(actor) for actor in actors} - set(test))
    if len(pool) <= n_val_actors:
        raise ValueError("Not enough non-test actors left for train and validation.")
    val = sorted(random.Random(seed).sample(pool, n_val_actors))
    train_actors = sorted(set(pool) - set(val))
    return train_actors, val, test


def assign_splits(manifest: pd.DataFrame, train_actors: list[int], val_actors: list[int],
                  test_actors: list[int]) -> pd.DataFrame:
    """Return a manifest copy whose ``split`` column follows the fold actors."""
    result = manifest.copy()
    result["split"] = "train"
    result.loc[result["actor"].isin(val_actors), "split"] = "val"
    result.loc[result["actor"].isin(test_actors), "split"] = "test"
    covered = set(train_actors) | set(val_actors) | set(test_actors)
    missing = sorted(set(result["actor"].astype(int)) - covered)
    if missing:
        raise ValueError(f"Manifest actors are not assigned to a fold split: {missing}")
    for split in ("train", "val", "test"):
        if not (result["split"] == split).any():
            raise ValueError(f"Fold has no {split} clips.")
    return result


def evaluate_fold(checkpoint_path, data_dir, manifest_path, args, device) -> dict:
    """Evaluate a fold checkpoint on its held-out test actors."""
    model, config = load_checkpoint(checkpoint_path, device)
    dataset = AVEmotionDataset(
        data_dir, "test", manifest_path=str(manifest_path),
        include_landmarks=config.use_landmarks,
        landmark_dir=args.landmark_dir,
        landmark_feature_mode=config.landmark_feature_mode,
        face_frame_dir=args.face_frame_dir if config.face_crop else None,
        mel_norm=config.mel_norm,
        mel_stats_path=resolve_mel_stats(config, ROOT),
        include_waveform=config.audio_encoder == "wavlm",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    acc, f1, auc, _, _, _, _ = evaluate_model(model, loader, device, config.n_classes)
    return {"accuracy": float(acc), "macro_f1": float(f1), "macro_ovr_auc": float(auc),
            "n_test": len(dataset)}


def build_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    parser.description = "Leave-actors-out cross-validation on RAVDESS."
    parser.add_argument("--output_dir", default="reports/cv")
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--val_actors", type=int, default=2,
                        help="Validation actors sampled from the training pool per run.")
    parser.add_argument("--actor_range", nargs=2, type=int, default=[1, 24],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--skip_existing", action="store_true",
                        help="Reuse fold metrics.json files that already exist.")
    return parser


def _validate_cv_args(args) -> None:
    if args.n_folds <= 0:
        raise ValueError("--n_folds must be positive.")
    if args.val_actors < 1:
        raise ValueError("--val_actors must be at least 1.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    low, high = sorted(args.actor_range)
    if low < 1 or high < low:
        raise ValueError("--actor_range must be MIN MAX with 1 <= MIN <= MAX.")


def _mean_std(values: pd.Series) -> dict:
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def summarize(results: pd.DataFrame) -> dict:
    """Overall and per-fold mean/std for each metric."""
    overall = {metric: _mean_std(results[metric]) for metric in METRIC_COLUMNS}
    per_fold = []
    for fold, group in results.groupby("fold", sort=True):
        entry = {"fold": int(fold), "test_actors": group["test_actors"].iloc[0]}
        for metric in METRIC_COLUMNS:
            entry[metric] = _mean_std(group[metric])
        per_fold.append(entry)
    return {
        "n_folds": int(results["fold"].nunique()),
        "seeds": sorted(int(seed) for seed in results["seed"].unique()),
        "n_runs": int(len(results)),
        "overall": overall,
        "per_fold": per_fold,
    }


def markdown_table(summary: dict) -> str:
    base = "# Leave-actors-out cross-validation\n\n"
    base += (f"{summary['n_runs']} runs: {summary['n_folds']} folds x seeds "
             f"{', '.join(str(seed) for seed in summary['seeds'])}. "
             "Each cell is mean +/- population std across seeds.\n\n")
    lines = [
        "| Fold | Test actors | Accuracy | Macro-F1 | Macro AUROC |",
        "|:--|:--|--:|--:|--:|",
    ]
    for entry in summary["per_fold"]:
        cells = [str(entry["fold"]), entry["test_actors"]]
        for metric in METRIC_COLUMNS:
            values = entry[metric]
            cells.append(f"{values['mean']:.3f} +/- {values['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    overall_cells = ["**All**", ""]
    for metric in METRIC_COLUMNS:
        values = summary["overall"][metric]
        overall_cells.append(f"**{values['mean']:.3f} +/- {values['std']:.3f}**")
    lines.append("| " + " | ".join(overall_cells) + " |")
    return base + "\n".join(lines) + "\n"


def run(args) -> dict:
    """Run every fold/seed and write ``cv_results.csv``/``cv_summary.*``."""
    _validate_cv_args(args)
    data_dir = Path(args.data_dir)
    source_manifest = (Path(args.manifest) if args.manifest
                       else data_dir / "manifest.csv")
    manifest = pd.read_csv(source_manifest)
    low, high = sorted(args.actor_range)
    manifest = manifest[(manifest["actor"] >= low) & (manifest["actor"] <= high)].copy()
    if manifest.empty:
        raise ValueError(f"No clips for actors {low}-{high} in {source_manifest}")
    actors = sorted(int(actor) for actor in manifest["actor"].unique())
    folds = make_folds(actors, args.n_folds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for fold_index, test_actors in enumerate(folds):
        for seed in args.seeds:
            fold_dir = output_dir / f"fold_{fold_index:02d}" / f"seed_{seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_actors, val_actors, test = split_actors(
                actors, test_actors, args.val_actors, seed=1000 * fold_index + seed,
            )
            fold_manifest = assign_splits(manifest, train_actors, val_actors, test)
            manifest_path = fold_dir / "manifest.csv"
            fold_manifest.to_csv(manifest_path, index=False)
            metrics_path = fold_dir / "metrics.json"

            if args.skip_existing and metrics_path.is_file():
                record = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {key: value for key, value in record.items() if key != "history"}
                print(f"fold {fold_index} seed {seed}: reusing {metrics_path}")
            else:
                run_args = copy.deepcopy(args)
                run_args.manifest = str(manifest_path)
                run_args.checkpoint_dir = str(fold_dir / "checkpoints")
                run_args.seed = seed
                print(f"\n=== fold {fold_index} seed {seed} | test {test} | val {val_actors} ===")
                result = train(run_args)
                test_metrics = evaluate_fold(result["checkpoint"], data_dir,
                                             manifest_path, args, device)
                split_counts = fold_manifest["split"].value_counts()
                row = {
                    "fold": fold_index,
                    "seed": seed,
                    "test_actors": " ".join(str(actor) for actor in test),
                    "val_actors": " ".join(str(actor) for actor in val_actors),
                    "n_train": int(split_counts.get("train", 0)),
                    "n_val": int(split_counts.get("val", 0)),
                    "n_test": int(test_metrics["n_test"]),
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                    "best_val_score": result["best_score"],
                    **{metric: test_metrics[metric] for metric in METRIC_COLUMNS},
                    "checkpoint": result["checkpoint"],
                }
                metrics_path.write_text(
                    json.dumps({**row, "history": result["history"]}, indent=2),
                    encoding="utf-8",
                )
            rows.append(row)
            results = pd.DataFrame(rows)
            results.to_csv(output_dir / "cv_results.csv", index=False)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    markdown = markdown_table(summary)
    (output_dir / "cv_summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved cross-validation results to {output_dir}")
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        _validate_cv_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    run(args)


if __name__ == "__main__":
    main()
