"""Fold construction, manifest subsetting, and CV CLI tests."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from affect_av.checkpoints import CheckpointConfig, save_checkpoint
from affect_av.cross_validate import (assign_splits, build_parser, evaluate_fold,
                                      make_folds, markdown_table, split_actors,
                                      summarize)
from affect_av.dataset import AVEmotionDataset, EMOTION_TO_IDX
from affect_av.models import AVEmotionModel

ROOT = Path(__file__).resolve().parents[1]


def _manifest(rows):
    return pd.DataFrame(rows)


def _row(actor, clip, split, label=0):
    return {"clip_id": clip, "actor": actor, "emotion": "angry", "label": label,
            "split": split, "source_path": f"raw/{clip}.mp4"}


def test_make_folds_is_disjoint_and_covers_all_actors():
    folds = make_folds(list(range(1, 25)), n_folds=6)
    assert [len(fold) for fold in folds] == [4] * 6
    assert sorted(actor for fold in folds for actor in fold) == list(range(1, 25))
    assert sum(len(set(fold)) for fold in folds) == 24


def test_make_folds_rejects_too_few_actors():
    with pytest.raises(ValueError):
        make_folds([1, 2, 3], n_folds=6)


def test_split_actors_is_deterministic_and_disjoint():
    first = split_actors(list(range(1, 25)), [1, 2, 3, 4], n_val_actors=2, seed=7)
    second = split_actors(list(range(1, 25)), [1, 2, 3, 4], n_val_actors=2, seed=7)
    train, val, test = first
    assert first == second
    assert test == [1, 2, 3, 4]
    assert len(val) == 2
    assert set(train) | set(val) | set(test) == set(range(1, 25))
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


def test_split_actors_rejects_full_validation_pool():
    with pytest.raises(ValueError):
        split_actors([1, 2, 3], [1], n_val_actors=2)


def test_assign_splits_follows_actors_and_keeps_input():
    manifest = _manifest([_row(actor, f"clip_{actor}", "test") for actor in (1, 2, 3)])
    fold = assign_splits(manifest, train_actors=[1], val_actors=[2], test_actors=[3])
    assert set(fold.loc[fold["actor"] == 1, "split"]) == {"train"}
    assert set(fold.loc[fold["actor"] == 2, "split"]) == {"val"}
    assert set(fold.loc[fold["actor"] == 3, "split"]) == {"test"}
    assert set(manifest["split"]) == {"test"}


def test_assign_splits_rejects_unassigned_actors():
    manifest = _manifest([_row(1, "a", "train"), _row(9, "b", "train")])
    with pytest.raises(ValueError, match="not assigned"):
        assign_splits(manifest, train_actors=[1], val_actors=[2], test_actors=[3])


def test_dataset_reads_split_from_manifest_override(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "clip_0.npz",
        frames=np.zeros((8, 112, 112, 3), dtype=np.uint8),
        mel=np.zeros((64, 94), dtype=np.float32),
        waveform=np.zeros(48000, dtype=np.int16),
    )
    manifest_path = tmp_path / "fold.csv"
    _manifest([_row(5, "clip_0", "test")]).to_csv(manifest_path, index=False)

    dataset = AVEmotionDataset(str(data_dir), "test", mel_norm="clip",
                               manifest_path=str(manifest_path))
    assert len(dataset) == 1
    assert dataset[0]["label"].item() == 0
    empty = AVEmotionDataset(str(data_dir), "train", mel_norm="clip",
                             manifest_path=str(manifest_path))
    assert len(empty) == 0


def test_evaluate_fold_uses_test_split_from_manifest(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    rows = [_row(1, "clip_0", "test", label=0), _row(2, "clip_1", "test", label=4),
            _row(3, "clip_2", "train", label=5)]
    for row in rows:
        np.savez_compressed(
            data_dir / f"{row['clip_id']}.npz",
            frames=np.zeros((8, 112, 112, 3), dtype=np.uint8),
            mel=np.zeros((64, 94), dtype=np.float32),
            waveform=np.zeros(48000, dtype=np.int16),
        )
    manifest_path = tmp_path / "fold.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    model = AVEmotionModel(modality="audio", n_classes=8, pretrained=False)
    config = CheckpointConfig(tag="audio", modality="audio", n_classes=8,
                              class_names=list(EMOTION_TO_IDX), mel_norm="clip")
    checkpoint = tmp_path / "audio.pt"
    save_checkpoint(model, checkpoint, config)

    args = build_parser().parse_args(["--modality", "audio"])
    metrics = evaluate_fold(checkpoint, data_dir, manifest_path, args, "cpu")
    assert metrics["n_test"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert math.isfinite(metrics["macro_f1"])
    assert math.isfinite(metrics["macro_ovr_auc"])


def test_cv_parser_inherits_training_flags():
    args = build_parser().parse_args(["--modality", "fusion", "--fusion", "attention"])
    assert args.n_folds == 6
    assert args.seeds == [0, 1, 2]
    assert args.actor_range == [1, 24]
    assert args.val_actors == 2
    assert args.fusion == "attention"
    assert args.aug_flip_p == 0.0
    assert args.manifest is None


def test_summarize_and_markdown_report_mean_and_std():
    results = pd.DataFrame([
        {"fold": 0, "seed": 0, "test_actors": "1 2", "accuracy": 0.5,
         "macro_f1": 0.4, "macro_ovr_auc": 0.6},
        {"fold": 0, "seed": 1, "test_actors": "1 2", "accuracy": 0.7,
         "macro_f1": 0.6, "macro_ovr_auc": 0.8},
    ])
    summary = summarize(results)
    assert summary["n_folds"] == 1
    assert summary["n_runs"] == 2
    assert summary["overall"]["accuracy"]["mean"] == pytest.approx(0.6)
    assert summary["overall"]["accuracy"]["std"] == pytest.approx(0.1)
    table = markdown_table(summary)
    assert "| Fold | Test actors | Accuracy |" in table
    assert "0.600 +/- 0.100" in table


def test_download_script_parses_actor_selection():
    spec = importlib.util.spec_from_file_location(
        "download_ravdess", ROOT / "scripts" / "download_ravdess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_actors("1-3") == [1, 2, 3]
    assert module.parse_actors("1, 3, 5") == [1, 3, 5]
    assert module.parse_actors("1-2,4") == [1, 2, 4]
    with pytest.raises(ValueError):
        module.parse_actors("8-1")
    with pytest.raises(ValueError):
        module.parse_actors("1,25")
