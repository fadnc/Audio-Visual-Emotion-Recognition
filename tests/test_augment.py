"""Per-sample augmentation and dataset integration tests."""
import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from affect_av.augment import (augment_config_from_args, augment_mel_gain,
                               augment_spectrogram, augment_video_frames,
                               augmentation_enabled)
from affect_av.dataset import AVEmotionDataset


def _args(**overrides):
    base = dict(
        aug_resized_crop_scale=0.0, aug_flip_p=0.0, aug_color_jitter=0.0,
        aug_rotation_deg=0.0, aug_gain_db=0.0, aug_noise_std=0.0,
        specaugment=False, specaug_time_masks=2, specaug_time_width=12,
        specaug_freq_masks=2, specaug_freq_width=8,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_augment_config_defaults_off():
    config = augment_config_from_args(_args())
    assert not augmentation_enabled(config)


def test_augment_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        augment_config_from_args(_args(aug_flip_p=1.5))


def test_video_flip_is_consistent_across_frames():
    rng = np.random.default_rng(0)
    frames = rng.random((8, 16, 16, 3), dtype=np.float32)
    config = augment_config_from_args(_args(aug_flip_p=1.0))
    flipped = augment_video_frames(frames, config)
    assert flipped.shape == frames.shape
    assert np.allclose(flipped, frames[:, :, ::-1])


def test_mel_gain_and_specaugment():
    mel = np.ones((64, 94), dtype=np.float32)
    gained = augment_mel_gain(mel, augment_config_from_args(_args(aug_gain_db=6.0)))
    assert gained.shape == mel.shape
    assert np.ptp(gained) == pytest.approx(0.0)
    assert np.all(np.abs(gained - 1.0) <= 6.0 + 1e-5)

    config = augment_config_from_args(_args(
        specaugment=True, specaug_time_masks=1, specaug_time_width=10,
        specaug_freq_masks=1, specaug_freq_width=8,
    ))
    masked = augment_spectrogram(np.ones((64, 94), dtype=np.float32), config)
    assert masked.shape == (64, 94)
    assert np.count_nonzero(masked == 0.0) > 0


def test_dataset_applies_augmentation_to_train_split_only(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    clip_id = "clip_00000"
    frames = np.zeros((8, 112, 112, 3), dtype=np.uint8)
    frames[..., 0] = np.arange(112, dtype=np.uint8)[None, None, :]
    np.savez_compressed(
        data_dir / f"{clip_id}.npz",
        frames=frames,
        mel=np.zeros((64, 94), dtype=np.float32),
        waveform=np.zeros(48000, dtype=np.int16),
    )
    pd.DataFrame([
        {"clip_id": clip_id, "actor": 1, "emotion": "happy", "label": 3,
         "split": "train", "source_path": "raw/1.mp4"},
        {"clip_id": clip_id, "actor": 1, "emotion": "happy", "label": 3,
         "split": "val", "source_path": "raw/1.mp4"},
    ]).to_csv(data_dir / "manifest.csv", index=False)
    (data_dir / "stats.json").write_text('{"mean": 0.0, "std": 1.0}', encoding="utf-8")

    config = augment_config_from_args(_args(aug_flip_p=1.0))
    train_frames = AVEmotionDataset(str(data_dir), "train", augment=config)[0]["frames"]
    val_frames = AVEmotionDataset(str(data_dir), "val", augment=config)[0]["frames"]
    assert tuple(train_frames.shape) == (8, 3, 112, 112)
    assert torch.allclose(train_frames[:, 0], torch.flip(val_frames[:, 0], dims=[-1]))
    assert not torch.allclose(train_frames, val_frames)
