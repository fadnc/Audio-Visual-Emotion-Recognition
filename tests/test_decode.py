"""Decoding, mel normalization, and dataset storage tests."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from imageio_ffmpeg import get_ffmpeg_exe

from affect_av.dataset import AVEmotionDataset, decode_clip, normalize_mel


def _make_video(path: Path, extra_args: list[str]) -> bool:
    command = [
        get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=96x64:rate=24", "-t", "3.2",
        *extra_args, str(path),
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


def test_decode_clip_mp4(tmp_path):
    video = tmp_path / "clip.mp4"
    assert _make_video(video, ["-pix_fmt", "yuv420p"])
    frames = decode_clip(str(video), n_frames=8, duration=3.0)
    assert frames is not None
    assert frames.shape == (8, 64, 96, 3)
    assert frames.dtype == np.uint8
    assert frames.max() > 0


def test_decode_clip_webm(tmp_path):
    video = tmp_path / "clip.webm"
    if not _make_video(video, ["-c:v", "libvpx", "-b:v", "200k"]):
        pytest.skip("ffmpeg build cannot encode VP8")
    frames = decode_clip(str(video), n_frames=8, duration=3.0)
    assert frames is not None
    assert frames.shape == (8, 64, 96, 3)


def test_decode_clip_missing_file(tmp_path):
    assert decode_clip(str(tmp_path / "missing.mp4")) is None


def test_clip_normalization_has_zero_mean():
    mel = (np.random.default_rng(0).standard_normal((64, 94)) * 5 + 3).astype(np.float32)
    normalized = normalize_mel(mel, "clip")
    assert abs(float(normalized.mean())) < 1e-4
    assert abs(float(normalized.std()) - 1.0) < 1e-3


def test_global_normalization_uses_stats():
    mel = np.full((64, 94), 5.0, dtype=np.float32)
    normalized = normalize_mel(mel, "global", {"mean": 5.0, "std": 2.0})
    assert np.allclose(normalized, 0.0)


def test_dataset_reads_uint8_frames_and_stats(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    clip_id = "clip_00000"
    frames = np.zeros((8, 112, 112, 3), dtype=np.uint8)
    frames[..., 0] = 255
    np.savez_compressed(
        data_dir / f"{clip_id}.npz",
        frames=frames,
        mel=np.full((64, 94), 5.0, dtype=np.float32),
        waveform=np.full(48000, 16384, dtype=np.int16),
    )
    pd.DataFrame([{
        "clip_id": clip_id, "actor": 1, "emotion": "happy", "label": 3,
        "split": "train", "source_path": "data/raw/Actor_01/fake.mp4",
    }]).to_csv(data_dir / "manifest.csv", index=False)
    (data_dir / "stats.json").write_text(
        json.dumps({"mean": 5.0, "std": 2.0}), encoding="utf-8"
    )

    dataset = AVEmotionDataset(str(data_dir), "train", mel_norm="global")
    sample = dataset[0]
    assert sample["frames"].shape == (8, 3, 112, 112)
    assert torch.allclose(sample["frames"][:, 0], torch.ones(8, 112, 112))
    assert sample["mel"].shape == (1, 64, 94)
    assert torch.allclose(sample["mel"], torch.zeros(1, 64, 94), atol=1e-5)
    assert int(sample["label"]) == 3
    assert "waveform" not in sample

    dataset = AVEmotionDataset(str(data_dir), "train", mel_norm="global",
                              include_waveform=True)
    waveform = dataset[0]["waveform"]
    assert waveform.dtype == torch.float32
    assert tuple(waveform.shape) == (48000,)
    assert torch.allclose(waveform, torch.full((48000,), 0.5))
