"""
RAVDESS audio-visual preprocessing and shared decoding utilities.

Scans data/raw/Actor_XX/*.mp4 (full audio-visual clips, modality code 01),
extracts:
  - N evenly-spaced RGB frames from the FIRST ``AUDIO_DURATION`` seconds of the
    video, decoded sequentially with the imageio-ffmpeg binary (no frame seeking)
  - the audio waveform from the same window, plus its log-mel spectrogram

Frames are stored as uint8 (N, H, W, 3). The mel is stored unnormalized; global
mean/std are written to ``stats.json`` from the training split and applied by
``AVEmotionDataset`` / the web app (`--mel_norm global`, the default). The legacy
per-clip normalization is available with `--mel_norm clip`.

The manifest stores ``source_path`` so downstream scripts never align raw and
processed clips by index.

Usage:
    affect-preprocess --raw_dir data/raw --out_dir data/processed
"""
import argparse
import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from imageio_ffmpeg import get_ffmpeg_exe
from torch.utils.data import Dataset
from tqdm import tqdm

from .augment import (augment_mel_gain, augment_spectrogram, augment_video_frames,
                      augmentation_enabled)
from .face_geometry import select_landmark_features

# --- config -----------------------------------------------------------
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
EMOTION_TO_IDX = {name: i for i, name in enumerate(sorted(set(EMOTIONS.values())))}

N_FRAMES = 8
FRAME_SIZE = 112          # square crop, matches small ResNet input well
AUDIO_SR = 16000
N_MELS = 64
AUDIO_DURATION = 3.0      # seconds, shared window for audio and video
AUDIO_FRAMES = int(AUDIO_DURATION * AUDIO_SR / 512)  # hop_length=512
N_FFT = 2048
HOP_LENGTH = 512
MEL_NORM_MODES = ("global", "clip")
MEL_FLOOR_DB = -100.0


def log_mel_spectrogram(y, sr=AUDIO_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Return an unnormalized log-mel spectrogram in dB re full scale.

    Uses a centered Hann-window STFT and triangular HTK mel filters. No
    per-clip mean/std (or max) subtraction happens here; loudness is preserved
    and global statistics are applied later.
    """
    y = np.asarray(y, dtype=np.float32)
    padded = np.pad(y, (n_fft // 2, n_fft // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(windows * window, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).T.astype(np.float32)

    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_hz = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    filters = np.zeros((n_mels, len(fft_hz)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz_points[index:index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum((fft_hz - left) / max(center - left, 1e-12),
                       (right - fft_hz) / max(right - center, 1e-12)),
        )
        filters[index] *= 2.0 / max(right - left, 1e-12)
    mel_power = filters @ power
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    return np.maximum(mel_db, MEL_FLOOR_DB).astype(np.float32)


def _stats_pair(stats) -> tuple[float, float]:
    if isinstance(stats, dict):
        return float(stats["mean"]), float(stats["std"])
    mean, std = stats
    return float(mean), float(std)


def normalize_mel(mel, mode: str = "global", stats=None) -> np.ndarray:
    """Apply global (training-split) or legacy per-clip mel normalization."""
    mel = np.asarray(mel, dtype=np.float32)
    if mode == "global":
        if stats is None:
            raise ValueError("Global mel normalization requires mean/std statistics.")
        mean, std = _stats_pair(stats)
        return ((mel - mean) / max(std, 1e-6)).astype(np.float32)
    if mode == "clip":
        shifted = np.maximum(mel - mel.max(), -80.0)
        return ((shifted - shifted.mean()) / (shifted.std() + 1e-6)).astype(np.float32)
    raise ValueError(f"Unknown mel normalization mode: {mode}")


def load_mel_stats(path) -> dict:
    """Load the training-split mel mean/std written by ``affect-preprocess``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Mel stats not found: {path}. Run affect-preprocess, or pass --mel_norm clip."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    mean, std = float(data["mean"]), float(data["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid mel statistics in {path}: {data}")
    return {**data, "mean": mean, "std": std}


def parse_filename(path):
    """RAVDESS filename: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4"""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("-")
    if len(parts) != 7:
        return None
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "01":  # keep only full audio-visual clips
        return None
    return {
        "emotion_code": emotion,
        "emotion": EMOTIONS.get(emotion),
        "actor": int(actor),
    }


def _read_exact(stream, size: int) -> Optional[bytes]:
    chunks, remaining = [], size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_ppm_frame(stream) -> Optional[np.ndarray]:
    """Read one self-describing P6 PPM frame from an ffmpeg image2pipe stream."""
    tokens = []
    while len(tokens) < 4:
        byte = stream.read(1)
        if not byte:
            return None
        if byte.isspace():
            continue
        if byte == b"#":
            while True:
                byte = stream.read(1)
                if not byte or byte in (b"\n", b"\r"):
                    break
            continue
        token = bytearray(byte)
        while True:
            byte = stream.read(1)
            if not byte or byte.isspace():
                break
            token += byte
        tokens.append(token.decode("ascii"))
    if tokens[0] != "P6":
        raise ValueError(f"Unexpected ffmpeg frame format: {tokens[0]}")
    width, height = int(tokens[1]), int(tokens[2])
    payload = _read_exact(stream, width * height * 3)
    if payload is None:
        return None
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()


def decode_clip(path, n_frames: int = N_FRAMES,
                duration: float = AUDIO_DURATION) -> Optional[np.ndarray]:
    """Decode ``n_frames`` RGB frames from the first ``duration`` seconds.

    Uses sequential ffmpeg decoding (PPM over a pipe) with an ``fps`` filter, so
    no frame seeking or frame-count metadata is required. This is what makes
    browser WebM clips (which often report zero frames) work. Frames are sampled
    evenly across the same window used for audio. Short clips repeat their last
    frame. Returns uint8 (N, H, W, 3), or None if nothing could be decoded.
    """
    fps = n_frames / duration
    command = [
        get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path), "-t", f"{duration:.3f}", "-an",
        "-vf", f"fps={fps:.6f}",
        "-f", "image2pipe", "-vcodec", "ppm", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    try:
        while len(frames) < n_frames:
            frame = _read_ppm_frame(process.stdout)
            if frame is None:
                break
            frames.append(frame)
    finally:
        if process.poll() is None:
            process.kill()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.read()
        process.wait()
    if not frames:
        return None
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames).astype(np.uint8)


def center_crop_frames(frames: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Center-crop each RGB frame to a square and resize to ``size``."""
    crops = []
    for frame in frames:
        height, width = frame.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        square = frame[top:top + side, left:left + side]
        crops.append(cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(crops).astype(np.uint8)


def extract_frames(video_path, n_frames: int = N_FRAMES, size: int = FRAME_SIZE,
                   duration: float = AUDIO_DURATION) -> Optional[np.ndarray]:
    frames = decode_clip(video_path, n_frames=n_frames, duration=duration)
    if frames is None:
        return None
    return center_crop_frames(frames, size=size)


def extract_audio_waveform(video_path, duration: float = AUDIO_DURATION):
    """Decode the first ``duration`` seconds of mono 16 kHz audio."""
    try:
        decoded = subprocess.run(
            [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", str(video_path), "-t", f"{duration:.3f}", "-vn", "-ac", "1",
             "-ar", str(AUDIO_SR), "-f", "f32le", "pipe:1"],
            capture_output=True,
            check=True,
        )
        y = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    except Exception as exc:
        print(f"\n[audio error] {video_path}: {exc}")
        return None
    if y.size == 0:
        return None
    target_len = int(duration * AUDIO_SR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32, copy=False)


def extract_audio_features(video_path, duration: float = AUDIO_DURATION):
    """Return the unnormalized log-mel of the first ``duration`` seconds."""
    waveform = extract_audio_waveform(video_path, duration=duration)
    if waveform is None:
        return None
    return log_mel_spectrogram(waveform)


def build_manifest(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)):
        meta = parse_filename(path)
        if meta is None or meta["emotion"] is None:
            continue
        rows.append({"path": path, **meta})
    df = pd.DataFrame(rows)
    return df


def assign_splits(df, val_actors=(7,), test_actors=(8,)):
    """Speaker-independent split: held-out actors for val/test."""
    def split_for(actor):
        if actor in val_actors:
            return "val"
        if actor in test_actors:
            return "test"
        return "train"
    df["split"] = df["actor"].apply(split_for)
    return df


def _mel_running_stats(sums: list[float], square_sums: list[float], count: list[int],
                       mel: np.ndarray) -> None:
    sums[0] += float(mel.sum(dtype=np.float64))
    square_sums[0] += float(np.square(mel, dtype=np.float64).sum())
    count[0] += int(mel.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_actors", nargs="+", type=int, default=[7])
    ap.add_argument("--test_actors", nargs="+", type=int, default=[8])
    ap.add_argument("--mel_norm", choices=MEL_NORM_MODES, default="global",
                    help="Normalization used later by datasets: global stats (default) "
                         "or the legacy per-clip z-score.")
    ap.add_argument("--mel_stats", default=None,
                    help="Output path for stats.json (default: <out_dir>/stats.json).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = Path(args.mel_stats or Path(args.out_dir) / "stats.json")
    df = build_manifest(args.raw_dir)
    if len(df) == 0:
        raise SystemExit(
            f"No clips found under {args.raw_dir} (searched recursively) with modality code 01. "
            "Check you downloaded the full audio-visual RAVDESS zips."
        )
    df = assign_splits(df, tuple(args.val_actors), tuple(args.test_actors))

    manifest_rows = []
    frame_fail, audio_fail, shown = 0, 0, 0
    sums, square_sums, count = [0.0], [0.0], [0]
    for i, row in tqdm(df.iterrows(), total=len(df), desc="extracting"):
        clip_id = f"clip_{i:05d}"
        frames = extract_frames(row["path"])
        waveform = extract_audio_waveform(row["path"])
        mel = log_mel_spectrogram(waveform) if waveform is not None else None
        if frames is None or mel is None:
            if frames is None:
                frame_fail += 1
            if mel is None:
                audio_fail += 1
            if shown < 3:
                print(f"\n[skip] {row['path']} -> frames_ok={frames is not None} "
                      f"audio_ok={mel is not None}")
                shown += 1
            continue
        if row["split"] == "train":
            _mel_running_stats(sums, square_sums, count, mel)
        np.savez_compressed(
            os.path.join(args.out_dir, f"{clip_id}.npz"),
            frames=frames,
            mel=mel,
            waveform=np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16),
        )
        manifest_rows.append({
            "clip_id": clip_id,
            "actor": row["actor"],
            "emotion": row["emotion"],
            "label": EMOTION_TO_IDX[row["emotion"]],
            "split": row["split"],
            "source_path": row["path"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"Processed {len(manifest)} clips -> {args.out_dir}/manifest.csv")
    if len(manifest):
        print(manifest["split"].value_counts())
        print(manifest["emotion"].value_counts())

    if count[0] == 0:
        raise SystemExit(
            f"No training clips were processed; cannot compute global mel statistics. "
            f"frame_extraction_failures={frame_fail} audio_extraction_failures={audio_fail}."
        )
    mean = sums[0] / count[0]
    variance = max(square_sums[0] / count[0] - mean ** 2, 1e-8)
    stats = {
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "mode": args.mel_norm,
        "source_split": "train",
        "n_values": count[0],
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "sample_rate": AUDIO_SR,
        "duration": AUDIO_DURATION,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Global mel stats (train split) -> {stats_path}: "
          f"mean={mean:.4f} std={stats['std']:.4f} over {count[0]:,} values")


class AVEmotionDataset(Dataset):
    """Clip dataset with global (default) or per-clip mel normalization."""

    def __init__(self, data_dir, split, include_landmarks: bool = False,
                 landmark_dir: str = "data/landmarks",
                 landmark_feature_mode: str = "geometry",
                 face_frame_dir: Optional[str] = None,
                 mel_norm: str = "global",
                 mel_stats_path: Optional[str] = None,
                 augment: Optional[dict] = None,
                 include_waveform: bool = False,
                 manifest_path: Optional[str] = None):
        if mel_norm not in MEL_NORM_MODES:
            raise ValueError(f"Unknown mel_norm: {mel_norm}")
        self.data_dir = data_dir
        self.split = split
        self.include_landmarks = include_landmarks
        self.landmark_dir = landmark_dir
        self.landmark_feature_mode = landmark_feature_mode
        self.face_frame_dir = face_frame_dir
        self.mel_norm = mel_norm
        self.include_waveform = include_waveform
        self.manifest_path = manifest_path
        self.mel_stats = (load_mel_stats(mel_stats_path or os.path.join(data_dir, "stats.json"))
                          if mel_norm == "global" else None)
        self.augment = augment or {}
        self.augment_enabled = split == "train" and augmentation_enabled(self.augment)
        manifest = pd.read_csv(manifest_path or os.path.join(data_dir, "manifest.csv"))
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def _load_frames(self, clip_id: str, npz) -> np.ndarray:
        if self.face_frame_dir:
            array = np.load(os.path.join(self.face_frame_dir, f"{clip_id}.npy"))
        else:
            array = npz["frames"]
        array = np.asarray(array)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32, copy=False)
        return array

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        clip_id = str(row["clip_id"])
        with np.load(os.path.join(self.data_dir, f"{clip_id}.npz")) as npz:
            frames = self._load_frames(clip_id, npz)
            raw_mel = np.asarray(npz["mel"])
            waveform = (npz["waveform"].astype(np.float32) / 32768.0
                        if self.include_waveform else None)
        if self.augment_enabled:
            frames = augment_video_frames(frames, self.augment)
            raw_mel = augment_mel_gain(raw_mel, self.augment)
        mel = normalize_mel(raw_mel, self.mel_norm, self.mel_stats)
        if self.augment_enabled:
            mel = augment_spectrogram(mel, self.augment)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        result = {
            "frames": torch.from_numpy(frames).permute(0, 3, 1, 2),  # (N, 3, H, W)
            "mel": torch.from_numpy(mel).unsqueeze(0),               # (1, N_MELS, T)
            "label": label,
        }
        if waveform is not None:
            result["waveform"] = torch.from_numpy(waveform)          # (samples,)
        if self.include_landmarks:
            landmark_path = os.path.join(self.landmark_dir, f"{clip_id}.npz")
            with np.load(landmark_path) as landmarks:
                features = select_landmark_features(landmarks["features"],
                                                    self.landmark_feature_mode)
                features[:, -1] = landmarks["valid_frames"].astype(np.float32)
                result["landmarks"] = torch.from_numpy(features)
        return result


if __name__ == "__main__":
    main()
