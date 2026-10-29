"""Per-sample training augmentation applied inside ``AVEmotionDataset``.

Geometric transforms (crop, rotation, flip) and color jitter use one random draw
per clip, so all frames of a clip receive the same transform. Audio gain is
applied to the raw log-mel before normalization; SpecAugment and noise are
applied after normalization. Every option defaults to off.
"""
from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np

AUGMENT_DEFAULTS: dict[str, Any] = {
    "resized_crop_scale": 0.0,
    "flip_p": 0.0,
    "color_jitter": 0.0,
    "rotation_deg": 0.0,
    "gain_db": 0.0,
    "noise_std": 0.0,
    "specaugment": False,
    "specaug_time_masks": 2,
    "specaug_time_width": 12,
    "specaug_freq_masks": 2,
    "specaug_freq_width": 8,
}


def augment_config_from_args(args) -> dict:
    """Build an augmentation config dict from parsed argparse options."""
    config = {
        "resized_crop_scale": float(args.aug_resized_crop_scale),
        "flip_p": float(args.aug_flip_p),
        "color_jitter": float(args.aug_color_jitter),
        "rotation_deg": float(args.aug_rotation_deg),
        "gain_db": float(args.aug_gain_db),
        "noise_std": float(args.aug_noise_std),
        "specaugment": bool(args.specaugment),
        "specaug_time_masks": int(args.specaug_time_masks),
        "specaug_time_width": int(args.specaug_time_width),
        "specaug_freq_masks": int(args.specaug_freq_masks),
        "specaug_freq_width": int(args.specaug_freq_width),
    }
    if not 0.0 <= config["resized_crop_scale"] <= 1.0:
        raise ValueError("--aug_resized_crop_scale must be in [0, 1].")
    if not 0.0 <= config["flip_p"] <= 1.0:
        raise ValueError("--aug_flip_p must be in [0, 1].")
    for key in ("color_jitter", "rotation_deg", "gain_db", "noise_std"):
        if config[key] < 0:
            raise ValueError(f"--aug_{key} must be >= 0.")
    return config


def augmentation_enabled(config: dict | None) -> bool:
    if not config:
        return False
    return (
        config.get("resized_crop_scale", 0.0) > 0
        or config.get("flip_p", 0.0) > 0
        or config.get("color_jitter", 0.0) > 0
        or config.get("rotation_deg", 0.0) > 0
        or config.get("gain_db", 0.0) > 0
        or config.get("noise_std", 0.0) > 0
        or bool(config.get("specaugment", False))
    )


def augment_video_frames(frames: np.ndarray, config: dict) -> np.ndarray:
    """Apply one consistent random video transform to all frames of a clip.

    ``frames`` is float32 (N, H, W, 3) in [0, 1]. Returns the same layout.
    """
    frames = np.asarray(frames, dtype=np.float32)
    _, height, width, _ = frames.shape

    scale = float(config.get("resized_crop_scale", 0.0))
    if scale > 0:
        factor = random.uniform(scale, 1.0)
        crop_h = max(1, min(height, int(round(height * math.sqrt(factor)))))
        crop_w = max(1, min(width, int(round(width * math.sqrt(factor)))))
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        frames = np.stack([
            cv2.resize(frame[top:top + crop_h, left:left + crop_w], (width, height),
                       interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ])

    rotation = float(config.get("rotation_deg", 0.0))
    if rotation > 0:
        angle = random.uniform(-rotation, rotation)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        frames = np.stack([
            cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
            for frame in frames
        ])

    flip_p = float(config.get("flip_p", 0.0))
    if flip_p > 0 and random.random() < flip_p:
        frames = frames[:, :, ::-1]

    jitter = float(config.get("color_jitter", 0.0))
    if jitter > 0:
        brightness = random.uniform(1.0 - jitter, 1.0 + jitter)
        contrast = random.uniform(1.0 - jitter, 1.0 + jitter)
        saturation = random.uniform(1.0 - jitter, 1.0 + jitter)
        frames = np.clip(frames * brightness, 0.0, 1.0)
        mean = frames.mean(axis=(1, 2, 3), keepdims=True)
        frames = np.clip((frames - mean) * contrast + mean, 0.0, 1.0)
        gray = frames.mean(axis=3, keepdims=True)
        frames = np.clip((frames - gray) * saturation + gray, 0.0, 1.0)

    return np.ascontiguousarray(frames, dtype=np.float32)


def augment_mel_gain(mel: np.ndarray, config: dict) -> np.ndarray:
    """Add a random gain (in dB) to the raw log-mel before normalization."""
    gain = float(config.get("gain_db", 0.0))
    mel = np.asarray(mel, dtype=np.float32)
    if gain <= 0:
        return mel
    return (mel + random.uniform(-gain, gain)).astype(np.float32)


def augment_spectrogram(mel: np.ndarray, config: dict) -> np.ndarray:
    """Apply SpecAugment masks and additive noise to a normalized log-mel."""
    mel = np.asarray(mel, dtype=np.float32).copy()
    if config.get("specaugment", False):
        n_mels, n_time = mel.shape
        for _ in range(int(config.get("specaug_time_masks", 0))):
            width = random.randint(1, max(1, int(config.get("specaug_time_width", 1))))
            start = random.randint(0, max(n_time - width, 0))
            mel[:, start:start + width] = 0.0
        for _ in range(int(config.get("specaug_freq_masks", 0))):
            width = random.randint(1, max(1, int(config.get("specaug_freq_width", 1))))
            start = random.randint(0, max(n_mels - width, 0))
            mel[start:start + width, :] = 0.0
    noise = float(config.get("noise_std", 0.0))
    if noise > 0:
        mel += np.random.normal(0.0, noise, size=mel.shape).astype(np.float32)
    return mel
