"""Config sidecars for checkpoints.

Each ``<name>.pt`` written by ``affect-train`` gets a ``<name>.json`` sidecar
describing everything needed to rebuild the model and its inputs: modality,
fusion type, audio encoder, face cropping, landmark settings, mel normalization,
class order, and WavLM configuration. Inference and evaluation load the model
from that sidecar instead of hard-coded tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Optional

import torch

from .dataset import EMOTION_TO_IDX, MEL_NORM_MODES
from .face_geometry import LANDMARK_FEATURE_DIM, LANDMARK_GEOMETRY_DIM
from .models import AVEmotionModel

FORMAT_VERSION = 1
MODALITIES = ("audio", "video", "fusion")
FUSION_TYPES = ("late", "attention")
AUDIO_ENCODERS = ("cnn", "wavlm")
ROOT = Path(__file__).resolve().parents[1]


class CheckpointConfigError(RuntimeError):
    """Raised when a checkpoint sidecar is missing or inconsistent."""


@dataclass
class CheckpointConfig:
    tag: str
    modality: str
    n_classes: int
    class_names: list[str]
    fusion_type: str = "late"
    audio_encoder: str = "cnn"
    embed_dim: int = 128
    use_landmarks: bool = False
    landmark_feature_mode: str = "geometry"
    landmark_feature_dim: int = LANDMARK_GEOMETRY_DIM
    face_crop: bool = False
    mel_norm: str = "global"
    mel_stats: Optional[str] = None
    modality_dropout: float = 0.0
    aux_heads: bool = False
    wavlm_model: Optional[str] = None
    wavlm_freeze_layers: int = 8
    wavlm_config: Optional[dict] = None
    created_utc: str = ""
    git_commit: Optional[str] = None
    torch_version: str = ""
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointConfig":
        required = ("tag", "modality", "n_classes", "class_names")
        missing = [key for key in required if key not in data]
        if missing:
            raise CheckpointConfigError(f"Checkpoint sidecar is missing fields: {missing}")
        known = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in data.items() if key in known})
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.format_version, int) or self.format_version > FORMAT_VERSION:
            raise CheckpointConfigError(
                f"Unsupported checkpoint sidecar format: {self.format_version}"
            )
        if self.modality not in MODALITIES:
            raise CheckpointConfigError(f"Unknown modality: {self.modality}")
        if self.fusion_type not in FUSION_TYPES:
            raise CheckpointConfigError(f"Unknown fusion type: {self.fusion_type}")
        if self.audio_encoder not in AUDIO_ENCODERS:
            raise CheckpointConfigError(f"Unknown audio encoder: {self.audio_encoder}")
        if self.mel_norm not in MEL_NORM_MODES:
            raise CheckpointConfigError(f"Unknown mel normalization: {self.mel_norm}")
        if not isinstance(self.class_names, list) or not all(
            isinstance(name, str) for name in self.class_names
        ):
            raise CheckpointConfigError("class_names must be a list of strings")
        if len(self.class_names) != self.n_classes:
            raise CheckpointConfigError(
                f"n_classes={self.n_classes} but {len(self.class_names)} class names were stored"
            )
        if self.modality != "fusion" and (
            self.use_landmarks or self.aux_heads or self.modality_dropout > 0
        ):
            raise CheckpointConfigError(
                "Landmarks, modality dropout, and auxiliary heads require a fusion checkpoint"
            )
        if self.audio_encoder == "wavlm" and self.modality == "video":
            raise CheckpointConfigError("A video-only checkpoint cannot use the WavLM encoder")
        if self.audio_encoder == "wavlm" and self.wavlm_config is None:
            raise CheckpointConfigError(
                "A WavLM checkpoint must store its Hugging Face config in the sidecar"
            )
        if self.mel_norm == "global" and not self.mel_stats:
            raise CheckpointConfigError(
                "Global mel normalization requires the sidecar mel_stats path"
            )


def sidecar_path(checkpoint_path) -> Path:
    """Return the JSON sidecar path for a ``.pt`` checkpoint."""
    return Path(checkpoint_path).with_suffix(".json")


def checkpoint_tag(checkpoint_path, root) -> str:
    """Stable UI/CLI identifier: checkpoint path relative to ``root`` without suffix."""
    return Path(checkpoint_path).relative_to(root).with_suffix("").as_posix()


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True, timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def config_from_training(args, tag: str, model: AVEmotionModel) -> CheckpointConfig:
    """Describe a model trained by ``affect-train`` in a serializable config."""
    audio_encoder = getattr(model, "audio_encoder_type", "cnn")
    mel_stats = None
    if args.mel_norm == "global":
        stats_path = (Path(args.mel_stats) if args.mel_stats
                      else Path(args.data_dir) / "stats.json").resolve()
        try:
            mel_stats = stats_path.relative_to(ROOT).as_posix()
        except ValueError:
            mel_stats = stats_path.as_posix()
    landmark_dim = (LANDMARK_GEOMETRY_DIM if args.landmark_feature_mode == "geometry"
                    else LANDMARK_FEATURE_DIM)
    return CheckpointConfig(
        tag=tag,
        modality=args.modality,
        fusion_type=args.fusion,
        audio_encoder=audio_encoder,
        n_classes=len(EMOTION_TO_IDX),
        class_names=list(EMOTION_TO_IDX),
        use_landmarks=bool(args.use_landmarks),
        landmark_feature_mode=args.landmark_feature_mode,
        landmark_feature_dim=landmark_dim,
        face_crop=bool(args.face_frame_dir),
        mel_norm=args.mel_norm,
        mel_stats=mel_stats,
        modality_dropout=float(args.modality_dropout),
        aux_heads=bool(args.aux_heads),
        wavlm_model=args.wavlm_model if audio_encoder == "wavlm" else None,
        wavlm_freeze_layers=int(args.wavlm_freeze_layers),
        wavlm_config=model.wavlm_config_dict(),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        torch_version=torch.__version__,
    )


def save_checkpoint(model: AVEmotionModel, checkpoint_path, config: CheckpointConfig) -> None:
    """Save weights plus a JSON sidecar, replacing both atomically."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_path(checkpoint_path)
    temp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temp_sidecar.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    torch.save(model.state_dict(), checkpoint_path)
    os.replace(temp_sidecar, sidecar)


def load_config(checkpoint_path) -> CheckpointConfig:
    sidecar = sidecar_path(checkpoint_path)
    if not sidecar.is_file():
        raise CheckpointConfigError(
            f"No checkpoint sidecar found: {sidecar}. Checkpoints written before the "
            "config-sidecar format are unsupported; retrain with affect-train so a "
            "<name>.json file is written next to <name>.pt."
        )
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointConfigError(f"Invalid checkpoint sidecar {sidecar}: {exc}") from exc
    return CheckpointConfig.from_dict(data)


def model_from_config(config: CheckpointConfig) -> AVEmotionModel:
    """Build an uninitialized model matching a sidecar config."""
    return AVEmotionModel(
        modality=config.modality,
        fusion_type=config.fusion_type,
        n_classes=config.n_classes,
        embed_dim=config.embed_dim,
        pretrained=False,
        use_landmarks=config.use_landmarks,
        landmark_feature_dim=config.landmark_feature_dim,
        audio_encoder=config.audio_encoder,
        wavlm_model=config.wavlm_model or "microsoft/wavlm-base",
        wavlm_freeze_layers=config.wavlm_freeze_layers,
        wavlm_config=config.wavlm_config,
        modality_dropout=config.modality_dropout,
        aux_heads=config.aux_heads,
    )


def load_checkpoint(checkpoint_path, device) -> tuple[AVEmotionModel, CheckpointConfig]:
    """Load weights and return an eval-mode model built from its sidecar."""
    config = load_config(checkpoint_path)
    model = model_from_config(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise CheckpointConfigError(
            f"Checkpoint {checkpoint_path} does not match its sidecar: {exc}"
        ) from exc
    model.eval()
    return model, config


def resolve_mel_stats(config: CheckpointConfig, root=ROOT) -> Optional[str]:
    """Return the absolute mel-stats path, or None for per-clip normalization."""
    if config.mel_norm != "global":
        return None
    if not config.mel_stats:
        raise CheckpointConfigError("Global mel normalization requires a mel_stats path.")
    path = Path(config.mel_stats)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def discover_checkpoints(directory) -> list[Path]:
    """List ``.pt`` files that have a sidecar, recursively."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.rglob("*.pt"))
            if sidecar_path(path).is_file()]
