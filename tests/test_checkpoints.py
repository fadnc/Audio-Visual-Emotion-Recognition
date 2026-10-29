"""Checkpoint sidecar save/load, validation, and discovery tests."""
import json
from pathlib import Path

import pytest
import torch

from affect_av.checkpoints import (CheckpointConfig, CheckpointConfigError,
                                   checkpoint_tag, discover_checkpoints, load_checkpoint,
                                   load_config, resolve_mel_stats, save_checkpoint,
                                   sidecar_path)
from affect_av.dataset import EMOTION_TO_IDX
from affect_av.models import AVEmotionModel


def _config(**overrides):
    values = dict(
        tag="audio",
        modality="audio",
        n_classes=len(EMOTION_TO_IDX),
        class_names=list(EMOTION_TO_IDX),
        mel_norm="clip",
    )
    values.update(overrides)
    return CheckpointConfig(**values)


def test_checkpoint_roundtrip(tmp_path):
    model = AVEmotionModel(modality="audio", n_classes=8, pretrained=False).eval()
    path = tmp_path / "audio.pt"
    config = _config()
    save_checkpoint(model, path, config)

    assert sidecar_path(path).is_file()
    assert json.loads(sidecar_path(path).read_text(encoding="utf-8"))["tag"] == "audio"
    loaded, loaded_config = load_checkpoint(path, "cpu")
    assert loaded_config == config

    mel = torch.randn(2, 1, 64, 94)
    with torch.inference_mode():
        expected = model(mel=mel)
        actual = loaded(mel=mel)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_loader_rejects_missing_sidecar(tmp_path):
    model = AVEmotionModel(modality="audio", n_classes=8, pretrained=False)
    path = tmp_path / "orphan.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(CheckpointConfigError):
        load_config(path)
    with pytest.raises(CheckpointConfigError):
        load_checkpoint(path, "cpu")


def test_sidecar_validation():
    base = {"tag": "x", "modality": "audio", "n_classes": 8,
            "class_names": ["a"], "mel_norm": "clip"}
    with pytest.raises(CheckpointConfigError):
        CheckpointConfig.from_dict(base)
    with pytest.raises(CheckpointConfigError):
        CheckpointConfig.from_dict({**base, "class_names": list(EMOTION_TO_IDX),
                                    "audio_encoder": "wavlm"})
    with pytest.raises(CheckpointConfigError):
        CheckpointConfig.from_dict({**base, "class_names": list(EMOTION_TO_IDX),
                                    "modality": "video", "audio_encoder": "wavlm"})


def test_discovery_and_tag(tmp_path):
    model = AVEmotionModel(modality="audio", n_classes=8, pretrained=False)
    nested = tmp_path / "nested"
    save_checkpoint(model, nested / "audio.pt", _config(tag="nested/audio"))
    torch.save(model.state_dict(), tmp_path / "orphan.pt")

    found = discover_checkpoints(tmp_path)
    assert found == [nested / "audio.pt"]
    assert checkpoint_tag(found[0], tmp_path) == "nested/audio"


def test_aux_and_dropout_checkpoint_roundtrip(tmp_path):
    model = AVEmotionModel(modality="fusion", fusion_type="late", n_classes=8,
                           pretrained=False, modality_dropout=0.5, aux_heads=True).eval()
    config = _config(tag="fusion_late_mdrop_aux", modality="fusion", fusion_type="late",
                     modality_dropout=0.5, aux_heads=True)
    path = tmp_path / "fusion.pt"
    save_checkpoint(model, path, config)
    loaded, loaded_config = load_checkpoint(path, "cpu")
    assert loaded_config.modality_dropout == 0.5
    assert hasattr(loaded, "audio_missing")
    assert hasattr(loaded, "aux_audio_head")

    mel = torch.zeros(1, 1, 64, 94)
    frames = torch.zeros(1, 8, 3, 112, 112)
    with torch.inference_mode():
        outputs = loaded(mel=mel, frames=frames, return_aux=True)
    assert tuple(outputs["logits"].shape) == (1, 8)


def test_resolve_mel_stats():
    config = _config(mel_norm="global", mel_stats="data/processed/stats.json")
    assert Path(resolve_mel_stats(config, "/repo")) == Path("/repo") / "data/processed/stats.json"
    assert resolve_mel_stats(_config(mel_norm="clip"), "/repo") is None
