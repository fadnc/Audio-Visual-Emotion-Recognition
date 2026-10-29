"""Smoke tests: every package module imports and the model runs on CPU."""
import importlib

import pytest
import torch

MODULES = [
    "affect_av",
    "affect_av.augment",
    "affect_av.checkpoints",
    "affect_av.dataset",
    "affect_av.face_geometry",
    "affect_av.metrics",
    "affect_av.face_landmarks",
    "affect_av.models",
    "affect_av.train",
    "affect_av.evaluate",
    "affect_av.cross_validate",
    "affect_av.web_app",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_fusion_model_forward_cpu():
    from affect_av.models import AVEmotionModel

    model = AVEmotionModel(
        modality="fusion", fusion_type="attention", n_classes=8, pretrained=False
    ).eval()
    mel = torch.zeros(2, 1, 64, 94)
    frames = torch.zeros(2, 8, 3, 112, 112)
    with torch.inference_mode():
        logits = model(mel=mel, frames=frames)
    assert tuple(logits.shape) == (2, 8)
    assert torch.isfinite(logits).all()


def test_audio_and_video_model_forward_cpu():
    from affect_av.models import AVEmotionModel

    mel = torch.zeros(1, 1, 64, 94)
    frames = torch.zeros(1, 8, 3, 112, 112)
    for modality in ("audio", "video"):
        model = AVEmotionModel(modality=modality, n_classes=8, pretrained=False).eval()
        inputs = {"mel": mel} if modality == "audio" else {"frames": frames}
        with torch.inference_mode():
            logits = model(**inputs)
        assert tuple(logits.shape) == (1, 8)
