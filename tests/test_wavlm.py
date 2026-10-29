"""WavLM encoder tests using an injected tiny backbone (no downloads)."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from affect_av.models import AVEmotionModel, WavLMAudioEncoder


class _FakeConfig:
    hidden_size = 16
    num_hidden_layers = 2

    def to_dict(self):
        return {"hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers}


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.feature_extractor = nn.Linear(1, 1)
        self.feature_projection = nn.Linear(16, 16)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

    def forward(self, input_values, output_hidden_states=True):
        base = input_values.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        states = [base] + [layer(base) for layer in self.encoder.layers]
        return SimpleNamespace(hidden_states=tuple(states))


def test_wavlm_encoder_freezes_lower_layers_and_pools():
    encoder = WavLMAudioEncoder(embed_dim=32, backbone=_FakeBackbone(),
                                freeze_layers=1, pretrained=False).train()
    assert all(not p.requires_grad for p in encoder.backbone.feature_extractor.parameters())
    assert all(not p.requires_grad for p in encoder.backbone.feature_projection.parameters())
    assert all(not p.requires_grad for p in encoder.backbone.encoder.layers[0].parameters())
    assert any(p.requires_grad for p in encoder.backbone.encoder.layers[1].parameters())
    assert encoder.backbone.encoder.layers[0].training is False
    assert encoder.backbone.encoder.layers[1].training is True

    waveform = torch.randn(2, 1600)
    output = encoder(waveform)
    assert tuple(output.shape) == (2, 32)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert encoder.proj.weight.grad is not None
    assert encoder.layer_weights.grad is not None
    assert encoder.backbone.encoder.layers[0].weight.grad is None
    assert encoder.backbone.encoder.layers[1].weight.grad is not None
    assert encoder.config_dict()["hidden_size"] == 16


def test_audio_model_with_wavlm_backbone():
    model = AVEmotionModel(modality="audio", n_classes=8, pretrained=False,
                           audio_encoder="wavlm", wavlm_backbone=_FakeBackbone()).eval()
    waveform = torch.randn(2, 1600)
    with torch.inference_mode():
        logits = model(waveform=waveform)
    assert tuple(logits.shape) == (2, 8)
    assert model.wavlm_config_dict()["num_hidden_layers"] == 2


def test_fusion_model_with_wavlm_backbone():
    model = AVEmotionModel(modality="fusion", fusion_type="late", n_classes=8,
                           pretrained=False, audio_encoder="wavlm",
                           wavlm_backbone=_FakeBackbone()).eval()
    frames = torch.zeros(1, 8, 3, 112, 112)
    with pytest.raises(ValueError):
        model(mel=torch.zeros(1, 1, 64, 94), frames=frames)
    with torch.inference_mode():
        logits = model(waveform=torch.randn(1, 1600), frames=frames)
    assert tuple(logits.shape) == (1, 8)


def test_wavlm_missing_dependency_message():
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformers"):
            WavLMAudioEncoder(pretrained=True)
    else:
        pytest.skip("transformers is installed; the missing-dependency path is not exercised")
