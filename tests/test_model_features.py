"""Modality dropout and auxiliary head tests."""
import pytest
import torch

from affect_av.models import AVEmotionModel


def _model(**kwargs):
    return AVEmotionModel(modality="fusion", fusion_type="attention", n_classes=8,
                          pretrained=False, **kwargs).eval()


def _inputs(batch_size=2):
    return (torch.zeros(batch_size, 1, 64, 94),
            torch.zeros(batch_size, 8, 3, 112, 112))


def _spy_head(model):
    captured = {}

    def forward(audio_emb, video_emb, video_frame_embs):
        captured["audio"] = audio_emb.detach().clone()
        captured["video"] = video_emb.detach().clone()
        captured["frames"] = video_frame_embs.detach().clone()
        return torch.zeros(audio_emb.size(0), 8)

    model.head.forward = forward
    return captured


def test_default_model_keeps_original_state_dict():
    keys = AVEmotionModel(modality="fusion", fusion_type="late", n_classes=8,
                          pretrained=False).state_dict().keys()
    assert not any("missing" in key or "aux_" in key for key in keys)


def test_modality_dropout_never_drops_both():
    model = _model(modality_dropout=1.0).train()
    for _ in range(16):
        drop_audio, drop_video = model._modality_dropout_mask(64, torch.device("cpu"))
        assert drop_audio.dtype == torch.bool and drop_video.dtype == torch.bool
        assert not (drop_audio & drop_video).any()


def test_modality_dropout_uses_learned_tokens():
    model = _model(modality_dropout=0.5).train()
    with torch.no_grad():
        model.audio_missing.fill_(3.0)
        model.video_missing.fill_(-2.0)
    captured = _spy_head(model)
    mel, frames = _inputs()

    model._modality_dropout_mask = lambda batch_size, device: (
        torch.ones(batch_size, dtype=torch.bool), torch.zeros(batch_size, dtype=torch.bool))
    with torch.no_grad():
        model(mel=mel, frames=frames)
    assert torch.allclose(captured["audio"], torch.full((2, 128), 3.0))

    model._modality_dropout_mask = lambda batch_size, device: (
        torch.zeros(batch_size, dtype=torch.bool), torch.ones(batch_size, dtype=torch.bool))
    with torch.no_grad():
        model(mel=mel, frames=frames)
    assert torch.allclose(captured["video"], torch.full((2, 128), -2.0))
    assert torch.allclose(captured["frames"], torch.full((2, 8, 128), -2.0))


def test_modality_dropout_disabled_in_eval():
    model = _model(modality_dropout=0.5)
    with torch.no_grad():
        model.audio_missing.fill_(3.0)
    captured = _spy_head(model)
    model._modality_dropout_mask = lambda batch_size, device: (
        torch.ones(batch_size, dtype=torch.bool), torch.zeros(batch_size, dtype=torch.bool))
    mel, frames = _inputs()
    with torch.no_grad():
        model(mel=mel, frames=frames)
    assert not torch.allclose(captured["audio"], torch.full((2, 128), 3.0))


def test_aux_heads_return_logits():
    model = _model(aux_heads=True)
    mel, frames = _inputs()
    with torch.inference_mode():
        outputs = model(mel=mel, frames=frames, return_aux=True)
    assert set(outputs) == {"logits", "aux_audio", "aux_video"}
    for value in outputs.values():
        assert tuple(value.shape) == (2, 8)


def test_aux_heads_are_opt_in():
    model = _model()
    mel, frames = _inputs(batch_size=1)
    with torch.inference_mode():
        logits = model(mel=mel, frames=frames)
    assert tuple(logits.shape) == (1, 8)
    with pytest.raises(ValueError):
        model(mel=mel, frames=frames, return_aux=True)
