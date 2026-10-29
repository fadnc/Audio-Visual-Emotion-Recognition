"""
Encoders and fusion heads for audio-visual emotion recognition.

- AudioEncoder: small CNN over log-mel spectrogram -> 128-d embedding
- VideoEncoder: pretrained ResNet18 per frame -> temporal mean pool -> 128-d
- LateFusionHead: concat embeddings -> MLP
- AttentionFusionHead: audio embedding attends over per-frame video embeddings
                       (and vice versa) before concatenation -> MLP
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

from .face_geometry import LANDMARK_FEATURE_DIM

EMBED_DIM = 128


class LandmarkEncoder(nn.Module):
    """Encode normalized face-mesh points and regional geometry per frame."""

    def __init__(self, input_dim=LANDMARK_FEATURE_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, landmarks):
        # landmarks: (B, N, points*xyz + expression geometry)
        per_frame = self.net(landmarks)
        return per_frame.mean(dim=1), per_frame


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, mel):
        # mel: (B, 1, N_MELS, T)
        x = self.net(mel).flatten(1)   # (B, 128)
        return self.proj(x)            # (B, embed_dim)


class VideoEncoder(nn.Module):
    """Per-frame ResNet18 features, temporally mean-pooled. Also returns
    per-frame embeddings for attention fusion."""

    def __init__(self, embed_dim=EMBED_DIM, freeze_backbone=True, pretrained=True,
                 fine_tune_last_block=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc -> (B, 512, 1, 1)
        self.freeze_backbone = freeze_backbone
        self.fine_tune_last_block = fine_tune_last_block
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if fine_tune_last_block:
                # Fine-tune only the final residual block when requested.
                for p in list(self.backbone.children())[-2].parameters():
                    p.requires_grad = True
        self.proj = nn.Linear(512, embed_dim)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            # Keep frozen pretrained BatchNorm statistics fixed; fine-tune the
            # last residual block only, as intended by the model definition.
            self.backbone.eval()
            if self.fine_tune_last_block:
                self.backbone[-2].train(True)
        return self

    def forward(self, frames):
        # frames: (B, N, 3, H, W)
        B, N, C, H, W = frames.shape
        x = frames.view(B * N, C, H, W)
        x = (x - self.imagenet_mean) / self.imagenet_std
        feats = self.backbone(x).flatten(1)          # (B*N, 512)
        feats = self.proj(feats).view(B, N, -1)       # (B, N, embed_dim)
        pooled = feats.mean(dim=1)                    # (B, embed_dim)
        return pooled, feats                           # pooled clip embedding, per-frame embeddings


class LateFusionHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_classes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, audio_emb, video_emb, video_frame_embs=None):
        x = torch.cat([audio_emb, video_emb], dim=-1)
        return self.mlp(x)


class AttentionFusionHead(nn.Module):
    """Audio embedding attends over per-frame video embeddings (single-query
    cross-attention), producing an audio-conditioned video summary. Concatenated
    with the plain audio embedding and classified."""

    def __init__(self, embed_dim=EMBED_DIM, n_classes=8, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, audio_emb, video_emb, video_frame_embs):
        # audio_emb: (B, D) used as query; video_frame_embs: (B, N, D) as key/value
        q = audio_emb.unsqueeze(1)                      # (B, 1, D)
        attended, _ = self.attn(q, video_frame_embs, video_frame_embs)  # (B, 1, D)
        attended = attended.squeeze(1)                    # (B, D)
        x = torch.cat([audio_emb, attended], dim=-1)
        return self.mlp(x)


class WavLMAudioEncoder(nn.Module):
    """WavLM waveform encoder: weighted hidden-layer sum, mean-pooled to 128-d.

    The convolutional feature extractor and the lowest ``freeze_layers``
    transformer layers are frozen and kept in eval mode while training. A
    pre-built ``backbone`` can be injected for tests or custom variants; it must
    expose ``config`` (with ``hidden_size``/``num_hidden_layers``/``to_dict``),
    ``feature_extractor``, ``feature_projection``, ``encoder.layers``, and return
    an output object with ``hidden_states``. Requires the optional ``transformers``
    dependency when no backbone is injected.
    """

    def __init__(self, embed_dim=EMBED_DIM, model_name="microsoft/wavlm-base",
                 freeze_feature_extractor=True, freeze_layers=8, pretrained=True,
                 config=None, backbone=None):
        super().__init__()
        if backbone is None:
            try:
                from transformers import WavLMConfig, WavLMModel
            except ImportError as exc:
                raise ImportError(
                    "The WavLM encoder requires transformers. Install it with "
                    '`pip install -e ".[wavlm]"` (or `pip install transformers`).'
                ) from exc
            if config is not None:
                backbone = WavLMModel(WavLMConfig(**config))
            elif pretrained:
                backbone = WavLMModel.from_pretrained(model_name)
            else:
                raise ValueError(
                    "WavLM without pretrained weights requires an explicit config or backbone."
                )
        self.backbone = backbone
        self.freeze_feature_extractor = bool(freeze_feature_extractor)
        self.freeze_layers = int(freeze_layers)
        if self.freeze_feature_extractor:
            for name, param in self.backbone.named_parameters():
                if name.startswith("feature_extractor") or name.startswith("feature_projection"):
                    param.requires_grad = False
        for index, layer in enumerate(self.backbone.encoder.layers):
            if index < self.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        hidden_size = int(self.backbone.config.hidden_size)
        num_layers = int(self.backbone.config.num_hidden_layers)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.proj = nn.Linear(hidden_size, embed_dim)
        self._config_dict = self.backbone.config.to_dict()

    def config_dict(self) -> dict:
        """Serializable Hugging Face config, stored in the checkpoint sidecar."""
        return dict(self._config_dict)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if self.freeze_feature_extractor:
                self.backbone.feature_extractor.eval()
                if hasattr(self.backbone, "feature_projection"):
                    self.backbone.feature_projection.eval()
            for layer in self.backbone.encoder.layers[:self.freeze_layers]:
                layer.eval()
        return self

    def forward(self, waveform):
        # waveform: (B, samples) mono 16 kHz
        outputs = self.backbone(input_values=waveform, output_hidden_states=True)
        hidden = tuple(outputs.hidden_states[1:])  # drop the feature-projection state
        if len(hidden) != self.layer_weights.numel():
            raise ValueError(
                f"WavLM returned {len(hidden)} layers, expected {self.layer_weights.numel()}."
            )
        stacked = torch.stack([state.mean(dim=1) for state in hidden], dim=0)  # (L, B, H)
        weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        pooled = (stacked * weights).sum(dim=0)                                # (B, H)
        return self.proj(pooled)                                                # (B, embed_dim)


class AVEmotionModel(nn.Module):
    """Unified model. modality: 'audio' | 'video' | 'fusion'.
    fusion_type only used when modality == 'fusion': 'late' | 'attention'.

    Fusion models can optionally drop one modality during training
    (``modality_dropout``), replacing its embedding with a learned missing
    token, and add audio-only/video-only auxiliary heads (``aux_heads``) that
    are returned by ``forward(..., return_aux=True)`` for the training loss.

    ``audio_encoder`` selects the log-mel CNN (``"cnn"``, default) or a WavLM
    waveform encoder (``"wavlm"``, audio-only and fusion models).
    """

    def __init__(self, modality="fusion", fusion_type="late", n_classes=8, embed_dim=EMBED_DIM,
                 pretrained=True, fine_tune_backbone=True, use_landmarks=False,
                 landmark_feature_dim=LANDMARK_FEATURE_DIM, modality_dropout: float = 0.0,
                 aux_heads: bool = False, audio_encoder: str = "cnn",
                 wavlm_model: str = "microsoft/wavlm-base", wavlm_freeze_layers: int = 8,
                 wavlm_config=None, wavlm_backbone=None):
        super().__init__()
        self.modality = modality
        self.fusion_type = fusion_type
        self.use_landmarks = use_landmarks
        self.modality_dropout = float(modality_dropout)
        self.aux_heads = bool(aux_heads)
        self.audio_encoder_type = audio_encoder
        if use_landmarks and modality != "fusion":
            raise ValueError("Face landmarks can only be enabled for a fusion model.")
        if not 0.0 <= self.modality_dropout <= 1.0:
            raise ValueError("modality_dropout must be in [0, 1].")
        if (self.modality_dropout > 0 or self.aux_heads) and modality != "fusion":
            raise ValueError("Modality dropout and auxiliary heads require a fusion model.")
        if audio_encoder not in ("cnn", "wavlm"):
            raise ValueError(f"Unknown audio encoder: {audio_encoder}")
        if audio_encoder == "wavlm" and modality == "video":
            raise ValueError("The WavLM audio encoder requires an audio or fusion model.")

        if modality in ("audio", "fusion"):
            if audio_encoder == "cnn":
                self.audio_encoder = AudioEncoder(embed_dim)
            else:
                self.audio_encoder = WavLMAudioEncoder(
                    embed_dim=embed_dim, model_name=wavlm_model,
                    freeze_layers=wavlm_freeze_layers, pretrained=pretrained,
                    config=wavlm_config, backbone=wavlm_backbone,
                )
        if use_landmarks:
            self.landmark_encoder = LandmarkEncoder(input_dim=landmark_feature_dim,
                                                     embed_dim=embed_dim)
        if modality in ("video", "fusion"):
            self.video_encoder = VideoEncoder(
                embed_dim, pretrained=pretrained, fine_tune_last_block=fine_tune_backbone
            )

        if modality == "audio":
            self.head = nn.Linear(embed_dim, n_classes)
        elif modality == "video":
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, n_classes))
        elif modality == "fusion":
            self.head = (LateFusionHead(embed_dim, n_classes) if fusion_type == "late"
                         else AttentionFusionHead(embed_dim, n_classes))
        else:
            raise ValueError(modality)

        if self.modality_dropout > 0:
            self.audio_missing = nn.Parameter(torch.zeros(embed_dim))
            self.video_missing = nn.Parameter(torch.zeros(embed_dim))
        if self.aux_heads:
            self.aux_audio_head = nn.Linear(embed_dim, n_classes)
            self.aux_video_head = nn.Linear(embed_dim, n_classes)

    def wavlm_config_dict(self):
        """Hugging Face config dict for a WavLM encoder, or None for the CNN."""
        if self.audio_encoder_type != "wavlm":
            return None
        return self.audio_encoder.config_dict()

    def _audio_embedding(self, mel, waveform):
        if self.audio_encoder_type == "wavlm":
            if waveform is None:
                raise ValueError(
                    "This checkpoint uses the WavLM audio encoder and requires a waveform."
                )
            return self.audio_encoder(waveform)
        if mel is None:
            raise ValueError(
                "This checkpoint uses the CNN audio encoder and requires a log-mel input."
            )
        return self.audio_encoder(mel)

    def _modality_dropout_mask(self, batch_size: int, device):
        """Sample per-sample drop masks; at least one modality is always kept."""
        drop_audio = torch.rand(batch_size, device=device) < self.modality_dropout
        drop_video = torch.rand(batch_size, device=device) < self.modality_dropout
        both = drop_audio & drop_video
        keep_video = both & (torch.rand(batch_size, device=device) < 0.5)
        drop_audio = (drop_audio & ~both) | keep_video
        drop_video = (drop_video & ~both) | (both & ~keep_video)
        return drop_audio, drop_video

    def forward(self, mel=None, frames=None, landmarks=None, waveform=None,
                return_aux: bool = False):
        if return_aux and not self.aux_heads:
            raise ValueError("return_aux requires a model built with aux_heads=True.")
        if self.modality == "audio":
            logits = self.head(self._audio_embedding(mel, waveform))
            return {"logits": logits} if return_aux else logits
        if self.modality == "video":
            pooled, _ = self.video_encoder(frames)
            logits = self.head(pooled)
            return {"logits": logits} if return_aux else logits
        # fusion
        audio_emb = self._audio_embedding(mel, waveform)
        video_emb, video_frame_embs = self.video_encoder(frames)
        aux = {}
        if self.aux_heads:
            # Auxiliary losses supervise each encoder before dropout or fusion.
            aux = {"aux_audio": self.aux_audio_head(audio_emb),
                   "aux_video": self.aux_video_head(video_emb)}
        if self.use_landmarks:
            if landmarks is None:
                raise ValueError("This fusion checkpoint requires face landmarks.")
            face_emb, face_frame_embs = self.landmark_encoder(landmarks)
            if self.fusion_type == "attention":
                video_frame_embs = video_frame_embs + face_frame_embs
            else:
                video_emb = video_emb + face_emb
        if self.training and self.modality_dropout > 0:
            drop_audio, drop_video = self._modality_dropout_mask(
                audio_emb.size(0), audio_emb.device
            )
            if drop_audio.any():
                audio_emb = torch.where(drop_audio.unsqueeze(-1),
                                        self.audio_missing.expand_as(audio_emb), audio_emb)
            if drop_video.any():
                video_emb = torch.where(drop_video.unsqueeze(-1),
                                        self.video_missing.expand_as(video_emb), video_emb)
                video_frame_embs = torch.where(
                    drop_video.view(-1, 1, 1),
                    self.video_missing.view(1, 1, -1).expand_as(video_frame_embs),
                    video_frame_embs,
                )
        logits = self.head(audio_emb, video_emb, video_frame_embs)
        if return_aux:
            return {"logits": logits, **aux}
        return logits
