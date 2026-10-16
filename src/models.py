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

try:
    from .face_geometry import LANDMARK_FEATURE_DIM
except ImportError:
    from face_geometry import LANDMARK_FEATURE_DIM

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


class AVEmotionModel(nn.Module):
    """Unified model. modality: 'audio' | 'video' | 'fusion'.
    fusion_type only used when modality == 'fusion': 'late' | 'attention'."""

    def __init__(self, modality="fusion", fusion_type="late", n_classes=8, embed_dim=EMBED_DIM,
                 pretrained=True, fine_tune_backbone=True, use_landmarks=False,
                 landmark_feature_dim=LANDMARK_FEATURE_DIM):
        super().__init__()
        self.modality = modality
        self.fusion_type = fusion_type
        self.use_landmarks = use_landmarks
        if use_landmarks and modality != "fusion":
            raise ValueError("Face landmarks can only be enabled for a fusion model.")

        if modality in ("audio", "fusion"):
            self.audio_encoder = AudioEncoder(embed_dim)
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

    def forward(self, mel=None, frames=None, landmarks=None):
        if self.modality == "audio":
            emb = self.audio_encoder(mel)
            return self.head(emb)
        if self.modality == "video":
            pooled, _ = self.video_encoder(frames)
            return self.head(pooled)
        # fusion
        audio_emb = self.audio_encoder(mel)
        video_emb, video_frame_embs = self.video_encoder(frames)
        if self.use_landmarks:
            if landmarks is None:
                raise ValueError("This fusion checkpoint requires face landmarks.")
            face_emb, face_frame_embs = self.landmark_encoder(landmarks)
            if self.fusion_type == "attention":
                video_frame_embs = video_frame_embs + face_frame_embs
            else:
                video_emb = video_emb + face_emb
        return self.head(audio_emb, video_emb, video_frame_embs)
