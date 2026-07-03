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

EMBED_DIM = 128


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

    def __init__(self, embed_dim=EMBED_DIM, freeze_backbone=True):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc -> (B, 512, 1, 1)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # fine-tune only the last residual block
            for p in list(self.backbone.children())[-2].parameters():
                p.requires_grad = True
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, frames):
        # frames: (B, N, 3, H, W)
        B, N, C, H, W = frames.shape
        x = frames.view(B * N, C, H, W)
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

    def __init__(self, modality="fusion", fusion_type="late", n_classes=8, embed_dim=EMBED_DIM):
        super().__init__()
        self.modality = modality
        self.fusion_type = fusion_type

        if modality in ("audio", "fusion"):
            self.audio_encoder = AudioEncoder(embed_dim)
        if modality in ("video", "fusion"):
            self.video_encoder = VideoEncoder(embed_dim)

        if modality == "audio":
            self.head = nn.Linear(embed_dim, n_classes)
        elif modality == "video":
            self.head = nn.Linear(embed_dim, n_classes)
        elif modality == "fusion":
            self.head = (LateFusionHead(embed_dim, n_classes) if fusion_type == "late"
                         else AttentionFusionHead(embed_dim, n_classes))
        else:
            raise ValueError(modality)

    def forward(self, mel=None, frames=None):
        if self.modality == "audio":
            emb = self.audio_encoder(mel)
            return self.head(emb)
        if self.modality == "video":
            pooled, _ = self.video_encoder(frames)
            return self.head(pooled)
        # fusion
        audio_emb = self.audio_encoder(mel)
        video_emb, video_frame_embs = self.video_encoder(frames)
        return self.head(audio_emb, video_emb, video_frame_embs)
