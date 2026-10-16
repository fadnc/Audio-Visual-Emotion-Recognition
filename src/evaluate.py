"""
Evaluate all trained checkpoints on the test split and print/save an
ablation table (audio-only vs video-only vs fusion variants) plus a
confusion matrix for the best model.

Usage:
    python src/evaluate.py --data_dir data/processed --checkpoint_dir checkpoints
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import AVEmotionDataset, EMOTION_TO_IDX
from models import AVEmotionModel
from face_geometry import LANDMARK_GEOMETRY_DIM, LANDMARK_FEATURE_DIM

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

CONFIGS = [
    {"tag": "audio", "modality": "audio", "fusion": "late"},
    {"tag": "video", "modality": "video", "fusion": "late"},
    {"tag": "fusion_late", "modality": "fusion", "fusion": "late"},
    {"tag": "fusion_attention", "modality": "fusion", "fusion": "attention"},
    {"tag": "video_face", "modality": "video", "fusion": "late", "face_crop": True},
    {"tag": "fusion_attention_face", "modality": "fusion", "fusion": "attention",
     "face_crop": True},
    {"tag": "fusion_attention_tuned", "modality": "fusion", "fusion": "attention",
     "checkpoint": os.path.join("tuning", "attention_smoothed_auc", "fusion_attention.pt")},
    {"tag": "fusion_attention_landmarks", "modality": "fusion", "fusion": "attention",
     "use_landmarks": True, "landmark_feature_mode": "geometry"},
]


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        mel = batch["mel"].to(device)
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        landmarks = batch.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device)
        logits = model(mel=mel, frames=frames, landmarks=landmarks)
        all_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, np.asarray(all_probs), multi_class="ovr",
                        average="macro", labels=np.arange(len(EMOTION_TO_IDX)))
    return acc, f1, auc, all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--landmark_dir", default="data/landmarks")
    ap.add_argument("--face_frame_dir", default="data/face_frames")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="Use validation while tuning; reserve test for final reporting.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_landmarks = any(
        cfg.get("use_landmarks")
        and os.path.exists(os.path.join(args.checkpoint_dir, f"{cfg['tag']}.pt"))
        for cfg in CONFIGS
    )
    n_classes = len(EMOTION_TO_IDX)

    results = []
    best_tag, best_f1, best_labels, best_preds = None, -1, None, None

    for cfg in CONFIGS:
        ckpt_path = os.path.join(args.checkpoint_dir,
                                 cfg.get("checkpoint", f"{cfg['tag']}.pt"))
        if not os.path.exists(ckpt_path):
            print(f"skip {cfg['tag']}: no checkpoint at {ckpt_path}")
            continue
        split_ds = AVEmotionDataset(
            args.data_dir, args.split,
            include_landmarks=cfg.get("use_landmarks", False),
            landmark_dir=args.landmark_dir,
            landmark_feature_mode=cfg.get("landmark_feature_mode", "geometry"),
            face_frame_dir=args.face_frame_dir if cfg.get("face_crop") else None,
        )
        test_loader = DataLoader(split_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        model = AVEmotionModel(modality=cfg["modality"], fusion_type=cfg["fusion"],
                                n_classes=n_classes,
                                use_landmarks=cfg.get("use_landmarks", False),
                                landmark_feature_dim=(LANDMARK_GEOMETRY_DIM
                                    if cfg.get("landmark_feature_mode", "geometry") == "geometry"
                                    else LANDMARK_FEATURE_DIM)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        acc, f1, auc, labels, preds = evaluate_model(model, test_loader, device)
        results.append({"model": cfg["tag"], "accuracy": round(acc, 3),
                        "macro_f1": round(f1, 3), "macro_ovr_auc": round(auc, 3)})
        print(f"{cfg['tag']:24s} | acc {acc:.3f} | macro-F1 {f1:.3f} | macro-AUROC {auc:.3f}")
        if f1 > best_f1:
            best_tag, best_f1, best_labels, best_preds = cfg["tag"], f1, labels, preds

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.checkpoint_dir, "ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ablation table -> {out_csv}")
    print(df.to_string(index=False))

    if best_tag is not None:
        cm = confusion_matrix(best_labels, best_preds)
        labels_sorted = [IDX_TO_EMOTION[i] for i in range(n_classes)]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — best model: {best_tag}")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        out_png = os.path.join(args.checkpoint_dir, "confusion_matrix.png")
        fig.savefig(out_png, dpi=150)
        print(f"Saved confusion matrix -> {out_png}")


if __name__ == "__main__":
    main()
