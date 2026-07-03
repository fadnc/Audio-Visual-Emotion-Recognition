"""
Train an audio-only, video-only, or fusion emotion classifier.

Usage:
    python src/train.py --data_dir data/processed --modality audio
    python src/train.py --data_dir data/processed --modality video
    python src/train.py --data_dir data/processed --modality fusion --fusion late
    python src/train.py --data_dir data/processed --modality fusion --fusion attention
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import AVEmotionDataset, EMOTION_TO_IDX
from models import AVEmotionModel


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            mel = batch["mel"].to(device)
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            logits = model(mel=mel, frames=frames)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--modality", choices=["audio", "video", "fusion"], required=True)
    ap.add_argument("--fusion", choices=["late", "attention"], default="late")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = AVEmotionDataset(args.data_dir, "train")
    val_ds = AVEmotionDataset(args.data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    n_classes = len(EMOTION_TO_IDX)
    model = AVEmotionModel(modality=args.modality, fusion_type=args.fusion,
                            n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    tag = args.modality if args.modality != "fusion" else f"fusion_{args.fusion}"
    best_f1, patience_ctr = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, device, optimizer=None)

        print(f"[{tag}] epoch {epoch:02d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{tag}.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val macro-F1 {best_f1:.3f})")
                break

    print(f"Best val macro-F1 for {tag}: {best_f1:.3f} -> saved to "
          f"{args.checkpoint_dir}/{tag}.pt")


if __name__ == "__main__":
    main()
