"""
RAVDESS audio-visual preprocessing.

Scans data/raw/Actor_XX/*.mp4 (full audio-visual clips, modality code 01),
extracts:
  - log-mel spectrogram from the audio track (fixed size, padded/cropped)
  - N evenly-sampled RGB frames from the video track (resized, center-cropped)

Saves each clip as a single .npz in data/processed/ plus a manifest.csv
with (clip_id, actor, emotion_label, split) columns.

Usage:
    python src/dataset.py --raw_dir data/raw --out_dir data/processed
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2
import librosa
try:
    from moviepy import AudioFileClip           # moviepy >= 2.0
except ImportError:
    from moviepy.editor import AudioFileClip    # moviepy 1.x
from tqdm import tqdm

# --- config -----------------------------------------------------------
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
EMOTION_TO_IDX = {name: i for i, name in enumerate(sorted(set(EMOTIONS.values())))}

N_FRAMES = 8
FRAME_SIZE = 112          # square crop, matches small ResNet input well
AUDIO_SR = 16000
N_MELS = 64
AUDIO_DURATION = 3.0      # seconds, pad/crop to this
AUDIO_FRAMES = int(AUDIO_DURATION * AUDIO_SR / 512)  # hop_length=512


def parse_filename(path):
    """RAVDESS filename: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4"""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("-")
    if len(parts) != 7:
        return None
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "01":  # keep only full audio-visual clips
        return None
    return {
        "emotion_code": emotion,
        "emotion": EMOTIONS.get(emotion),
        "actor": int(actor),
    }


def extract_frames(video_path, n_frames=N_FRAMES, size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    for target_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            m = min(h, w)
            frame = frame[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
            frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    frames = np.stack(frames).astype(np.float32) / 255.0  # (N, H, W, 3)
    return frames


def extract_audio_features(video_path):
    try:
        clip = AudioFileClip(video_path)
        arr = clip.to_soundarray(fps=AUDIO_SR)
        clip.close()
    except Exception as e:
        print(f"\n[audio error] {video_path}: {e}")
        return None
    y = arr.mean(axis=1) if arr.ndim > 1 else arr
    y = y.astype(np.float32)
    target_len = int(AUDIO_DURATION * AUDIO_SR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, n_mels=N_MELS, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # (N_MELS, T)
    # normalize to roughly [-1, 1]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def build_manifest(raw_dir):
    rows = []
    for path in glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True):
        meta = parse_filename(path)
        if meta is None or meta["emotion"] is None:
            continue
        rows.append({"path": path, **meta})
    df = pd.DataFrame(rows)
    return df


def assign_splits(df, val_actors=(7,), test_actors=(8,)):
    """Speaker-independent split: held-out actors for val/test."""
    def split_for(actor):
        if actor in val_actors:
            return "val"
        if actor in test_actors:
            return "test"
        return "train"
    df["split"] = df["actor"].apply(split_for)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_actors", nargs="+", type=int, default=[7])
    ap.add_argument("--test_actors", nargs="+", type=int, default=[8])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = build_manifest(args.raw_dir)
    if len(df) == 0:
        raise SystemExit(
            f"No clips found under {args.raw_dir} (searched recursively) with modality code 01. "
            "Check you downloaded the full audio-visual RAVDESS zips."
        )
    df = assign_splits(df, tuple(args.val_actors), tuple(args.test_actors))

    manifest_rows = []
    frame_fail, audio_fail, shown = 0, 0, 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="extracting"):
        clip_id = f"clip_{i:05d}"
        frames = extract_frames(row["path"])
        mel = extract_audio_features(row["path"])
        if frames is None or mel is None:
            if frames is None:
                frame_fail += 1
            if mel is None:
                audio_fail += 1
            if shown < 3:
                print(f"\n[skip] {row['path']} -> frames_ok={frames is not None} audio_ok={mel is not None}")
                shown += 1
            continue
        np.savez_compressed(
            os.path.join(args.out_dir, f"{clip_id}.npz"),
            frames=frames, mel=mel,
        )
        manifest_rows.append({
            "clip_id": clip_id,
            "actor": row["actor"],
            "emotion": row["emotion"],
            "label": EMOTION_TO_IDX[row["emotion"]],
            "split": row["split"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"Processed {len(manifest)} clips -> {args.out_dir}/manifest.csv")
    if len(manifest) == 0:
        print(f"\nAll clips failed. frame_extraction_failures={frame_fail} "
              f"audio_extraction_failures={audio_fail} (out of {len(df)}).")
        if frame_fail > 0:
            print(
                "Likely cause: OpenCV on this machine can't decode these .mp4 files "
                "(missing H.264 codec backend). Fix: pip install opencv-python "
                "(instead of / alongside opencv-python-headless), or install the "
                "K-Lite / OpenH264 codec, then retry. As a workaround we can also "
                "switch frame extraction to moviepy/ffmpeg instead of cv2 — say the "
                "word and I'll patch it."
            )
        return
    print(manifest["split"].value_counts())
    print(manifest["emotion"].value_counts())


# --- PyTorch Dataset used by train.py / evaluate.py --------------------
import torch
from torch.utils.data import Dataset


class AVEmotionDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        manifest = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        npz = np.load(os.path.join(self.data_dir, f"{row['clip_id']}.npz"))
        frames = torch.from_numpy(npz["frames"]).permute(0, 3, 1, 2)  # (N, 3, H, W)
        mel = torch.from_numpy(npz["mel"]).unsqueeze(0)               # (1, N_MELS, T)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {"frames": frames, "mel": mel, "label": label}


if __name__ == "__main__":
    main()