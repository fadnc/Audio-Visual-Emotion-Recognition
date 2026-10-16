"""Download and evaluate a balanced 50-clip CREMA-D external test sample.

The sample uses the official CREMA-D audiovisual majority vote and one clip per
actor. This is an external test only; no CREMA-D samples are used for training.
"""
import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torchvision  # noqa: F401

venv_site = ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.is_dir():
    sys.path.append(str(venv_site))

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score, top_k_accuracy_score)
from torch.utils.data import DataLoader

from src.dataset import EMOTION_TO_IDX, extract_audio_features, extract_frames
from src.face_landmarks import create_landmarker, extract_face_cropped_frames
from src.models import AVEmotionModel
from src.face_geometry import LANDMARK_GEOMETRY_DIM


DATASET_URL = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv"
VIDEO_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/VideoFlash/{filename}"
CODE_TO_EMOTION = {"A": "angry", "D": "disgust", "F": "fearful",
                   "H": "happy", "N": "neutral", "S": "sad"}
FILENAME_CODE_TO_EMOTION = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
                            "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CLASS_ORDER = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]
QUOTAS = {"A": 9, "D": 9, "F": 8, "H": 8, "N": 8, "S": 8}
USER_AGENT = "AV-Emotion-Recognition external evaluation"


def fetch(url, target=None, retries=4):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            return payload
        except Exception as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download failed: {url}: {error}")


def select_sample(table):
    rng = random.Random(20261016)
    candidates = {code: [] for code in QUOTAS}
    for row in table.to_dict("records"):
        name = str(row.get("FileName", ""))
        vote = str(row.get("MultiModalVote", ""))
        if vote not in candidates or not name or name.lower() == "nan":
            continue
        if not name.endswith(".flv"):
            name += ".flv"
        candidates[vote].append({
            "filename": name,
            "actor": name.split("_")[0],
            "vote_code": vote,
            "emotion": CODE_TO_EMOTION[vote],
            "intended_emotion": FILENAME_CODE_TO_EMOTION[name.split("_")[2]],
        })
    for rows in candidates.values():
        rng.shuffle(rows)
    chosen, used_actors = [], set()
    for code, quota in QUOTAS.items():
        selected = []
        for row in candidates[code]:
            if row["actor"] in used_actors:
                continue
            selected.append(row)
            used_actors.add(row["actor"])
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise RuntimeError(f"Only selected {len(selected)}/{quota} clips for label {code}")
        chosen.extend(selected)
    rng.shuffle(chosen)
    return chosen


def prepare_data(data_dir, prepare_only=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir = data_dir / "VideoFlash"
    manifest_path = data_dir / "external_manifest.csv"
    if not manifest_path.is_file():
        metadata = fetch(DATASET_URL)
        table = pd.read_csv(io.BytesIO(metadata))
        chosen = select_sample(table)
        pd.DataFrame(chosen).to_csv(manifest_path, index=False)
    else:
        chosen = pd.read_csv(manifest_path).to_dict("records")
        actors = [row["actor"] for row in chosen]
        if len(set(actors)) != len(actors):
            print("Refreshing sample: prior manifest repeated an actor.", flush=True)
            table = pd.read_csv(io.BytesIO(fetch(DATASET_URL)))
            chosen = select_sample(table)
            pd.DataFrame(chosen).to_csv(manifest_path, index=False)
    missing = [row for row in chosen if not (video_dir / row["filename"]).is_file()]
    for index, row in enumerate(missing, start=1):
        destination = video_dir / row["filename"]
        payload = fetch(VIDEO_URL.format(filename=row["filename"]), destination)
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Git LFS pointer received instead of video: {row['filename']}")
        print(f"Downloaded {index}/{len(missing)}: {row['filename']} ({len(payload):,} bytes)",
              flush=True)
    if prepare_only:
        print(f"Prepared {len(chosen)} CREMA-D clips under {data_dir}")
    return chosen, video_dir


@torch.no_grad()
def predict(model, rows, video_dir, device, face_crop=True):
    probabilities, truth, filenames, failures = [], [], [], []
    landmarker = create_landmarker() if face_crop else None
    try:
        for index, row in enumerate(rows):
            path = video_dir / row["filename"]
            try:
                audio = extract_audio_features(str(path))
                if audio is None:
                    raise ValueError("audio stream could not be decoded")
                if face_crop:
                    frames = extract_face_cropped_frames(
                        str(path), landmarker=landmarker, timestamp_offset_ms=index * 10000
                    )
                else:
                    frames = extract_frames(str(path))
                    if frames is None:
                        raise ValueError("video frames could not be decoded")
                mel_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
                frame_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                logits = model(mel=mel_tensor, frames=frame_tensor)
                probabilities.append(torch.softmax(logits[0], dim=-1).cpu().numpy())
                truth.append(EMOTION_TO_IDX[row["emotion"]])
                filenames.append(row["filename"])
                print(f"Evaluated {len(filenames)}/{len(rows)}: {row['filename']}", flush=True)
            except Exception as exc:
                failures.append({"filename": row["filename"], "reason": str(exc)})
                print(f"Skipped {row['filename']}: {exc}", flush=True)
    finally:
        if landmarker is not None:
            landmarker.close()
    return np.asarray(probabilities), np.asarray(truth), filenames, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/external/crema_d")
    parser.add_argument("--checkpoint", default="checkpoints/fusion_attention_face.pt")
    parser.add_argument("--report_dir", default="reports/crema_d_external")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--full_frame", action="store_true",
                        help="Use center-cropped frames for a non-face-crop checkpoint.")
    args = parser.parse_args()
    rows, video_dir = prepare_data(args.data_dir, args.prepare_only)
    if args.prepare_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVEmotionModel(modality="fusion", fusion_type="attention",
                           n_classes=len(EMOTION_TO_IDX), pretrained=False,
                           landmark_feature_dim=LANDMARK_GEOMETRY_DIM).to(device)
    checkpoint_path = ROOT / args.checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    probs, y_true, filenames, failures = predict(
        model, rows, video_dir, device, face_crop=not args.full_frame
    )
    if len(y_true) < 2:
        raise RuntimeError("Fewer than two clips were successfully evaluated.")

    class_indices = [EMOTION_TO_IDX[name] for name in CLASS_ORDER]
    y_pred_all = probs.argmax(axis=1)
    known_prediction = np.isin(y_pred_all, class_indices)
    y_pred_6 = np.where(known_prediction, y_pred_all, -1)
    y_prob_6 = probs[:, class_indices]
    accuracy = accuracy_score(y_true, y_pred_6)
    macro_f1 = f1_score(y_true, y_pred_6, labels=class_indices,
                        average="macro", zero_division=0)
    y_true_binary = np.stack([y_true == class_index for class_index in class_indices], axis=1)
    macro_auc = roc_auc_score(y_true_binary, y_prob_6, average="macro")
    balanced_acc = balanced_accuracy_score(y_true, y_pred_6)
    top2 = top_k_accuracy_score(y_true, probs, k=2,
                                labels=np.arange(len(EMOTION_TO_IDX)))
    full_labels = list(range(len(EMOTION_TO_IDX)))
    cm = confusion_matrix(y_true, y_pred_all, labels=full_labels)
    report = classification_report(y_true, y_pred_6, labels=class_indices,
                                   target_names=CLASS_ORDER, output_dict=True,
                                   zero_division=0)
    output_dir = ROOT / args.report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    intended_by_file = {row["filename"]: row["intended_emotion"] for row in rows}
    intended_names = [intended_by_file[name] for name in filenames]
    intended_y = np.asarray([EMOTION_TO_IDX[name] for name in intended_names])
    intended_auc = None
    intended_report = None
    intended_cm = confusion_matrix(intended_y, y_pred_all, labels=full_labels)
    if all(np.any(intended_y == class_index) for class_index in class_indices):
        intended_binary = np.stack([intended_y == class_index for class_index in class_indices], axis=1)
        intended_auc = float(roc_auc_score(intended_binary, y_prob_6, average="macro"))
        intended_report = classification_report(
            intended_y, y_pred_6, labels=class_indices, target_names=CLASS_ORDER,
            output_dict=True, zero_division=0
        )
    intended_accuracy = float(accuracy_score(intended_y, y_pred_6))
    intended_balanced_acc = float(balanced_accuracy_score(intended_y, y_pred_6))
    intended_macro_f1 = float(f1_score(intended_y, y_pred_6, labels=class_indices,
                                       average="macro", zero_division=0))
    intended_top2 = float(top_k_accuracy_score(
        intended_y, probs, k=2, labels=np.arange(len(EMOTION_TO_IDX))
    ))
    metrics = {
        "dataset": "CREMA-D",
        "n_requested": len(rows),
        "n_evaluated": len(y_true),
        "n_failed": len(failures),
        "sample_selection": "50 clips, balanced by CREMA-D MultiModalVote; one clip per actor; seed 20261016",
        "model": str(args.checkpoint),
        "accuracy": float(accuracy),
        "multimodal_vote_accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "macro_ovr_auroc_6class": float(macro_auc),
        "intended_label_accuracy": intended_accuracy,
        "intended_label_balanced_accuracy": intended_balanced_acc,
        "intended_label_macro_f1": intended_macro_f1,
        "intended_label_macro_ovr_auroc_6class": intended_auc,
        "intended_label_top2_accuracy_8outputs": intended_top2,
        "intended_label_confusion_matrix_8outputs": intended_cm.tolist(),
        "top2_accuracy_8outputs": float(top2),
        "unsupported_output_predictions": int((~known_prediction).sum()),
        "class_order": CLASS_ORDER,
        "confusion_matrix_8outputs": cm.tolist(),
        "per_class": report,
        "intended_label_per_class": intended_report,
        "failures": failures,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    sample_out = pd.DataFrame({
        "filename": filenames,
        "true_emotion": [CLASS_ORDER[class_indices.index(label)] for label in y_true],
        "intended_emotion": intended_names,
        "predicted_emotion": [list(EMOTION_TO_IDX)[pred] for pred in y_pred_all],
        "confidence": probs.max(axis=1),
    })
    sample_out.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(cm, index=list(EMOTION_TO_IDX), columns=list(EMOTION_TO_IDX)).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    print(json.dumps({k: metrics[k] for k in (
        "n_evaluated", "n_failed", "accuracy", "balanced_accuracy", "macro_f1",
        "macro_ovr_auroc_6class", "top2_accuracy_8outputs",
        "unsupported_output_predictions", "intended_label_accuracy",
        "intended_label_balanced_accuracy", "intended_label_macro_f1",
        "intended_label_macro_ovr_auroc_6class", "intended_label_top2_accuracy_8outputs")}, indent=2))
    print(f"Saved external evaluation report to {output_dir}")


if __name__ == "__main__":
    main()
