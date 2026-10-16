"""FastAPI inference service and static localhost web UI."""
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.dataset import EMOTION_TO_IDX, extract_audio_features, extract_frames
from src.face_geometry import (LANDMARK_COUNT, LANDMARK_FEATURE_DIM,
                               LANDMARK_GEOMETRY_DIM, select_landmark_features)
from src.face_landmarks import extract_face_landmarks, extract_face_cropped_frames
from src.models import AVEmotionModel


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
WEB_DIR = ROOT / "web"
MODEL_CONFIGS = {
    "fusion_attention_landmarks": ("fusion", "attention", True, "geometry"),
    "fusion_attention_tuned": ("fusion", "attention"),
    "video_face": ("video", "late", False, "geometry", True),
    "fusion_attention_face": ("fusion", "attention", False, "geometry", True),
    "video": ("video", "late"),
    "fusion_attention": ("fusion", "attention"),
    "fusion_late": ("fusion", "late"),
    "audio": ("audio", "late"),
}
IDX_TO_EMOTION = {idx: name for name, idx in EMOTION_TO_IDX.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(tag: str) -> Path:
    if tag == "fusion_attention_tuned":
        return CHECKPOINT_DIR / "tuning" / "attention_smoothed_auc" / "fusion_attention.pt"
    return CHECKPOINT_DIR / f"{tag}.pt"

app = FastAPI(title="Affect AV", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@lru_cache(maxsize=4)
def load_model(tag: str) -> AVEmotionModel:
    if tag not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {tag}")
    checkpoint = checkpoint_path(tag)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint.name}")
    config = MODEL_CONFIGS[tag]
    modality, fusion = config[:2]
    use_landmarks = len(config) > 2 and config[2]
    landmark_feature_mode = config[3] if len(config) > 3 else "geometry"
    model = AVEmotionModel(
        modality=modality,
        fusion_type=fusion,
        n_classes=len(EMOTION_TO_IDX),
        pretrained=False,
        use_landmarks=use_landmarks,
        landmark_feature_dim=(LANDMARK_GEOMETRY_DIM if landmark_feature_mode == "geometry"
                              else LANDMARK_FEATURE_DIM),
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/models")
def models():
    return {
        "device": str(DEVICE),
        "models": [tag for tag in MODEL_CONFIGS if checkpoint_path(tag).is_file()],
    }


@app.post("/api/predict")
def predict(model: str = Form(...), file: UploadFile = File(...)):
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Unknown model selection.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload an MP4 or WebM video clip.")
    try:
        loaded_model = load_model(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    config = MODEL_CONFIGS[model]
    modality, _ = config[:2]
    use_landmarks = len(config) > 2 and config[2]
    landmark_feature_mode = config[3] if len(config) > 3 else "geometry"
    face_coverage = None
    facial_geometry = None
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = Path(temp.name)
            total_bytes = 0
            while chunk := file.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > 200 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Video exceeds the 200 MB limit.")
                temp.write(chunk)
            temp.flush()
        mel = None
        frames = None
        landmarks = None
        if modality in ("audio", "fusion"):
            audio = extract_audio_features(str(temp_path))
            if audio is None:
                raise ValueError("No readable audio track was found.")
            mel = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)
        if modality in ("video", "fusion"):
            video = (extract_face_cropped_frames(str(temp_path))
                     if len(config) > 4 and config[4]
                     else extract_frames(str(temp_path)))
            if video is None:
                raise ValueError("No readable video frames were found.")
            frames = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)
        if use_landmarks:
            face_features, valid_frames = extract_face_landmarks(str(temp_path))
            detected = int(valid_frames.sum())
            if detected == 0:
                raise ValueError("No face was detected. Center your face and improve the lighting.")
            face_coverage = detected / len(valid_frames)
            valid_features = face_features[valid_frames.astype(bool)]
            geometry_start = LANDMARK_COUNT * 3
            geometry = valid_features[:, geometry_start:geometry_start + 21].mean(axis=0)
            facial_geometry = {
                "eye_openness": float(geometry[:2].mean()),
                "brow_eye_gap": float(geometry[4:6].mean()),
                "mouth_opening": float(geometry[8]),
                "left_lip_corner_lift": float(geometry[11]),
                "right_lip_corner_lift": float(geometry[12]),
                "forehead_height": float(geometry[14]),
                "face_width": float(geometry[15]),
                "face_height": float(geometry[16]),
            }
            model_features = select_landmark_features(face_features, landmark_feature_mode)
            landmarks = torch.from_numpy(model_features).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = loaded_model(mel=mel, frames=frames, landmarks=landmarks)
            probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process this clip: {exc}") from exc
    finally:
        file.file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    order = np.argsort(probabilities)[::-1]
    return {
        "model": model,
        "device": str(DEVICE),
        "prediction": IDX_TO_EMOTION[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "face_coverage": face_coverage,
        "facial_geometry": facial_geometry,
        "scores": [
            {"emotion": IDX_TO_EMOTION[int(index)], "probability": float(probabilities[index])}
            for index in order
        ],
    }
