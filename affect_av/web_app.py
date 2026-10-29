"""FastAPI inference service and static localhost web UI.

Models are discovered from checkpoint sidecars, so inputs (face crops, landmarks,
WavLM waveforms, mel normalization) follow the trained configuration instead of
a hand-maintained table. The MediaPipe landmarker is created once at startup and
serialized by a lock; each upload is decoded exactly once.
"""
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .checkpoints import (CheckpointConfig, CheckpointConfigError, checkpoint_tag,
                          discover_checkpoints, load_checkpoint, resolve_mel_stats)
from .dataset import (AUDIO_DURATION, FRAME_SIZE, center_crop_frames, decode_clip,
                      extract_audio_features, extract_audio_waveform, load_mel_stats,
                      normalize_mel)
from .face_geometry import LANDMARK_COUNT, select_landmark_features
from .face_landmarks import create_landmarker, face_crops_from_frames, landmarks_from_frames
from .models import AVEmotionModel


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
WEB_DIR = ROOT / "web"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_landmarker = None
_landmarker_lock = Lock()
_timestamp_counter = 0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _landmarker
    try:
        _landmarker = create_landmarker()
        print("Face landmarker ready.", flush=True)
    except FileNotFoundError as exc:
        print(f"Face landmarker unavailable: {exc}", flush=True)
    yield
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None


app = FastAPI(title="Affect AV", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _checkpoint_index() -> dict[str, Path]:
    return {checkpoint_tag(path.resolve(), CHECKPOINT_DIR): path.resolve()
            for path in discover_checkpoints(CHECKPOINT_DIR)}


@lru_cache(maxsize=4)
def load_model(tag: str) -> tuple[AVEmotionModel, CheckpointConfig]:
    path = _checkpoint_index().get(tag)
    if path is None:
        raise FileNotFoundError(f"Missing checkpoint: {tag}")
    return load_checkpoint(path, DEVICE)


@lru_cache(maxsize=4)
def mel_stats(path: str) -> dict:
    return load_mel_stats(path)


def _allocate_timestamp_offset() -> int:
    """Reserve a monotonic timestamp block for one clip (caller holds the lock)."""
    global _timestamp_counter
    offset = _timestamp_counter
    _timestamp_counter += int(AUDIO_DURATION * 1000) + 1
    return offset


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/models")
def models():
    return {
        "device": str(DEVICE),
        "models": sorted(_checkpoint_index()),
    }


@app.post("/api/predict")
def predict(model: str = Form(...), file: UploadFile = File(...)):
    try:
        loaded_model, config = load_model(model)
    except (FileNotFoundError, CheckpointConfigError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload an MP4 or WebM video clip.")

    stats = None
    if config.audio_encoder == "cnn" and config.mel_norm == "global":
        try:
            stats = mel_stats(resolve_mel_stats(config, ROOT))
        except (FileNotFoundError, CheckpointConfigError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
        waveform = None
        decoded = None
        if config.modality in ("video", "fusion"):
            decoded = decode_clip(str(temp_path))
            if decoded is None:
                raise ValueError("No readable video frames were found.")
        if config.modality in ("audio", "fusion"):
            if config.audio_encoder == "wavlm":
                samples = extract_audio_waveform(str(temp_path))
                if samples is None:
                    raise ValueError("No readable audio track was found.")
                waveform = torch.from_numpy(samples).unsqueeze(0).to(DEVICE)
            else:
                audio = extract_audio_features(str(temp_path))
                if audio is None:
                    raise ValueError("No readable audio track was found.")
                mel = (torch.from_numpy(normalize_mel(audio, config.mel_norm, stats))
                       .unsqueeze(0).unsqueeze(0).to(DEVICE))
        if config.modality in ("video", "fusion"):
            if config.face_crop or config.use_landmarks:
                if _landmarker is None:
                    raise ValueError("Face landmarker is unavailable on this server.")
                with _landmarker_lock:
                    if config.face_crop:
                        offset = _allocate_timestamp_offset()
                        video = face_crops_from_frames(decoded, _landmarker, size=FRAME_SIZE,
                                                       timestamp_offset_ms=offset)
                    else:
                        video = center_crop_frames(decoded, size=FRAME_SIZE)
                    if config.use_landmarks:
                        offset = _allocate_timestamp_offset()
                        face_features, valid_frames = landmarks_from_frames(
                            decoded, _landmarker, timestamp_offset_ms=offset
                        )
            else:
                video = center_crop_frames(decoded, size=FRAME_SIZE)
            frames = (torch.from_numpy(video).permute(0, 3, 1, 2).float().div(255.0)
                      .unsqueeze(0).to(DEVICE))
        if config.use_landmarks:
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
            model_features = select_landmark_features(face_features,
                                                      config.landmark_feature_mode)
            landmarks = torch.from_numpy(model_features).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = loaded_model(mel=mel, frames=frames, landmarks=landmarks,
                                  waveform=waveform)
            probabilities = torch.softmax(logits[0], dim=-1).cpu().numpy()
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
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
        "prediction": config.class_names[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "face_coverage": face_coverage,
        "facial_geometry": facial_geometry,
        "scores": [
            {"emotion": config.class_names[int(index)],
             "probability": float(probabilities[index])}
            for index in order
        ],
    }


def main() -> None:
    """Run the local web app (console entry point ``affect-web``)."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the local affect-AV web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
