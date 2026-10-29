"""MediaPipe face-mesh extraction for dataset preprocessing and web inference.

All entry points decode video through :func:`affect_av.dataset.decode_clip`
(sequential ffmpeg decoding of the first 3 seconds), so they work with browser
WebM clips that do not report a frame count. The ``*_from_frames`` helpers let
callers decode a clip once and reuse the frames for both crops and landmarks.
"""
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .dataset import AUDIO_DURATION, N_FRAMES, decode_clip
from .face_geometry import LANDMARK_COUNT, LANDMARK_FEATURE_DIM, normalize_face_landmarks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "assets" / "face_landmarker.task"


def create_landmarker(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing face landmark model: {model_path}. "
            "The repository ships assets/face_landmarker.task."
        )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _detect(landmarker, frame_rgb: np.ndarray, timestamp_ms: int):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(image, int(timestamp_ms))


def _face_crop(frame_rgb: np.ndarray, points_xy: np.ndarray, size: int,
               margin: float) -> np.ndarray:
    """Square crop around the mesh bounding box, padded and resized."""
    height, width = frame_rgb.shape[:2]
    left, top = points_xy.min(axis=0) * (width, height)
    right, bottom = points_xy.max(axis=0) * (width, height)
    face_w, face_h = max(right - left, 1), max(bottom - top, 1)
    side = max(face_w, face_h) * margin
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2 + face_h * 0.03
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    side = int(round(side))
    x1, y1 = x0 + side, y0 + side
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)
    frame = frame_rgb
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_REFLECT_101)
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_rgb
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def face_crops_from_frames(frames: np.ndarray, landmarker, size: int = 112,
                           timestamp_offset_ms: int = 0, margin: float = 1.55,
                           duration: float = AUDIO_DURATION) -> np.ndarray:
    """Return uint8 face-centered RGB crops for already-decoded frames."""
    count = len(frames)
    interval_ms = duration * 1000.0 / max(count, 1)
    crops: list[Optional[np.ndarray]] = []
    previous_timestamp = timestamp_offset_ms - 1
    for index, frame in enumerate(frames):
        timestamp = max(timestamp_offset_ms + int(index * interval_ms),
                        previous_timestamp + 1)
        previous_timestamp = timestamp
        result = _detect(landmarker, frame, timestamp)
        crop = None
        if result.face_landmarks:
            points = np.asarray([(p.x, p.y) for p in result.face_landmarks[0]],
                                dtype=np.float32)
            crop = _face_crop(frame, points, size, margin)
        crops.append(crop)
    valid = [i for i, crop in enumerate(crops) if crop is not None]
    if not valid:
        raise ValueError("No face was detected. Move closer and face the camera.")
    for i, crop in enumerate(crops):
        if crop is None:
            nearest = min(valid, key=lambda j: abs(j - i))
            crops[i] = crops[nearest].copy()
    return np.stack(crops).astype(np.uint8)


def landmarks_from_frames(frames: np.ndarray, landmarker,
                          timestamp_offset_ms: int = 0,
                          duration: float = AUDIO_DURATION):
    """Return normalized meshes and validity flags for decoded frames.

    Missing detections are interpolated from the nearest detected frame. If no
    face is found, zero features are returned with zero validity flags.
    """
    count = len(frames)
    interval_ms = duration * 1000.0 / max(count, 1)
    samples: list[Optional[np.ndarray]] = []
    valid: list[bool] = []
    previous_timestamp = timestamp_offset_ms - 1
    for index, frame in enumerate(frames):
        timestamp = max(timestamp_offset_ms + int(index * interval_ms),
                        previous_timestamp + 1)
        previous_timestamp = timestamp
        result = _detect(landmarker, frame, timestamp)
        points = None
        if result.face_landmarks:
            face = result.face_landmarks[0]
            raw_points = np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32)
            if raw_points.shape == (LANDMARK_COUNT, 3):
                blendshapes = {}
                if result.face_blendshapes:
                    blendshapes = {category.category_name: category.score
                                   for category in result.face_blendshapes[0]}
                points = normalize_face_landmarks(raw_points, blendshapes)
        samples.append(points)
        valid.append(points is not None)

    output = np.zeros((count, LANDMARK_FEATURE_DIM), dtype=np.float32)
    valid_indices = [i for i, found in enumerate(valid) if found]
    if valid_indices:
        for i, points in enumerate(samples):
            nearest = min(valid_indices, key=lambda candidate: abs(candidate - i))
            output[i] = samples[nearest]
            output[i, -1] = float(valid[i])
    return output, np.asarray(valid, dtype=np.uint8)


def extract_face_cropped_frames(video_path, n_frames: int = N_FRAMES, size: int = 112,
                                landmarker=None, timestamp_offset_ms: int = 0,
                                margin: float = 1.55,
                                duration: float = AUDIO_DURATION) -> np.ndarray:
    """Decode once, then return uint8 face-centered RGB frames."""
    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = create_landmarker()
    try:
        frames = decode_clip(video_path, n_frames=n_frames, duration=duration)
        if frames is None:
            raise ValueError(f"No readable video frames: {video_path}")
        return face_crops_from_frames(frames, landmarker, size=size,
                                      timestamp_offset_ms=timestamp_offset_ms,
                                      margin=margin, duration=duration)
    finally:
        if owns_landmarker:
            landmarker.close()


def extract_face_landmarks(video_path, n_frames: int = N_FRAMES, landmarker=None,
                           timestamp_offset_ms: int = 0,
                           duration: float = AUDIO_DURATION):
    """Decode once, then return normalized meshes and validity flags."""
    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = create_landmarker()
    try:
        frames = decode_clip(video_path, n_frames=n_frames, duration=duration)
        if frames is None:
            return (np.zeros((n_frames, LANDMARK_FEATURE_DIM), dtype=np.float32),
                    np.zeros(n_frames, dtype=np.uint8))
        return landmarks_from_frames(frames, landmarker,
                                     timestamp_offset_ms=timestamp_offset_ms,
                                     duration=duration)
    finally:
        if owns_landmarker:
            landmarker.close()
