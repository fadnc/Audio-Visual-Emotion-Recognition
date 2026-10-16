"""MediaPipe face-mesh extraction for dataset preprocessing and web inference."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    from .face_geometry import LANDMARK_COUNT, LANDMARK_FEATURE_DIM, normalize_face_landmarks
except ImportError:
    from face_geometry import LANDMARK_COUNT, LANDMARK_FEATURE_DIM, normalize_face_landmarks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "assets" / "face_landmarker.task"
N_FRAMES = 8


def create_landmarker(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing face landmark model: {model_path}. Run scripts/download_face_model.py."
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


def extract_face_cropped_frames(video_path, n_frames=N_FRAMES, size=112, landmarker=None,
                                timestamp_offset_ms=0, margin=1.55):
    """Return face-centered RGB frames using a square crop around the face mesh.

    The crop includes forehead and chin context. If detection misses an isolated
    sample, the nearest detected crop is reused; clips with no detected face fail.
    """
    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = create_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            raise ValueError(f"No readable video frames: {video_path}")
        indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        crops, previous_timestamp = [], timestamp_offset_ms - 1
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            crop = None
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(timestamp_offset_ms + int(index * 1000.0 / fps),
                                previous_timestamp + 1)
                previous_timestamp = timestamp
                result = landmarker.detect_for_video(image, timestamp)
                if result.face_landmarks:
                    points = np.asarray([(p.x, p.y) for p in result.face_landmarks[0]],
                                        dtype=np.float32)
                    h, w = frame.shape[:2]
                    left, top = points.min(axis=0) * (w, h)
                    right, bottom = points.max(axis=0) * (w, h)
                    face_w, face_h = max(right - left, 1), max(bottom - top, 1)
                    side = max(face_w, face_h) * margin
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2 + face_h * 0.03
                    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
                    side = int(round(side))
                    x1, y1 = x0 + side, y0 + side
                    pad_left, pad_top = max(0, -x0), max(0, -y0)
                    pad_right, pad_bottom = max(0, x1 - w), max(0, y1 - h)
                    if any((pad_left, pad_top, pad_right, pad_bottom)):
                        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left,
                                                   pad_right, cv2.BORDER_REFLECT_101)
                        x0 += pad_left
                        y0 += pad_top
                        x1 += pad_left
                        y1 += pad_top
                    crop = frame[y0:y1, x0:x1]
                    if crop.size:
                        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        valid = [i for i, crop in enumerate(crops) if crop is not None]
        if not valid:
            raise ValueError("No face was detected. Move closer and face the camera.")
        for i, crop in enumerate(crops):
            if crop is None:
                nearest = min(valid, key=lambda j: abs(j - i))
                crops[i] = crops[nearest].copy()
        return np.stack(crops).astype(np.float32) / 255.0
    finally:
        cap.release()
        if owns_landmarker:
            landmarker.close()


def extract_face_landmarks(video_path, n_frames=N_FRAMES, landmarker=None,
                           timestamp_offset_ms=0):
    """Track one face through evenly spaced frames and return normalized meshes.

    Missing detections are interpolated from the nearest detected frame. If the
    clip contains no detectable face, zero features are returned with zero
    validity flags rather than manufacturing a face signal.
    """
    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = create_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            return (np.zeros((n_frames, LANDMARK_FEATURE_DIM), dtype=np.float32),
                    np.zeros(n_frames, dtype=np.uint8))
        indices = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        samples = []
        valid = []
        previous_timestamp = timestamp_offset_ms - 1
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            points = None
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(timestamp_offset_ms + int(index * 1000.0 / fps),
                                previous_timestamp + 1)
                previous_timestamp = timestamp
                result = landmarker.detect_for_video(image, timestamp)
                if result.face_landmarks:
                    face = result.face_landmarks[0]
                    raw_points = np.array(
                        [(point.x, point.y, point.z) for point in face], dtype=np.float32
                    )
                    if raw_points.shape == (LANDMARK_COUNT, 3):
                        blendshapes = {}
                        if result.face_blendshapes:
                            blendshapes = {
                                category.category_name: category.score
                                for category in result.face_blendshapes[0]
                            }
                        points = normalize_face_landmarks(raw_points, blendshapes)
            samples.append(points)
            valid.append(points is not None)
        cap.release()
        valid_indices = [i for i, found in enumerate(valid) if found]
        output = np.zeros((n_frames, LANDMARK_FEATURE_DIM), dtype=np.float32)
        if valid_indices:
            for i, points in enumerate(samples):
                nearest = min(valid_indices, key=lambda candidate: abs(candidate - i))
                output[i] = samples[nearest]
                output[i, -1] = float(valid[i])
        return output, np.asarray(valid, dtype=np.uint8)
    finally:
        cap.release()
        if owns_landmarker:
            landmarker.close()
