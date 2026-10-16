"""Face-mesh geometry features shared by training and live inference."""
import numpy as np


LANDMARK_COUNT = 478
GEOMETRY_FEATURES = (
    "left_eye_aperture", "right_eye_aperture", "left_eye_width", "right_eye_width",
    "left_brow_gap", "right_brow_gap", "left_brow_tilt", "right_brow_tilt",
    "mouth_opening", "mouth_width", "mouth_aspect", "left_corner_lift",
    "right_corner_lift", "lip_curve", "forehead_height", "face_width",
    "face_height", "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
)
FACE_BLENDSHAPE_NAMES = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
)
LANDMARK_FEATURE_DIM = (LANDMARK_COUNT * 3 + len(GEOMETRY_FEATURES)
                        + len(FACE_BLENDSHAPE_NAMES) + 1)
LANDMARK_GEOMETRY_DIM = LANDMARK_COUNT * 3 + len(GEOMETRY_FEATURES) + 1


def select_landmark_features(features, mode="geometry"):
    """Select a stable feature layout from full mesh + metrics + blendshapes."""
    features = np.asarray(features, dtype=np.float32)
    if mode == "blendshapes":
        return features
    if mode != "geometry":
        raise ValueError(f"Unknown landmark feature mode: {mode}")
    geometry_end = LANDMARK_COUNT * 3 + len(GEOMETRY_FEATURES)
    return np.concatenate([features[..., :geometry_end], features[..., -1:]], axis=-1)


def _distance(points, first, second):
    return float(np.linalg.norm(points[first, :2] - points[second, :2]))


def normalize_face_landmarks(points, blendshape_scores=None):
    """Return scale/roll-normalized mesh coordinates plus interpretable ratios.

    Coordinates use the eye centers as origin and inter-eye distance as unit.
    Roll is removed, while yaw/pitch and expression shape remain represented.
    The final value marks a successfully detected face.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected ({LANDMARK_COUNT}, 3) face points, got {points.shape}")

    left_eye = points[[33, 133, 159, 145]].mean(axis=0)
    right_eye = points[[362, 263, 386, 374]].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2.0
    eye_delta = right_eye[:2] - left_eye[:2]
    eye_distance = max(float(np.linalg.norm(eye_delta)), 1e-5)
    angle = float(np.arctan2(eye_delta[1], eye_delta[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    delta = points - eye_center
    normalized = np.empty_like(delta)
    normalized[:, 0] = (delta[:, 0] * cosine + delta[:, 1] * sine) / eye_distance
    normalized[:, 1] = (-delta[:, 0] * sine + delta[:, 1] * cosine) / eye_distance
    normalized[:, 2] = delta[:, 2] / eye_distance
    normalized = np.clip(normalized, -3.0, 3.0)

    left_eye_width = _distance(points, 33, 133)
    right_eye_width = _distance(points, 362, 263)
    mouth_width = max(_distance(points, 61, 291), 1e-5)
    mouth_opening = _distance(points, 13, 14)
    mouth_center_y = float((normalized[13, 1] + normalized[14, 1]) / 2)
    left_brow = points[[70, 63, 105, 66, 55]].mean(axis=0)
    right_brow = points[[300, 293, 334, 296, 285]].mean(axis=0)
    left_eye_y = float(points[[159, 145]].mean(axis=0)[1])
    right_eye_y = float(points[[386, 374]].mean(axis=0)[1])
    left_brow_gap = (left_eye_y - float(left_brow[1])) / eye_distance
    right_brow_gap = (right_eye_y - float(right_brow[1])) / eye_distance
    left_brow_tilt = float((points[105, 1] - points[55, 1]) / eye_distance)
    right_brow_tilt = float((points[334, 1] - points[285, 1]) / eye_distance)
    left_corner_lift = mouth_center_y - float(normalized[61, 1])
    right_corner_lift = mouth_center_y - float(normalized[291, 1])
    left_iris = normalized[468:473, :2].mean(axis=0)
    right_iris = normalized[473:478, :2].mean(axis=0)

    metrics = np.array([
        _distance(points, 159, 145) / max(left_eye_width, 1e-5),
        _distance(points, 386, 374) / max(right_eye_width, 1e-5),
        left_eye_width / eye_distance,
        right_eye_width / eye_distance,
        left_brow_gap,
        right_brow_gap,
        left_brow_tilt,
        right_brow_tilt,
        mouth_opening / eye_distance,
        mouth_width / eye_distance,
        mouth_opening / mouth_width,
        left_corner_lift,
        right_corner_lift,
        float((normalized[61, 1] + normalized[291, 1]) / 2 - mouth_center_y),
        float(normalized[10, 1]),
        _distance(points, 234, 454) / eye_distance,
        _distance(points, 10, 152) / eye_distance,
        float(left_iris[0]), float(left_iris[1]),
        float(right_iris[0]), float(right_iris[1]),
    ], dtype=np.float32)
    blendshapes = np.zeros(len(FACE_BLENDSHAPE_NAMES), dtype=np.float32)
    if blendshape_scores:
        index_by_name = {name: index for index, name in enumerate(FACE_BLENDSHAPE_NAMES)}
        for name, score in blendshape_scores.items():
            if name in index_by_name:
                blendshapes[index_by_name[name]] = score
    return np.concatenate([
        normalized.reshape(-1), metrics, blendshapes, np.ones(1, dtype=np.float32)
    ])
