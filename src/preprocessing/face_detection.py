"""Face detection, landmarks and similarity geometry used by current preprocessing."""

from __future__ import annotations

import math

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable, Sequence

import cv2

import numpy as np

FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)

IMAGE_LEFT_EYE_INDICES = (33, 133, 159, 145, 153, 154, 155)

IMAGE_RIGHT_EYE_INDICES = (362, 263, 386, 374, 380, 381, 382)

NOSE_TIP_INDEX = 1

IMAGE_LEFT_MOUTH_INDEX = 61

IMAGE_RIGHT_MOUTH_INDEX = 291

@dataclass(frozen=True)
class FaceDetection:
    """One clipped face detection in source-image pixel coordinates."""

    index: int
    bbox: tuple[int, int, int, int]
    confidence: float

class SampleFailure(RuntimeError):
    """Expected per-sample failure with an explicit pipeline status."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason

def read_image_rgb(path: Path) -> np.ndarray | None:
    """Read an image through imdecode for reliable Unicode paths on Windows."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def detect_faces(
    image_rgb: np.ndarray,
    detector: Any,
) -> list[FaceDetection]:
    """Run MediaPipe face detection and return clipped pixel bounding boxes."""
    height, width = image_rgb.shape[:2]
    results = detector.process(image_rgb)
    detections: list[FaceDetection] = []
    for index, detection in enumerate(results.detections or []):
        relative = detection.location_data.relative_bounding_box
        x1 = max(0, int(math.floor(relative.xmin * width)))
        y1 = max(0, int(math.floor(relative.ymin * height)))
        x2 = min(width, int(math.ceil((relative.xmin + relative.width) * width)))
        y2 = min(height, int(math.ceil((relative.ymin + relative.height) * height)))
        if x2 <= x1 or y2 <= y1:
            continue
        scores = list(detection.score or [])
        confidence = float(scores[0]) if scores else float("nan")
        detections.append(
            FaceDetection(index=index, bbox=(x1, y1, x2 - x1, y2 - y1), confidence=confidence)
        )
    return detections

def select_face(
    detections: Sequence[FaceDetection],
    image_shape: Sequence[int],
) -> FaceDetection | None:
    """Select a large, central face using a deterministic combined score."""
    if not detections:
        return None
    height, width = image_shape[:2]
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    diagonal = max(math.hypot(width, height), 1.0)

    def rank(item: FaceDetection) -> tuple[float, float, int]:
        x, y, w, h = item.bbox
        area_ratio = (w * h) / float(width * height)
        center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
        center_distance = float(np.linalg.norm(center - image_center)) / diagonal
        combined_score = area_ratio / (1.0 + center_distance)
        return combined_score, area_ratio, -item.index

    return max(detections, key=rank)

def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.20,
    side_ratio: float = 0.20,
) -> tuple[int, int, int, int]:
    """Expand a face box and clip it to image boundaries."""
    x, y, width, height = bbox
    image_height, image_width = image_shape[:2]
    x1 = max(0, int(math.floor(x - side_ratio * width)))
    x2 = min(image_width, int(math.ceil(x + width + side_ratio * width)))
    y1 = max(0, int(math.floor(y - top_ratio * height)))
    y2 = min(image_height, int(math.ceil(y + height + bottom_ratio * height)))
    return x1, y1, x2 - x1, y2 - y1

def run_facemesh(crop_rgb: np.ndarray, face_mesh: Any) -> list[Any] | None:
    """Run strict FaceMesh on the expanded crop without fallback behavior."""
    results = face_mesh.process(crop_rgb)
    if not results.multi_face_landmarks:
        return None
    return list(results.multi_face_landmarks[0].landmark)

def _mean_landmarks(
    landmarks: Sequence[Any],
    indices: Iterable[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = np.array(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
        dtype=np.float32,
    )
    return points.mean(axis=0)

def _single_landmark(
    landmarks: Sequence[Any],
    index: int,
    width: int,
    height: int,
) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

def extract_alignment_landmarks(
    landmarks: Sequence[Any],
    crop_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract robust eye centers, nose, mouth corners, and oval points."""
    required_indices = set(
        IMAGE_LEFT_EYE_INDICES
        + IMAGE_RIGHT_EYE_INDICES
        + (
            NOSE_TIP_INDEX,
            IMAGE_LEFT_MOUTH_INDEX,
            IMAGE_RIGHT_MOUTH_INDEX,
        )
        + FACE_OVAL_INDICES
    )
    if not landmarks or max(required_indices) >= len(landmarks):
        raise SampleFailure(
            "failed_landmark_incomplete",
            f"FaceMesh returned {len(landmarks) if landmarks else 0} landmarks",
        )

    height, width = crop_shape[:2]
    image_left_eye = _mean_landmarks(
        landmarks, IMAGE_LEFT_EYE_INDICES, width, height
    )
    image_right_eye = _mean_landmarks(
        landmarks, IMAGE_RIGHT_EYE_INDICES, width, height
    )
    nose = _single_landmark(landmarks, NOSE_TIP_INDEX, width, height)
    image_left_mouth = _single_landmark(
        landmarks, IMAGE_LEFT_MOUTH_INDEX, width, height
    )
    image_right_mouth = _single_landmark(
        landmarks, IMAGE_RIGHT_MOUTH_INDEX, width, height
    )
    alignment_points = np.vstack(
        [image_left_eye, image_right_eye, nose, image_left_mouth, image_right_mouth]
    ).astype(np.float32)
    oval_points = np.array(
        [
            [landmarks[index].x * width, landmarks[index].y * height]
            for index in FACE_OVAL_INDICES
        ],
        dtype=np.float32,
    )

    if not np.isfinite(alignment_points).all() or not np.isfinite(oval_points).all():
        raise SampleFailure(
            "failed_landmark_incomplete", "FaceMesh contains non-finite coordinates"
        )
    eye_distance = float(np.linalg.norm(image_right_eye - image_left_eye))
    if eye_distance < 1.0 or cv2.contourArea(oval_points) <= 1.0:
        raise SampleFailure(
            "failed_landmark_incomplete", "Degenerate eye distance or face oval"
        )
    return alignment_points, oval_points, eye_distance

def canonical_alignment_template(image_size: int) -> np.ndarray:
    """Return five canonical points in the output coordinate system."""
    normalized = np.array(
        [
            [0.35, 0.38],
            [0.65, 0.38],
            [0.50, 0.55],
            [0.40, 0.72],
            [0.60, 0.72],
        ],
        dtype=np.float32,
    )
    return normalized * float(image_size - 1)

def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a rotation + uniform-scale + translation transform.

    ``estimateAffinePartial2D`` constrains the linear block to
    [[a, -b], [b, a]]. Unlike a full affine transform, this cannot introduce
    shear or independent x/y scaling, so facial width/height proportions and
    local phenotype geometry are not warped.
    """
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.LMEDS,
        refineIters=10,
    )
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise SampleFailure("failed_alignment", "Similarity transform estimation failed")

    a, minus_b, translation_x = matrix[0]
    b, a_second, translation_y = matrix[1]
    scale = float(math.hypot(a, b))
    if scale <= 0 or not math.isfinite(scale):
        raise SampleFailure("failed_alignment", "Invalid similarity scale")

    # Guard the mathematical constraint explicitly even if OpenCV changes.
    tolerance = max(1e-5, scale * 1e-4)
    if abs(a - a_second) > tolerance or abs(minus_b + b) > tolerance:
        raise SampleFailure(
            "failed_alignment", "Estimated matrix is not a strict similarity transform"
        )
    if inliers is not None and int(np.asarray(inliers).sum()) < 2:
        raise SampleFailure("failed_alignment", "Insufficient alignment inliers")

    parameters = {
        "rotation_angle": float(math.degrees(math.atan2(b, a))),
        "scale_factor": scale,
        "translation_x": float(translation_x),
        "translation_y": float(translation_y),
    }
    return matrix.astype(np.float32), parameters

def save_png(output_path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB uint8 image as PNG, including Unicode-safe Windows paths."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SampleFailure("failed_save", "Output must be HxWx3 uint8 RGB")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise SampleFailure("failed_save", "cv2.imencode('.png') failed")
    try:
        output_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise SampleFailure("failed_save", str(exc)) from exc
