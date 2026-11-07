from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from . import facemesh_regions as regions
from .schemas import GeometryConfig, SampleFailure


@dataclass(frozen=True)
class GeometryResult:
    source_landmarks: np.ndarray
    canvas_landmarks: np.ndarray
    roll_matrix: np.ndarray
    affine_source_to_canvas: np.ndarray
    affine_canvas_to_source: np.ndarray
    crop_left: float
    crop_top: float
    crop_width: float
    crop_height: float
    uniform_scale: float
    roll_angle_degrees: float
    residual_roll_degrees: float
    rolled_eye_y_delta: float
    rolled_face_left: float
    rolled_face_right: float
    rolled_face_top: float
    rolled_chin_y: float
    matrix_singular_values: tuple[float, float]
    roundtrip_max_error: float


def landmarks_to_source_array(landmarks: Sequence[Any], crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> np.ndarray:
    if not landmarks:
        raise SampleFailure("facemesh_failed", "FaceMesh returned no landmarks")
    points = np.array(
        [[float(lm.x) * crop_width + crop_x, float(lm.y) * crop_height + crop_y] for lm in landmarks],
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise SampleFailure("invalid_landmarks", "FaceMesh landmarks contain non-finite values")
    if points.shape[0] <= max(regions.FACE_OVAL_INDICES):
        raise SampleFailure("invalid_landmarks", f"FaceMesh returned only {points.shape[0]} landmarks")
    return points


def mean_points(landmarks: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return regions.points_for(indices, landmarks).mean(axis=0)


def extract_key_points(landmarks: np.ndarray) -> dict[str, Any]:
    left_eye = mean_points(landmarks, regions.IMAGE_LEFT_EYE_INDICES)
    right_eye = mean_points(landmarks, regions.IMAGE_RIGHT_EYE_INDICES)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < 1.0:
        raise SampleFailure("degenerate_eye_distance", f"eye distance too small: {eye_distance:.6f}")
    face_oval = regions.points_for(regions.FACE_OVAL_INDICES, landmarks)
    if cv2.contourArea(face_oval.astype(np.float32)) <= 1.0:
        raise SampleFailure("invalid_face_oval", "Face oval contour area is degenerate")
    return {
        "left_eye": left_eye.astype(np.float32),
        "right_eye": right_eye.astype(np.float32),
        "nose_tip": regions.points_for((regions.NOSE_TIP_INDEX,), landmarks)[0],
        "left_mouth": regions.points_for((regions.IMAGE_LEFT_MOUTH_INDEX,), landmarks)[0],
        "right_mouth": regions.points_for((regions.IMAGE_RIGHT_MOUTH_INDEX,), landmarks)[0],
        "chin": regions.points_for((regions.CHIN_INDEX,), landmarks)[0],
        "face_oval": face_oval.astype(np.float32),
        "eye_distance": eye_distance,
    }


def _to_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = np.asarray(matrix_2x3, dtype=np.float64)
    return matrix


def transform_points(points: np.ndarray, matrix_3x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homo @ np.asarray(matrix_3x3, dtype=np.float64).T
    return out[:, :2].astype(np.float32)


def compute_crop_from_rolled(
    rolled_landmarks: np.ndarray,
    cfg: GeometryConfig,
) -> dict[str, float]:
    oval = regions.points_for(regions.FACE_OVAL_INDICES, rolled_landmarks)
    x_left = float(np.quantile(oval[:, 0], cfg.face_quantile_low))
    x_right = float(np.quantile(oval[:, 0], cfg.face_quantile_high))
    face_top = float(np.quantile(oval[:, 1], cfg.face_quantile_low))
    chin_y = float(rolled_landmarks[regions.CHIN_INDEX, 1])
    face_width = x_right - x_left
    if face_width <= 1.0:
        raise SampleFailure("invalid_crop_geometry", "rolled face width is degenerate")
    crop_width = face_width / (1.0 - 2.0 * cfg.side_margin_ratio)
    crop_height = crop_width * (cfg.output_height / cfg.output_width)
    crop_center_x = (x_left + x_right) / 2.0
    crop_bottom = chin_y + cfg.bottom_margin_ratio * crop_height
    needed_height = (crop_bottom - face_top) / max(1.0e-6, 1.0 - cfg.top_margin_ratio)
    if needed_height > crop_height:
        crop_height = needed_height
        crop_width = crop_height * (cfg.output_width / cfg.output_height)
    crop_left = crop_center_x - crop_width / 2.0
    crop_top = crop_bottom - crop_height
    if not np.isfinite([crop_left, crop_top, crop_width, crop_height]).all():
        raise SampleFailure("invalid_crop_geometry", "crop geometry contains non-finite values")
    if crop_width <= 1.0 or crop_height <= 1.0:
        raise SampleFailure("invalid_crop_geometry", "crop dimensions are too small")
    return {
        "crop_left": float(crop_left),
        "crop_top": float(crop_top),
        "crop_width": float(crop_width),
        "crop_height": float(crop_height),
        "crop_center_x": float(crop_center_x),
        "rolled_face_left": x_left,
        "rolled_face_right": x_right,
        "rolled_face_top": face_top,
        "rolled_chin_y": chin_y,
    }


def validate_similarity_matrix(matrix_3x3: np.ndarray, cfg: GeometryConfig) -> tuple[float, float]:
    matrix = np.asarray(matrix_3x3, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise SampleFailure("invalid_affine_matrix", "affine matrix is not finite 3x3")
    linear = matrix[:2, :2]
    u, singular_values, vt = np.linalg.svd(linear)
    if min(singular_values) <= 0:
        raise SampleFailure("invalid_affine_matrix", "affine matrix is singular")
    if abs(float(singular_values[0] - singular_values[1])) > cfg.matrix_atol * max(1.0, float(singular_values[0])):
        raise SampleFailure("invalid_affine_matrix", "affine linear block is not uniform scale")
    if not np.allclose(linear.T @ linear, np.eye(2) * float(singular_values[0] ** 2), atol=cfg.matrix_atol, rtol=cfg.matrix_atol):
        raise SampleFailure("invalid_affine_matrix", "affine linear block contains shear")
    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise SampleFailure("invalid_affine_matrix", "affine matrix is not invertible") from exc
    return float(singular_values[0]), float(singular_values[1])


def build_geometry(source_landmarks: np.ndarray, cfg: GeometryConfig) -> GeometryResult:
    keys = extract_key_points(source_landmarks)
    left_eye = keys["left_eye"]
    right_eye = keys["right_eye"]
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    roll_angle = math.atan2(dy, dx)
    roll_degrees = math.degrees(roll_angle)
    eye_mid = ((left_eye + right_eye) / 2.0).astype(np.float32)
    roll_matrix = _to_homogeneous(cv2.getRotationMatrix2D(tuple(map(float, eye_mid)), roll_degrees, 1.0))
    rolled = transform_points(source_landmarks, roll_matrix)
    rolled_left = transform_points(left_eye.reshape(1, 2), roll_matrix)[0]
    rolled_right = transform_points(right_eye.reshape(1, 2), roll_matrix)[0]
    rolled_eye_y_delta = float(rolled_right[1] - rolled_left[1])
    residual = math.degrees(math.atan2(rolled_eye_y_delta, float(rolled_right[0] - rolled_left[0])))
    crop = compute_crop_from_rolled(rolled, cfg)
    scale_x = cfg.output_width / crop["crop_width"]
    scale_y = cfg.output_height / crop["crop_height"]
    if not math.isclose(scale_x, scale_y, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise SampleFailure("invalid_crop_geometry", "crop aspect ratio is not 4:5")
    translate = np.array(
        [[1.0, 0.0, -crop["crop_left"]], [0.0, 1.0, -crop["crop_top"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale = np.array([[scale_x, 0.0, 0.0], [0.0, scale_x, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    affine_source_to_canvas = scale @ translate @ roll_matrix
    singular_values = validate_similarity_matrix(affine_source_to_canvas, cfg)
    affine_canvas_to_source = np.linalg.inv(affine_source_to_canvas)
    canvas_landmarks = transform_points(source_landmarks, affine_source_to_canvas)
    roundtrip = transform_points(canvas_landmarks, affine_canvas_to_source)
    roundtrip_error = float(np.max(np.linalg.norm(roundtrip - source_landmarks, axis=1)))
    if roundtrip_error > cfg.roundtrip_tolerance_px:
        raise SampleFailure("invalid_affine_matrix", f"landmark roundtrip error {roundtrip_error:.6f}px")
    return GeometryResult(
        source_landmarks=source_landmarks.astype(np.float32),
        canvas_landmarks=canvas_landmarks.astype(np.float32),
        roll_matrix=roll_matrix.astype(np.float64),
        affine_source_to_canvas=affine_source_to_canvas.astype(np.float64),
        affine_canvas_to_source=affine_canvas_to_source.astype(np.float64),
        crop_left=crop["crop_left"],
        crop_top=crop["crop_top"],
        crop_width=crop["crop_width"],
        crop_height=crop["crop_height"],
        uniform_scale=float(scale_x),
        roll_angle_degrees=float(roll_degrees),
        residual_roll_degrees=float(residual),
        rolled_eye_y_delta=rolled_eye_y_delta,
        rolled_face_left=crop["rolled_face_left"],
        rolled_face_right=crop["rolled_face_right"],
        rolled_face_top=crop["rolled_face_top"],
        rolled_chin_y=crop["rolled_chin_y"],
        matrix_singular_values=singular_values,
        roundtrip_max_error=roundtrip_error,
    )
