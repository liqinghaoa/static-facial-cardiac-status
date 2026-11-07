"""Alignment and mask operations required by the three manuscript ROIs."""

from __future__ import annotations

from typing import Any, Iterable

import cv2

import numpy as np

from preprocessing import face_detection as alignment

from preprocessing import parsing_model as parsing

def estimate_original_coordinate_alignment(
    image_rgb: np.ndarray,
    selected_face: alignment.FaceDetection,
    face_mesh: Any,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Estimate and apply original-image -> output similarity alignment."""
    crop_x, crop_y, crop_width, crop_height = alignment.expand_bbox(
        selected_face.bbox, image_rgb.shape
    )
    if crop_width <= 1 or crop_height <= 1:
        raise alignment.SampleFailure(
            "failed_alignment", "expanded_face_bbox_is_empty"
        )
    crop_rgb = image_rgb[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ].copy()
    landmarks = alignment.run_facemesh(crop_rgb, face_mesh)
    if landmarks is None:
        raise alignment.SampleFailure(
            "failed_landmark_incomplete", "mediapipe_facemesh_returned_none"
        )
    crop_points, _, eye_distance = alignment.extract_alignment_landmarks(
        landmarks, crop_rgb.shape
    )
    original_points = crop_points + np.array(
        [float(crop_x), float(crop_y)], dtype=np.float32
    )
    matrix, parameters = alignment.estimate_similarity_transform(
        original_points,
        alignment.canonical_alignment_template(image_size),
    )
    aligned_rgb = cv2.warpAffine(
        image_rgb,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if aligned_rgb.shape != (image_size, image_size, 3):
        raise alignment.SampleFailure(
            "failed_alignment", "direct original-coordinate warp failed"
        )
    transformed_points = cv2.transform(
        original_points.reshape(1, -1, 2), matrix
    )[0]
    eye_y_after_align = float(transformed_points[:2, 1].mean())

    source_valid = np.full(image_rgb.shape[:2], 255, dtype=np.uint8)
    aligned_valid = cv2.warpAffine(
        source_valid,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    top_band_height = max(4, int(round(image_size * 0.05)))
    top_invalid_ratio = float(
        (aligned_valid[:top_band_height] == 0).mean()
    )
    parameters.update(
        {
            "alignment_source": "original_coordinate",
            "eye_distance": float(eye_distance),
            "eye_y_after_align": eye_y_after_align,
            "top_invalid_area_ratio": top_invalid_ratio,
            "top_cut_warning": bool(top_invalid_ratio > 0.20),
            "expanded_bbox_x": crop_x,
            "expanded_bbox_y": crop_y,
            "expanded_bbox_w": crop_width,
            "expanded_bbox_h": crop_height,
        }
    )
    return aligned_rgb, crop_rgb, parameters, matrix

def build_semantic_regularized_mask(selected_mask: np.ndarray) -> np.ndarray:
    """Lightly regularize semantics without replacing the natural face shape."""
    binary = (selected_mask > 0).astype(np.uint8) * 255
    if int((binary > 0).sum()) < max(64, int(binary.size * 0.0025)):
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "selected_semantic_mask_too_small"
        )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    regularized = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (regularized > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "no_semantic_face_component"
        )
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    regularized = (labels == largest).astype(np.uint8) * 255

    # Fill only small enclosed holes. Large hair-shaped forehead gaps remain
    # available for the explicit hybrid repair decision.
    inverse = (regularized == 0).astype(np.uint8)
    hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    max_hole_area = max(32, int(round(regularized.size * 0.003)))
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    hole_labels[0, :],
                    hole_labels[-1, :],
                    hole_labels[:, 0],
                    hole_labels[:, -1],
                ]
            )
        ).tolist()
    )
    for label_index in range(1, hole_count):
        if (
            label_index not in border_labels
            and int(hole_stats[label_index, cv2.CC_STAT_AREA]) <= max_hole_area
        ):
            regularized[hole_labels == label_index] = 255

    # A small blur/threshold smooths contour stair-stepping without dilation.
    smoothed = cv2.GaussianBlur(regularized, (5, 5), sigmaX=1.0)
    regularized = (smoothed >= 127).astype(np.uint8) * 255
    if int((regularized > 0).sum()) < max(64, int(regularized.size * 0.0025)):
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "semantic_regularized_mask_too_small"
        )
    return regularized

def compute_forehead_top_jaggedness(
    semantic_mask: np.ndarray,
    forehead_band_ratio: float,
) -> float:
    xmin, ymin, xmax, ymax = parsing._binary_bbox(semantic_mask)
    face_height = max(1, ymax - ymin)
    band_ymax = min(
        ymax, int(round(ymin + forehead_band_ratio * face_height))
    )
    top_values = np.full(xmax - xmin + 1, np.nan, dtype=np.float32)
    for offset, x in enumerate(range(xmin, xmax + 1)):
        ys = np.where(semantic_mask[ymin : band_ymax + 1, x] > 0)[0]
        if len(ys):
            top_values[offset] = float(ymin + ys.min())
    valid = np.isfinite(top_values)
    if int(valid.sum()) < 3:
        return 1.0
    indices = np.arange(len(top_values))
    top_values = np.interp(indices, indices[valid], top_values[valid])
    return float(np.mean(np.abs(np.diff(top_values))) / face_height)

def build_final_mask(
    semantic_mask: np.ndarray,
    candidate_envelope: np.ndarray,
    mode: str,
    forehead_band_ratio: float,
    hair_candidate_ratio: float,
    hair_repair_threshold: float,
    jaggedness: float,
    jaggedness_threshold: float,
    enable_jaggedness_trigger: bool,
) -> tuple[np.ndarray, bool, str]:
    if mode == "envelope":
        return candidate_envelope.copy(), False, "none"
    if mode == "semantic":
        return semantic_mask.copy(), False, "none"

    hair_trigger = hair_candidate_ratio > hair_repair_threshold
    jagged_trigger = (
        enable_jaggedness_trigger and jaggedness > jaggedness_threshold
    )
    if not hair_trigger and not jagged_trigger:
        return semantic_mask.copy(), False, "none"

    xmin, ymin, xmax, ymax = parsing._binary_bbox(semantic_mask)
    _, envelope_ymin, _, _ = parsing._binary_bbox(candidate_envelope)
    face_height = max(1, ymax - ymin)
    band_ymax = min(
        ymax, int(round(ymin + forehead_band_ratio * face_height))
    )
    # Start at the candidate envelope's rounded top instead of semantic ymin.
    # Starting exactly at semantic ymin can create an artificial horizontal
    # "flat-top" edge when a large hair notch triggered the repair.
    band_ymin = min(ymin, envelope_ymin)
    final_mask = semantic_mask.copy()
    final_mask[band_ymin : band_ymax + 1] = cv2.bitwise_or(
        semantic_mask[band_ymin : band_ymax + 1],
        candidate_envelope[band_ymin : band_ymax + 1],
    )
    if hair_trigger and jagged_trigger:
        reason = "hair_occlusion+jaggedness"
    elif hair_trigger:
        reason = "hair_occlusion"
    else:
        reason = "jaggedness"
    return final_mask, True, reason

def _ratio_inside_mask(
    label_map: np.ndarray,
    final_mask: np.ndarray,
    classes: Iterable[int],
) -> tuple[int, float]:
    foreground = final_mask > 0
    total = int(foreground.sum())
    if total == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "final_mask_is_empty"
        )
    pixels = int((np.isin(label_map, tuple(classes)) & foreground).sum())
    return pixels, pixels / float(total)
