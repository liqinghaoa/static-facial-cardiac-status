from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import face_parser, facemesh_regions as regions
from .image_io import save_rgb_png


def resize_panel(image: np.ndarray, size: tuple[int, int] = (360, 450)) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image_rgb.copy()
    hit = mask > 0
    color_arr = np.array(color, dtype=np.float32)
    out[hit] = np.clip(out[hit].astype(np.float32) * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)
    return out


def draw_polyline(image_rgb: np.ndarray, points: np.ndarray, color: tuple[int, int, int], closed: bool = True, thickness: int = 2) -> np.ndarray:
    out = image_rgb.copy()
    cv2.polylines(out, [np.rint(points).astype(np.int32)], closed, color, thickness, lineType=cv2.LINE_AA)
    return out


def label_panel(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, title[:42], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_detection(image_rgb: np.ndarray, detections: list, selected: object | None) -> np.ndarray:
    panel = image_rgb.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = (0, 255, 0) if selected is not None and det.index == selected.index else (255, 180, 0)
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, max(2, image_rgb.shape[1] // 500))
    return panel


def draw_source_geometry(image_rgb: np.ndarray, source_landmarks: np.ndarray, crop_corners: np.ndarray | None = None) -> np.ndarray:
    panel = image_rgb.copy()
    pts = np.rint(source_landmarks).astype(np.int32)
    stride = max(1, len(pts) // 120)
    for point in pts[::stride]:
        cv2.circle(panel, tuple(point), max(1, image_rgb.shape[1] // 900), (0, 220, 255), -1)
    oval = source_landmarks[np.asarray(regions.FACE_OVAL_INDICES)]
    panel = draw_polyline(panel, oval, (0, 255, 0), True, max(2, image_rgb.shape[1] // 600))
    if crop_corners is not None:
        panel = draw_polyline(panel, crop_corners, (255, 80, 0), True, max(2, image_rgb.shape[1] // 600))
    return panel


def draw_canvas_regions(image_rgb: np.ndarray, canvas_landmarks: np.ndarray) -> np.ndarray:
    panel = image_rgb.copy()
    panel = draw_polyline(panel, canvas_landmarks[np.asarray(regions.FACE_OVAL_INDICES)], (0, 255, 0), True, 2)
    left_eye = canvas_landmarks[np.asarray(regions.IMAGE_LEFT_EYE_INDICES)].mean(axis=0)
    right_eye = canvas_landmarks[np.asarray(regions.IMAGE_RIGHT_EYE_INDICES)].mean(axis=0)
    cv2.line(panel, tuple(np.rint(left_eye).astype(int)), tuple(np.rint(right_eye).astype(int)), (255, 255, 0), 2, lineType=cv2.LINE_AA)
    for name, indices in regions.FACEMESH_EXCLUSION_POLYGONS.items():
        color = (255, 0, 0) if "eye" in name else (255, 120, 0) if "brow" in name else (0, 0, 255) if "lip" in name else (0, 255, 255)
        panel = draw_polyline(panel, canvas_landmarks[np.asarray(indices)], color, True, 2)
    return panel


def make_preview_panel(
    path: Path,
    original_rgb: np.ndarray,
    detections: list,
    selected: object | None,
    source_landmarks: np.ndarray,
    source_crop_corners: np.ndarray,
    aligned_srgb: np.ndarray,
    canvas_landmarks: np.ndarray,
    parsing_label: np.ndarray,
    skin_candidate: np.ndarray,
    semantic_exclusion: np.ndarray,
    facemesh_exclusion: np.ndarray,
    skin_valid_mask: np.ndarray,
    masked_skin_preview: np.ndarray,
    source_valid_mask: np.ndarray,
) -> None:
    exclusion = cv2.bitwise_or(semantic_exclusion, facemesh_exclusion)
    panels = [
        label_panel(resize_panel(draw_detection(original_rgb, detections, selected)), "source + selected detection"),
        label_panel(resize_panel(draw_source_geometry(original_rgb, source_landmarks, source_crop_corners)), "source + FaceMesh + 4:5 crop"),
        label_panel(resize_panel(aligned_srgb), "aligned_srgb"),
        label_panel(resize_panel(draw_canvas_regions(aligned_srgb, canvas_landmarks)), "aligned + regions"),
        label_panel(resize_panel(face_parser.colorize(parsing_label)), "parsing_label"),
        label_panel(resize_panel(overlay_mask(aligned_srgb, skin_candidate, (40, 220, 90))), "skin candidate"),
        label_panel(resize_panel(overlay_mask(aligned_srgb, exclusion, (255, 80, 30))), "semantic/facemesh exclusion"),
        label_panel(resize_panel(overlay_mask(aligned_srgb, skin_valid_mask, (0, 255, 180))), "skin_valid_mask"),
        label_panel(resize_panel(masked_skin_preview), "masked_skin_preview"),
        label_panel(resize_panel(overlay_mask(aligned_srgb, 255 - source_valid_mask, (255, 0, 0))), "source_valid_mask border"),
    ]
    row1 = np.concatenate(panels[:5], axis=1)
    row2 = np.concatenate(panels[5:], axis=1)
    save_rgb_png(path, np.concatenate([row1, row2], axis=0))


def make_contact_sheet(panel_paths: list[Path], output_path: Path, columns: int = 4) -> None:
    images = []
    for path in panel_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(resize_panel(rgb, (420, 210)))
    if not images:
        return
    rows = []
    for start in range(0, len(images), columns):
        chunk = images[start : start + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros_like(images[0]))
        rows.append(np.concatenate(chunk, axis=1))
    save_rgb_png(output_path, np.concatenate(rows, axis=0))
