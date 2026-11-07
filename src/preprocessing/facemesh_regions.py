from __future__ import annotations

from typing import Iterable

import numpy as np


# Reused from preprocessing/build_global_face_oval_blackbg_png_simalign_strict.py.
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
CHIN_INDEX = 152
IMAGE_LEFT_MOUTH_INDEX = 61
IMAGE_RIGHT_MOUTH_INDEX = 291

LEFT_EYE_POLYGON = (
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
)
RIGHT_EYE_POLYGON = (
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
)
LEFT_BROW_POLYGON = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_BROW_POLYGON = (336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
LIPS_POLYGON = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)

# MediaPipe does not expose a single official nostril polygon. These compact
# project-local polygons use stable nose-wing/base landmarks and are drawn in QC.
LEFT_NOSTRIL_POLYGON = (49, 48, 64, 98, 97, 2, 94, 129)
RIGHT_NOSTRIL_POLYGON = (279, 278, 294, 327, 326, 2, 94, 358)

FACEMESH_EXCLUSION_POLYGONS = {
    "left_eye": LEFT_EYE_POLYGON,
    "right_eye": RIGHT_EYE_POLYGON,
    "left_brow": LEFT_BROW_POLYGON,
    "right_brow": RIGHT_BROW_POLYGON,
    "lips_mouth": LIPS_POLYGON,
    "left_nostril": LEFT_NOSTRIL_POLYGON,
    "right_nostril": RIGHT_NOSTRIL_POLYGON,
}


def require_indices(indices: Iterable[int], landmarks: np.ndarray) -> None:
    max_index = max(indices)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or max_index >= landmarks.shape[0]:
        raise ValueError(f"FaceMesh landmarks do not contain index {max_index}")


def points_for(indices: Iterable[int], landmarks: np.ndarray) -> np.ndarray:
    require_indices(indices, landmarks)
    return landmarks[np.asarray(tuple(indices), dtype=np.int32), :].astype(np.float32)
