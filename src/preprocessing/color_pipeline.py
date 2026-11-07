from __future__ import annotations

import cv2
import numpy as np

from .schemas import SampleFailure


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    c = np.clip(srgb.astype(np.float32), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.clip(linear.astype(np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1.0 / 2.4)) - 0.055).astype(np.float32)


def uint8_rgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SampleFailure("invalid_output_shape", "input RGB must be HxWx3 uint8")
    return srgb_to_linear(rgb.astype(np.float32) / 255.0)


def linear_to_uint8_rgb(linear: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(linear)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def warp_linear_rgb(
    linear_rgb: np.ndarray,
    affine_source_to_canvas: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    aligned = cv2.warpAffine(
        linear_rgb.astype(np.float32),
        affine_source_to_canvas[:2, :].astype(np.float32),
        (int(output_width), int(output_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if aligned is None or aligned.shape != (output_height, output_width, 3):
        raise SampleFailure("warp_failed", "linear RGB warp produced invalid shape")
    if not np.isfinite(aligned).all():
        raise SampleFailure("nonfinite_linear_rgb", "aligned linear RGB contains non-finite values")
    return np.clip(aligned, 0.0, 1.0).astype(np.float32)


def encode_linear_outputs(aligned_linear_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if aligned_linear_rgb.ndim != 3 or aligned_linear_rgb.shape[2] != 3:
        raise SampleFailure("invalid_output_shape", "aligned linear RGB must be HxWx3")
    aligned_srgb = linear_to_uint8_rgb(aligned_linear_rgb)
    chw_float16 = np.transpose(aligned_linear_rgb, (2, 0, 1)).astype(np.float16)
    return aligned_srgb, chw_float16
