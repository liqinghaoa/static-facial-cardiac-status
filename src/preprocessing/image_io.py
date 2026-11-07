from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .schemas import SampleFailure


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image_for_id(
    image_id: str,
    image_dir: Path,
    extensions: Iterable[str] = VALID_EXTENSIONS,
    filename: str | None = None,
) -> Path | None:
    if filename:
        nested_candidate = image_dir / image_id / filename
        if nested_candidate.is_file():
            return nested_candidate
    for extension in extensions:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def read_image_rgb_exif(path: Path) -> tuple[np.ndarray, bool]:
    try:
        with Image.open(path) as image:
            before = image.size
            transposed = ImageOps.exif_transpose(image)
            applied = transposed.size != before or transposed is not image
            rgb = np.asarray(transposed.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise SampleFailure("image_decode_failed", f"{type(exc).__name__}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SampleFailure("image_decode_failed", "decoded image is not RGB")
    return rgb, bool(applied)


def save_rgb_png(path: Path, image_rgb: np.ndarray) -> None:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SampleFailure("output_write_failed", "RGB PNG output must be HxWx3 uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise SampleFailure("output_write_failed", "cv2.imencode('.png') failed")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise SampleFailure("output_write_failed", str(exc)) from exc


def save_gray_png(path: Path, image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise SampleFailure("output_write_failed", "gray PNG output must be HxW uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise SampleFailure("output_write_failed", "cv2.imencode('.png') failed")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise SampleFailure("output_write_failed", str(exc)) from exc
