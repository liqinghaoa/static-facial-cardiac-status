"""Assemble de-identified Grad-CAM arrays into a six-panel spatial-response figure.

The manifest supplied by the user contains only panel labels and paths to approved
CAM arrays. It must not contain patient identifiers or original image filenames.
The CAM arrays are normally produced by the package's Grad-CAM evaluation entry
point; this script performs only the final figure composition.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PANELS = tuple("ABCDEF")
FACE_SIZE = (256, 320)


def read_manifest(path: Path) -> dict[str, Path]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"panel", "cam_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest must contain panel and cam_path columns")
    records: dict[str, Path] = {}
    for row in rows:
        panel = row["panel"].strip().upper()
        if panel not in PANELS or panel in records:
            raise ValueError("Manifest must contain one unique panel A-F")
        cam_path = Path(row["cam_path"]).expanduser()
        if not cam_path.is_absolute():
            cam_path = path.parent / cam_path
        records[panel] = cam_path.resolve()
    if set(records) != set(PANELS):
        raise ValueError("Manifest must contain exactly panels A-F")
    return records


def load_cam(path: Path) -> np.ndarray:
    cam = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if cam.shape != FACE_SIZE or not np.isfinite(cam).all():
        raise ValueError(f"Expected finite CAM with shape {FACE_SIZE}: {path}")
    low, high = float(cam.min()), float(cam.max())
    if high <= low:
        raise ValueError(f"CAM has no dynamic range: {path}")
    return (cam - low) / (high - low)


def build_overlay(template: np.ndarray, cam: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    binary = np.where(gray > 12, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a foreground contour in the supplied template")
    mask = np.zeros(FACE_SIZE, dtype=np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.7)
    heatmap = plt.get_cmap("jet")(cam)[..., :3].astype(np.float32)
    image = template.astype(np.float32) / 255.0
    return np.clip((0.55 * image + 0.45 * heatmap) * mask[..., None], 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template-image", type=Path, required=True,
                        help="Approved de-identified face template image.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    records = read_manifest(args.manifest.resolve())
    with Image.open(args.template_image) as image:
        template = np.asarray(image.convert("RGB"))
    template = cv2.resize(template, (FACE_SIZE[1], FACE_SIZE[0]), interpolation=cv2.INTER_AREA)
    overlays = {panel: build_overlay(template, load_cam(path)) for panel, path in records.items()}

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(10.42, 7.83), facecolor="white")
    figure.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025, wspace=0.06, hspace=0.10)
    for axis, panel in zip(axes.flat, PANELS):
        axis.imshow(overlays[panel], interpolation="nearest", origin="upper", aspect="equal")
        axis.text(0.015, 0.985, panel, transform=axis.transAxes, ha="left", va="top",
                  fontsize=13, fontweight="bold", color="black",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5})
        axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"COMPLETED: {output}")


if __name__ == "__main__":
    main()
