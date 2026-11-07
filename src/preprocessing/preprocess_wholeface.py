"""Build 256x320 black-background real-face images from raw phone photos.

Scheme B mask keeps facial semantic features (skin, nose, brows, eyes, mouth,
lips) and removes hair, background, ears, neck, clothes, hats and glasses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from preprocessing import face_detection as legacy_alignment

from preprocessing import color_pipeline, face_geometry, face_parser, image_io, qc, skin_mask
from preprocessing.schemas import DetectionConfig, GeometryConfig, ParserConfig, SampleFailure


DEFAULT_CONFIG = Path("../configs/preprocess_wholeface_with_r3dpr.yaml")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_under_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(config_path: Path | None, root: Path) -> dict[str, Any]:
    path = resolve_under_root(config_path or DEFAULT_CONFIG, root)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping")
    loaded["_config_path"] = str(path)
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--full-500",
        action="store_true",
        help="Process every row in the fixed 500-case split and write full_output_dir.",
    )
    parser.add_argument("--pilot-count", type=int, default=None)
    parser.add_argument("--pilot-id-file", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite-confirmed", action="store_true")
    return parser


def dataclass_from_config(config: dict[str, Any], key: str, cls: Any) -> Any:
    values = config.get(key, {}) or {}
    return cls(**values)


def validate_config(config: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    paths = config.get("paths", {}) or {}
    image_dir = resolve_under_root(paths.get("image_dir"), root)
    image_filename = paths.get("image_filename")
    if image_filename is not None and not isinstance(image_filename, str):
        raise ValueError("paths.image_filename must be a string when provided")
    split_csv = resolve_under_root(paths.get("split_csv"), root)
    output_key = "full_output_dir" if args.full_500 else "pilot_output_dir"
    output_dir = resolve_under_root(paths.get(output_key), root)
    parser_cfg = dataclass_from_config(config, "parser", ParserConfig)
    if args.device:
        parser_cfg = ParserConfig(model=parser_cfg.model, checkpoint=parser_cfg.checkpoint, device=args.device)
    checkpoint = resolve_under_root(parser_cfg.checkpoint, root)
    geometry_cfg = dataclass_from_config(config, "geometry", GeometryConfig)
    detection_cfg = dataclass_from_config(config, "detection", DetectionConfig)
    if geometry_cfg.output_width != 256 or geometry_cfg.output_height != 320:
        raise ValueError("This pipeline requires output_width=256 and output_height=320")
    if args.pilot_count is not None and args.pilot_count <= 0:
        raise ValueError("--pilot-count must be positive")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {split_csv}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"BiSeNet checkpoint does not exist: {checkpoint}. "
            "Third-party weights are not bundled; see README.md "
            "in the package root for acquisition and placement."
        )
    return {
        "image_dir": image_dir,
        "image_filename": image_filename,
        "split_csv": split_csv,
        "output_dir": output_dir,
        "checkpoint": checkpoint,
        "geometry": geometry_cfg,
        "detection": detection_cfg,
        "parser": parser_cfg,
        "min_component_area_ratio": float((config.get("face_mask", {}) or {}).get("min_component_area_ratio", 0.0002)),
        "pilot_count": int(args.pilot_count or (config.get("pilot", {}) or {}).get("count", 32)),
        "full_500": bool(args.full_500),
        "config_path": Path(config["_config_path"]),
    }


def prepare_output_tree(output_dir: Path, root: Path, overwrite_confirmed: bool) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    allowed_output_roots = (
        (root / "runs").resolve(),
        (root / "data" / "processed" / "global_face").resolve(),
        (root / "data" / "processed" / "global_face_R3DPR").resolve(),
        (root / "data" / "processed" / "global_face_R3DPR_v2").resolve(),
    )
    if not any(allowed_root in output_dir.parents for allowed_root in allowed_output_roots):
        allowed = ", ".join(str(path) for path in allowed_output_roots)
        raise ValueError(f"Output must stay under one of: {allowed}. Got: {output_dir}")
    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite_confirmed:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite-confirmed to rebuild.")
        if any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
    names = (
        "images",
        "face_valid_mask",
        "source_valid_mask",
        "parsing_label",
        "aligned_srgb",
        "metadata",
        "qc/preview_panels",
        "qc/pilot_contact_sheets",
        "manifests",
        "config",
    )
    dirs = {"root": output_dir}
    for name in names:
        key = name.replace("/", "_")
        dirs[key] = output_dir / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def select_pilot_ids(split_df: pd.DataFrame, count: int, pilot_id_file: Path | None) -> list[str]:
    if pilot_id_file is not None:
        ids = [line.strip() for line in pilot_id_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return ids[:count]
    ordered = split_df.assign(ID=split_df["ID"].astype(str)).sort_values("ID", kind="mergesort").reset_index(drop=True)
    if len(ordered) < count:
        raise ValueError(f"Requested {count} IDs but split has {len(ordered)}")
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return ordered.iloc[indices]["ID"].astype(str).tolist()


def warp_source_valid(source_shape: tuple[int, int], matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    source_mask = np.full(source_shape, 255, dtype=np.uint8)
    warped = cv2.warpAffine(
        source_mask,
        matrix[:2, :].astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return ((warped > 0).astype(np.uint8) * 255)


def source_valid_stats(mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    h, w = mask.shape
    band_h = max(1, int(round(h * 0.05)))
    band_w = max(1, int(round(w * 0.05)))
    return {
        "source_valid_fraction": float(valid.mean()),
        "top_invalid_fraction": float((~valid[:band_h]).mean()),
        "bottom_invalid_fraction": float((~valid[-band_h:]).mean()),
        "left_invalid_fraction": float((~valid[:, :band_w]).mean()),
        "right_invalid_fraction": float((~valid[:, -band_w:]).mean()),
    }


def face_region_mask(shape: tuple[int, int], canvas_landmarks: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    oval = canvas_landmarks[np.asarray(face_geometry.regions.FACE_OVAL_INDICES)]
    cv2.fillPoly(mask, [np.rint(oval).astype(np.int32)], 255)
    return mask


def crop_corners_in_source(geom: face_geometry.GeometryResult) -> np.ndarray:
    corners = np.array(
        [
            [geom.crop_left, geom.crop_top],
            [geom.crop_left + geom.crop_width, geom.crop_top],
            [geom.crop_left + geom.crop_width, geom.crop_top + geom.crop_height],
            [geom.crop_left, geom.crop_top + geom.crop_height],
        ],
        dtype=np.float32,
    )
    return face_geometry.transform_points(corners, np.linalg.inv(geom.roll_matrix))


def make_scheme_b_qc(
    path: Path,
    original_rgb: np.ndarray,
    detections: list[Any],
    selected: Any,
    source_landmarks: np.ndarray,
    crop_corners: np.ndarray,
    aligned_srgb: np.ndarray,
    canvas_landmarks: np.ndarray,
    parsing_label: np.ndarray,
    face_keep_mask: np.ndarray,
    face_exclude_mask: np.ndarray,
    face_valid_mask: np.ndarray,
    blackbg: np.ndarray,
    source_valid: np.ndarray,
) -> None:
    panels = [
        qc.label_panel(qc.resize_panel(qc.draw_detection(original_rgb, detections, selected), (256, 320)), "source + selected detection"),
        qc.label_panel(qc.resize_panel(qc.draw_source_geometry(original_rgb, source_landmarks, crop_corners), (256, 320)), "source + FaceMesh + crop"),
        qc.label_panel(qc.resize_panel(aligned_srgb, (256, 320)), "aligned_srgb 256x320"),
        qc.label_panel(qc.resize_panel(qc.draw_canvas_regions(aligned_srgb, canvas_landmarks), (256, 320)), "aligned + regions"),
        qc.label_panel(qc.resize_panel(face_parser.colorize(parsing_label), (256, 320)), "parsing_label"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned_srgb, face_keep_mask, (0, 220, 80)), (256, 320)), "scheme B keep classes"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned_srgb, face_exclude_mask, (255, 70, 40)), (256, 320)), "removed classes"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned_srgb, face_valid_mask, (0, 255, 180)), (256, 320)), "face_valid_mask overlay"),
        qc.label_panel(qc.resize_panel(blackbg, (256, 320)), "final blackbg image"),
        qc.label_panel(qc.resize_panel(qc.overlay_mask(aligned_srgb, 255 - source_valid, (255, 0, 0)), (256, 320)), "source_valid border"),
    ]
    row1 = np.concatenate(panels[:5], axis=1)
    row2 = np.concatenate(panels[5:], axis=1)
    image_io.save_rgb_png(path, np.concatenate([row1, row2], axis=0))


def process_one(
    split_row: dict[str, Any],
    dirs: dict[str, Path],
    image_dir: Path,
    detector: Any,
    mesh: Any,
    parser_model: Any,
    parser_device: Any,
    geometry_cfg: GeometryConfig,
    min_component_area_ratio: float,
    image_filename: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(split_row["ID"])
    manifest = dict(split_row)
    metadata: dict[str, Any] = {"sample_id": sample_id, "status": "failed", "failure_code": "", "failure_detail": "", "warning_codes": []}
    outputs = {
        "image_path": str(dirs["images"] / f"{sample_id}.png"),
        "face_valid_mask_path": str(dirs["face_valid_mask"] / f"{sample_id}.png"),
        "source_valid_mask_path": str(dirs["source_valid_mask"] / f"{sample_id}.png"),
        "parsing_label_path": str(dirs["parsing_label"] / f"{sample_id}.png"),
        "aligned_srgb_path": str(dirs["aligned_srgb"] / f"{sample_id}.png"),
        "metadata_path": str(dirs["metadata"] / f"{sample_id}.json"),
        "qc_preview_path": str(dirs["qc_preview_panels"] / f"{sample_id}.png"),
    }
    manifest.update(outputs)
    try:
        source_path = image_io.find_image_for_id(sample_id, image_dir, filename=image_filename)
        if source_path is None:
            raise SampleFailure("missing_source_image", "cannot find image by ID")
        rgb, exif_applied = image_io.read_image_rgb_exif(source_path)
        h, w = rgb.shape[:2]
        metadata.update({"source_path": str(source_path), "source_width": w, "source_height": h, "exif_orientation_applied": exif_applied})
        detections = legacy_alignment.detect_faces(rgb, detector)
        selected = legacy_alignment.select_face(detections, rgb.shape)
        if selected is None:
            raise SampleFailure("face_detection_failed", "MediaPipe Face Detection returned no valid face")
        x, y, bw, bh = selected.bbox
        crop_x, crop_y, crop_w, crop_h = legacy_alignment.expand_bbox(selected.bbox, rgb.shape)
        crop_rgb = rgb[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        landmarks = legacy_alignment.run_facemesh(crop_rgb, mesh)
        if landmarks is None:
            raise SampleFailure("facemesh_failed", "MediaPipe FaceMesh returned no landmarks")
        source_landmarks = face_geometry.landmarks_to_source_array(landmarks, crop_x, crop_y, crop_w, crop_h)
        keys = face_geometry.extract_key_points(source_landmarks)
        geom = face_geometry.build_geometry(source_landmarks, geometry_cfg)
        linear = color_pipeline.uint8_rgb_to_linear(rgb)
        aligned_linear = color_pipeline.warp_linear_rgb(linear, geom.affine_source_to_canvas, geometry_cfg.output_width, geometry_cfg.output_height)
        aligned_srgb = color_pipeline.linear_to_uint8_rgb(aligned_linear)
        expected_shape = (geometry_cfg.output_height, geometry_cfg.output_width)
        source_valid = warp_source_valid((h, w), geom.affine_source_to_canvas, geometry_cfg.output_width, geometry_cfg.output_height)
        parsing_label = face_parser.run(aligned_srgb, parser_model, parser_device)
        if parsing_label.shape != expected_shape:
            raise SampleFailure("parser_failed", f"parser returned shape {parsing_label.shape}")
        mask_parts = skin_mask.build_face_valid_mask(parsing_label, source_valid, min_component_area_ratio)
        face_valid = mask_parts["face_valid_mask"]
        skin_mask.assert_mask_contract(face_valid, source_valid, expected_shape)
        blackbg = aligned_srgb.copy()
        blackbg[face_valid == 0] = 0
        face_mask = face_region_mask(expected_shape, geom.canvas_landmarks)
        face_pixels = max(1, int((face_mask > 0).sum()))
        hair_id = face_parser.class_ids(("hair",))[0]
        metadata.update(
            {
                "detection_score": float(selected.confidence),
                "detection_bbox": [x, y, bw, bh],
                "expanded_bbox": [crop_x, crop_y, crop_w, crop_h],
                "num_faces_detected": len(detections),
                "selected_face_index": int(selected.index),
                "source_left_eye_center": keys["left_eye"].tolist(),
                "source_right_eye_center": keys["right_eye"].tolist(),
                "source_chin": keys["chin"].tolist(),
                "roll_angle_degrees": geom.roll_angle_degrees,
                "residual_roll_degrees": geom.residual_roll_degrees,
                "crop_left": geom.crop_left,
                "crop_top": geom.crop_top,
                "crop_width": geom.crop_width,
                "crop_height": geom.crop_height,
                "crop_aspect_ratio": geom.crop_width / geom.crop_height,
                "affine_source_to_canvas": geom.affine_source_to_canvas.tolist(),
                "affine_canvas_to_source": geom.affine_canvas_to_source.tolist(),
                "uniform_scale": geom.uniform_scale,
                "matrix_singular_values": list(geom.matrix_singular_values),
                "output_width": geometry_cfg.output_width,
                "output_height": geometry_cfg.output_height,
                "face_valid_fraction": float((face_valid > 0).sum() / face_valid.size),
                "hair_fraction_in_face_region": float(((parsing_label == hair_id) & (face_mask > 0)).sum() / face_pixels),
                "aligned_face_width_fraction": float((geom.canvas_landmarks[:, 0].max() - geom.canvas_landmarks[:, 0].min()) / geometry_cfg.output_width),
                "aligned_face_height_fraction": float((geom.canvas_landmarks[:, 1].max() - geom.canvas_landmarks[:, 1].min()) / geometry_cfg.output_height),
                "scheme": "B_keep_facial_features_remove_hair_background",
                "kept_parser_classes": list(skin_mask.FACE_VALID_KEEP_NAMES),
                "removed_parser_classes": list(skin_mask.FACE_VALID_EXCLUDE_NAMES),
                "min_component_area_px": int(mask_parts["min_component_area_px"]),
            }
        )
        metadata.update(source_valid_stats(source_valid))
        warnings: list[str] = []
        if abs(float(metadata["roll_angle_degrees"])) > 12.0:
            warnings.append("large_roll")
        if float(metadata["source_valid_fraction"]) < 0.98:
            warnings.append("source_boundary_visible")
        if float(metadata["hair_fraction_in_face_region"]) > 0.12:
            warnings.append("high_hair_occlusion")
        metadata["warning_codes"] = warnings
        image_io.save_rgb_png(dirs["images"] / f"{sample_id}.png", blackbg)
        image_io.save_gray_png(dirs["face_valid_mask"] / f"{sample_id}.png", face_valid)
        image_io.save_gray_png(dirs["source_valid_mask"] / f"{sample_id}.png", source_valid)
        image_io.save_gray_png(dirs["parsing_label"] / f"{sample_id}.png", parsing_label.astype(np.uint8))
        image_io.save_rgb_png(dirs["aligned_srgb"] / f"{sample_id}.png", aligned_srgb)
        make_scheme_b_qc(
            dirs["qc_preview_panels"] / f"{sample_id}.png",
            rgb,
            detections,
            selected,
            source_landmarks,
            crop_corners_in_source(geom),
            aligned_srgb,
            geom.canvas_landmarks,
            parsing_label,
            mask_parts["face_keep_mask"],
            mask_parts["face_exclude_mask"],
            face_valid,
            blackbg,
            source_valid,
        )
        metadata["status"] = "success"
        manifest["preprocess_status"] = "success"
        manifest["warning_codes"] = ";".join(warnings)
    except SampleFailure as exc:
        metadata["failure_code"] = exc.code
        metadata["failure_detail"] = exc.detail
        manifest["preprocess_status"] = "failed"
        manifest["failure_code"] = exc.code
        manifest["failure_detail"] = exc.detail
    except Exception as exc:
        metadata["failure_code"] = "unexpected_error"
        metadata["failure_detail"] = f"{type(exc).__name__}: {exc}"
        manifest["preprocess_status"] = "failed"
        manifest["failure_code"] = metadata["failure_code"]
        manifest["failure_detail"] = metadata["failure_detail"]
    (dirs["metadata"] / f"{sample_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest, metadata


def stats(values: list[float], names: tuple[str, ...] = ("min", "median", "p95", "max")) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {name: None for name in names}
    out: dict[str, float | None] = {}
    for name in names:
        if name == "min":
            out[name] = float(np.min(arr))
        elif name == "median":
            out[name] = float(np.median(arr))
        elif name == "p95":
            out[name] = float(np.quantile(arr, 0.95))
        elif name == "max":
            out[name] = float(np.max(arr))
    return out


def summarize(metadata_rows: list[dict[str, Any]], dirs: dict[str, Path]) -> dict[str, Any]:
    success = [m for m in metadata_rows if m.get("status") == "success"]
    failed = [m for m in metadata_rows if m.get("status") != "success"]
    summary = {
        "pilot_total": len(metadata_rows),
        "success_count": len(success),
        "failure_count": len(failed),
        "warning_case_count": int(sum(bool(m.get("warning_codes")) for m in success)),
        "face_valid_fraction": stats([float(m["face_valid_fraction"]) for m in success]),
        "aligned_face_width_fraction": stats([float(m["aligned_face_width_fraction"]) for m in success]),
        "aligned_face_height_fraction": stats([float(m["aligned_face_height_fraction"]) for m in success]),
        "source_valid_fraction": stats([float(m["source_valid_fraction"]) for m in success], ("min", "median", "p95")),
        "roll_angle_abs": stats([abs(float(m["roll_angle_degrees"])) for m in success]),
        "failure_code_counts": dict(sorted(Counter(str(m.get("failure_code", "")) for m in failed).items())),
        "warning_code_counts": dict(sorted(Counter(code for m in success for code in m.get("warning_codes", [])).items())),
    }
    (dirs["root"] / "qc" / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "ID": m.get("sample_id"),
                "status": m.get("status"),
                "failure_code": m.get("failure_code", ""),
                "warning_codes": ";".join(m.get("warning_codes", [])),
                "face_valid_fraction": m.get("face_valid_fraction"),
                "aligned_face_width_fraction": m.get("aligned_face_width_fraction"),
                "aligned_face_height_fraction": m.get("aligned_face_height_fraction"),
                "source_valid_fraction": m.get("source_valid_fraction"),
                "roll_angle_degrees": m.get("roll_angle_degrees"),
            }
            for m in metadata_rows
        ]
    ).to_csv(dirs["root"] / "qc" / "pilot_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def make_contact_sheets(panel_paths: list[Path], output_dir: Path, rows_per_sheet: int = 10, columns: int = 4) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = rows_per_sheet * columns
    outputs: list[str] = []
    for index, start in enumerate(range(0, len(panel_paths), per_sheet), start=1):
        out = output_dir / f"contact_sheet_{index:03d}.png"
        qc.make_contact_sheet(panel_paths[start : start + per_sheet], out, columns=columns)
        if out.is_file():
            outputs.append(str(out))
    return outputs


def write_report(
    dirs: dict[str, Path],
    sample_ids: list[str],
    summary: dict[str, Any],
    config_path: Path,
    full_500: bool,
    contact_sheets: list[str],
) -> None:
    report_name = (
        "realface_256x320_blackbg_from_raw_v1_full500_report.md"
        if full_500
        else "realface_256x320_blackbg_from_raw_v1_pilot_report.md"
    )
    report = dirs["root"] / report_name
    scope = (
        "Generated scheme-B 256x320 black-background face images directly from raw high-resolution phone photos for the full fixed 500-case split."
        if full_500
        else "Generated scheme-B 256x320 black-background face images directly from raw high-resolution phone photos. This is a 32-case pilot only; full 500 was not run."
    )
    text = [
        "# RealFace 256x320 BlackBG From Raw v1 Report",
        "",
        "## Scope",
        scope,
        "",
        "## Scheme B mask",
        f"Kept classes: {', '.join(skin_mask.FACE_VALID_KEEP_NAMES)}.",
        f"Removed classes: {', '.join(skin_mask.FACE_VALID_EXCLUDE_NAMES)}.",
        "",
        "## Pipeline",
        "Raw image -> EXIF transpose -> MediaPipe primary-face detection -> expanded FaceMesh crop -> FaceMesh landmarks mapped to source -> roll + 4:5 crop + uniform scale composed into one affine -> 256x320 aligned_srgb -> BiSeNet parsing -> scheme-B face_valid_mask -> black background PNG.",
        "",
        "## Output directories",
        "- `images/`: final scheme-B black-background PNG files.",
        "- `aligned_srgb/`: unmasked aligned 256x320 RGB PNG files.",
        "- `face_valid_mask/`: binary 0/255 scheme-B masks.",
        "- `parsing_label/`, `source_valid_mask/`, `metadata/`, `qc/`.",
        "",
        "## Config",
        str(config_path),
        "",
        "## Sample IDs",
        ", ".join(sample_ids),
        "",
        "## Summary",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Contact sheets",
        *[f"- {path}" for path in contact_sheets],
    ]
    report.write_text("\n".join(text) + "\n", encoding="utf-8")


def run_pipeline(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    config = load_config(args.config, root)
    resolved = validate_config(config, root, args)
    print(f"[validate] config={resolved['config_path']}")
    print(f"[validate] split_csv={resolved['split_csv']}")
    print(f"[validate] image_dir={resolved['image_dir']}")
    print(f"[validate] checkpoint={resolved['checkpoint']}")
    if args.validate_only:
        print("[validate] OK")
        return 0
    split_df = pd.read_csv(resolved["split_csv"], dtype={"ID": str})
    if "ID" not in split_df.columns or split_df["ID"].duplicated().any():
        raise ValueError("Split CSV must contain unique ID column")
    if resolved["full_500"]:
        selected_ids = split_df["ID"].astype(str).tolist()
        run_df = split_df.copy()
    else:
        selected_ids = select_pilot_ids(split_df, resolved["pilot_count"], args.pilot_id_file)
        run_df = split_df[split_df["ID"].astype(str).isin(set(selected_ids))].copy()
        order = {sample_id: idx for idx, sample_id in enumerate(selected_ids)}
        run_df["_pilot_order"] = run_df["ID"].map(order)
        run_df = run_df.sort_values("_pilot_order").drop(columns=["_pilot_order"])
    dirs = prepare_output_tree(resolved["output_dir"], root, args.overwrite_confirmed)
    id_file_name = "full_ids.txt" if resolved["full_500"] else "pilot_ids.txt"
    (dirs["root"] / id_file_name).write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    shutil.copy2(resolved["config_path"], dirs["config"] / "resolved_config.yaml")
    (dirs["root"] / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(resolved["config_path"]),
                "sample_count": len(selected_ids),
                "full_500_run": bool(resolved["full_500"]),
                "scheme": "B_keep_facial_features_remove_hair_background",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    random.seed(42)
    np.random.seed(42)
    parser_device = face_parser.resolve_device(resolved["parser"].device)
    parser_model = face_parser.load_model(resolved["parser"].model, resolved["checkpoint"], parser_device)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=resolved["detection"].model_selection,
        min_detection_confidence=resolved["detection"].min_detection_confidence,
    )
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=resolved["detection"].min_detection_confidence,
        min_tracking_confidence=0.5,
    )
    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    try:
        for position, split_row in enumerate(run_df.to_dict("records"), start=1):
            sample_id = str(split_row["ID"])
            print(f"[{position:03d}/{len(run_df):03d}] {sample_id}")
            manifest, metadata = process_one(
                split_row,
                dirs,
                resolved["image_dir"],
                detector,
                mesh,
                parser_model,
                parser_device,
                resolved["geometry"],
                resolved["min_component_area_ratio"],
                resolved["image_filename"],
            )
            manifest_rows.append(manifest)
            metadata_rows.append(metadata)
            print(f"  -> {metadata['status']} {metadata.get('failure_code', '')} {';'.join(metadata.get('warning_codes', []))}")
    finally:
        detector.close()
        mesh.close()
    pd.DataFrame(manifest_rows).to_csv(dirs["manifests"] / "realface_256x320_manifest.csv", index=False, encoding="utf-8-sig")
    summary = summarize(metadata_rows, dirs)
    panel_paths = sorted((dirs["root"] / "qc" / "preview_panels").glob("*.png"))
    contact_sheets = make_contact_sheets(panel_paths, dirs["root"] / "qc" / "pilot_contact_sheets")
    write_report(dirs, selected_ids, summary, resolved["config_path"], bool(resolved["full_500"]), contact_sheets)
    (dirs["root"] / "COMPLETED.json").write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "sample_total": summary["pilot_total"],
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
                "full_500_run": bool(resolved["full_500"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        return run_pipeline()
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
