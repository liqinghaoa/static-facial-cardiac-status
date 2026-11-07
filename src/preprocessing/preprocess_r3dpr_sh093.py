"""Run the study's SH093 PortraitRelighting batch preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "preprocess_r3dpr_sh093.yaml"

SH_INDEX = 93
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

np = None
torch = None
Image = None
ImageOps = None
crop_module = None
render_tensor = None
initialize_models = None
warm_up_models = None

FIXED_CAMERA_ARGS = {
    "cam_pivot": [0.0, 0.0, 0.0],
    "radius": 2.7,
    "focal_length": 18.837,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
}

CROPPOSE_DETECTOR_VARIANTS = [
    {
        "name": "official_resize0.5_thr0.95",
        "resize": 0.5,
        "thresholds": [0.95, 0.95, 0.95],
    },
    {
        "name": "fallback_resize0.5_thr0.90",
        "resize": 0.5,
        "thresholds": [0.90, 0.90, 0.90],
    },
    {
        "name": "fallback_resize0.5_thr0.80",
        "resize": 0.5,
        "thresholds": [0.80, 0.80, 0.80],
    },
]

STATUS_FIELDS = [
    "case_id",
    "input_path",
    "success",
    "status",
    "error_message",
    "elapsed_seconds",
    "input_ext",
    "original_w",
    "original_h",
    "output_w",
    "output_h",
    "croppose_detector",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--relighting-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--example-lightings", type=Path, default=None)
    parser.add_argument("--sh-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check paths, input count and SH index without importing or initializing models.",
    )
    return parser


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else PROJECT_ROOT
    )
    config_path = DEFAULT_CONFIG if args.config is None else resolve_path(args.config, project_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")

    paths = config.get("paths", {}) or {}
    sh_config = config.get("sh", {}) or {}
    run_config = config.get("run", {}) or {}
    if not all(isinstance(value, dict) for value in (paths, sh_config, run_config)):
        raise ValueError("Config sections 'paths', 'sh' and 'run' must be mappings")

    def configured_path(
        cli_value: Path | None, key: str, fallback: str | Path | None = None
    ) -> Path:
        value = cli_value if cli_value is not None else paths.get(key, fallback)
        if value is None:
            raise ValueError(f"Missing required path: paths.{key}")
        return resolve_path(value, project_root)

    def configured_value(
        cli_value: Any, section: dict[str, Any], key: str, fallback: Any
    ) -> Any:
        return cli_value if cli_value is not None else section.get(key, fallback)

    relighting_root = configured_path(args.relighting_root, "relighting_root")
    input_dir = configured_path(args.input_dir, "input_dir")
    output_root = configured_path(args.output_root, "output_root")
    lightings_path = configured_path(
        args.example_lightings,
        "example_lightings",
        relighting_root / "examples" / "example_lightings.npy",
    )
    sh_index = configured_value(args.sh_index, sh_config, "index", SH_INDEX)
    limit = configured_value(args.limit, run_config, "limit", None)
    case_ids = args.case_id if args.case_id is not None else run_config.get("case_ids")
    resume = configured_value(args.resume, run_config, "resume", False)
    overwrite = configured_value(args.overwrite, run_config, "overwrite", False)

    if not isinstance(sh_index, int) or isinstance(sh_index, bool):
        raise ValueError("sh.index must be an integer")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("run.limit must be a positive integer or null")
    if case_ids is not None and (
        not isinstance(case_ids, list)
        or not all(isinstance(value, str) for value in case_ids)
    ):
        raise ValueError("run.case_ids must be a list of strings or null")
    if not isinstance(resume, bool) or not isinstance(overwrite, bool):
        raise ValueError("run.resume and run.overwrite must be booleans")

    return {
        "project_root": project_root,
        "config_path": config_path.resolve(),
        "relighting_root": relighting_root,
        "input_dir": input_dir,
        "output_root": output_root,
        "example_lightings": lightings_path,
        "sh_index": sh_index,
        "limit": limit,
        "case_ids": case_ids,
        "resume": resume,
        "overwrite": overwrite,
    }


def load_runtime(relighting_root: Path) -> None:
    global Image, ImageOps, crop_module, initialize_models, np, render_tensor, torch
    global warm_up_models

    # Use this local cache for externally acquired models unless overridden.
    # Model files are not distributed; see README.md.
    bundled_torch_home = PACKAGE_ROOT / "assets" / "torch"
    if bundled_torch_home.is_dir():
        os.environ.setdefault("TORCH_HOME", str(bundled_torch_home))

    import numpy as np_module
    import torch as torch_module
    from PIL import Image as Image_module
    from PIL import ImageOps as ImageOps_module

    root_string = str(relighting_root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    os.chdir(relighting_root)

    from example import initialize_models as initialize_models_fn
    from example import warm_up_models as warm_up_models_fn
    import third_party.CropPose.crop as crop_module_value
    from utils import render_tensor as render_tensor_fn

    np = np_module
    torch = torch_module
    Image = Image_module
    ImageOps = ImageOps_module
    crop_module = crop_module_value
    render_tensor = render_tensor_fn
    initialize_models = initialize_models_fn
    warm_up_models = warm_up_models_fn


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "original": output_root / "images" / "original",
        "fixedSH_origcam": output_root / "images" / "fixedSH_origcam",
        "fixedSH_fixedcam": output_root / "images" / "fixedSH_fixedcam",
        "case_metadata": output_root / "metadata" / "case_metadata",
        "metadata": output_root / "metadata",
        "logs": output_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def scan_inputs(input_dir: Path, case_ids: list[str] | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if case_ids:
        wanted = set(case_ids)
        paths = [p for p in paths if p.stem in wanted]
        found = {p.stem for p in paths}
        missing = sorted(wanted - found)
        if missing:
            raise RuntimeError(f"Requested case_id not found: {missing}")
    return paths


def load_selected_sh(index: int, lightings_path: Path) -> tuple[Any, tuple[int, ...]]:
    import numpy as np_module

    raw = np_module.load(lightings_path)
    raw_shape = tuple(raw.shape)
    bank = np_module.asarray(raw, dtype=np_module.float32).squeeze()
    if bank.shape != (100, 9):
        bank = bank.reshape(100, 9)
    if bank.shape != (100, 9):
        raise RuntimeError(f"Expected SH bank shape (100, 9), got {bank.shape}")
    if not 0 <= index < bank.shape[0]:
        raise ValueError(f"SH index out of range: {index}")
    return bank[index].reshape(1, 9).astype(np_module.float32), raw_shape


def validate_relighting_root(relighting_root: Path) -> list[Path]:
    required = [
        relighting_root / "example.py",
        relighting_root / "utils.py",
        relighting_root / "networks",
        relighting_root / "third_party",
        relighting_root / "examples" / "example_lightings.npy",
        relighting_root / "checkpoints" / "model.pth",
        relighting_root / "checkpoints" / "NeRFFaceLighting.pkl",
        relighting_root / "third_party" / "CropPose" / "checkpoints" / "pretrained" / "epoch_20.pth",
    ]
    return [path for path in required if not path.exists()]


def set_croppose_detector(variant: dict[str, Any], device: torch.device) -> None:
    crop_module.fast_mtcnn = crop_module.FastMTCNN(
        stride=4,
        resize=variant["resize"],
        margin=14,
        factor=0.5,
        keep_all=False,
        device=device,
        thresholds=variant["thresholds"],
    )


def run_croppose_with_fallbacks(
    cropposer,
    image: Image.Image,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    errors = []
    for variant in CROPPOSE_DETECTOR_VARIANTS:
        set_croppose_detector(variant, device)
        try:
            result = cropposer.wild2all(image)
            return result, {
                "name": variant["name"],
                "resize": variant["resize"],
                "thresholds": variant["thresholds"],
                "fallback_used": variant["name"] != CROPPOSE_DETECTOR_VARIANTS[0]["name"],
                "previous_errors": errors,
            }
        except ValueError as exc:
            errors.append(
                {
                    "variant": variant["name"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    raise ValueError(
        "Cannot find valid face in the image after CropPose detector fallbacks: "
        + json.dumps(errors, ensure_ascii=False)
    )


def outputs_exist(output_root: Path, case_id: str) -> bool:
    required = [
        output_root / "images" / "original" / f"{case_id}.png",
        output_root / "images" / "fixedSH_origcam" / f"{case_id}.png",
        output_root / "images" / "fixedSH_fixedcam" / f"{case_id}.png",
        output_root / "metadata" / "case_metadata" / f"{case_id}.json",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def case_outputs_present(output_root: Path, case_id: str) -> bool:
    possible = [
        output_root / "images" / "original" / f"{case_id}.png",
        output_root / "images" / "fixedSH_origcam" / f"{case_id}.png",
        output_root / "images" / "fixedSH_fixedcam" / f"{case_id}.png",
        output_root / "metadata" / "case_metadata" / f"{case_id}.json",
    ]
    return any(path.exists() for path in possible)


def status_row(
    case_id: str,
    input_path: Path,
    success: bool,
    status: str,
    error_message: str = "",
    elapsed_seconds: float = 0.0,
    original_size: tuple[int, int] | None = None,
    output_size: tuple[int, int] | None = None,
    croppose_detector: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "input_path": str(input_path),
        "success": int(success),
        "status": status,
        "error_message": error_message,
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "input_ext": input_path.suffix.lower().lstrip("."),
        "original_w": original_size[0] if original_size else "",
        "original_h": original_size[1] if original_size else "",
        "output_w": output_size[0] if output_size else "",
        "output_h": output_size[1] if output_size else "",
        "croppose_detector": croppose_detector,
    }


def render_with_planes(
    relighting,
    cropped_tensor: torch.Tensor,
    camera_tensor: torch.Tensor,
    sh_tensor: torch.Tensor,
    planes: torch.Tensor,
    superres_ws: torch.Tensor,
) -> torch.Tensor:
    if cropped_tensor.device.type == "cuda":
        with torch.cuda.amp.autocast():
            ret, _, _ = relighting.image_forward(
                cropped_tensor,
                camera_tensor,
                sh_tensor,
                gt_planes=planes,
                gt_superres_ws=superres_ws,
            )
    else:
        ret, _, _ = relighting.image_forward(
            cropped_tensor,
            camera_tensor,
            sh_tensor,
            gt_planes=planes,
            gt_superres_ws=superres_ws,
        )
    return ret["image"]


def process_case(
    input_path: Path,
    output_root: Path,
    device: torch.device,
    cropposer,
    relighting,
    dpr,
    selected_sh_tensor: torch.Tensor,
    sh_index: int,
    canonical_camera: torch.Tensor,
    overwrite: bool,
    resume: bool,
) -> dict[str, Any]:
    case_id = input_path.stem
    if resume and not overwrite and outputs_exist(output_root, case_id):
        return status_row(
            case_id,
            input_path,
            success=True,
            status="SKIPPED_ALREADY_EXISTS",
        )

    start = time.time()
    stage = "FAILED_INPUT_READ"
    original_size = None
    output_size = None
    detector_info: dict[str, Any] = {}

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        original_image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
        original_size = original_image.size

        stage = "FAILED_CROPPOSE"
        with torch.inference_mode():
            crop_result, detector_info = run_croppose_with_fallbacks(
                cropposer,
                original_image,
                device,
            )
            cropped_tensor = crop_result["img_cropped"].to(device)
            original_camera = crop_result["cam"].to(device).float()

        stage = "FAILED_DPR"
        with torch.inference_mode():
            original_sh_tensor = (
                dpr.dpr.extract_lighting(cropped_tensor)
                .squeeze()
                .reshape(1, 9)
                .to(device=device, dtype=torch.float32)
            )

        stage = "FAILED_RELIGHT"
        with torch.inference_mode():
            recon_result, planes, _ = relighting.image_forward(
                cropped_tensor,
                original_camera,
                original_sh_tensor,
            )
            fixed_sh_origcam = render_with_planes(
                relighting,
                cropped_tensor,
                original_camera,
                selected_sh_tensor,
                planes,
                recon_result["superres_ws"],
            )
            fixed_sh_fixedcam = render_with_planes(
                relighting,
                cropped_tensor,
                canonical_camera,
                selected_sh_tensor,
                planes,
                recon_result["superres_ws"],
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        original_out = original_image
        origcam_image = render_tensor(fixed_sh_origcam)
        fixedcam_image = render_tensor(fixed_sh_fixedcam)
        output_size = fixedcam_image.size

        stage = "FAILED_SAVE"
        original_out.save(output_root / "images" / "original" / f"{case_id}.png")
        origcam_image.save(output_root / "images" / "fixedSH_origcam" / f"{case_id}.png")
        fixedcam_image.save(output_root / "images" / "fixedSH_fixedcam" / f"{case_id}.png")

        elapsed = time.time() - start
        metadata = {
            "case_id": case_id,
            "input_path": str(input_path),
            "sh_index": sh_index,
            "selected_sh_values": (
                selected_sh_tensor.detach().cpu().numpy().reshape(9).astype(float).tolist()
            ),
            "outputs": {
                "original": str(output_root / "images" / "original" / f"{case_id}.png"),
                "fixedSH_origcam": str(
                    output_root / "images" / "fixedSH_origcam" / f"{case_id}.png"
                ),
                "fixedSH_fixedcam": str(
                    output_root / "images" / "fixedSH_fixedcam" / f"{case_id}.png"
                ),
            },
            "original_size": list(original_size),
            "output_image_size": list(output_size),
            "original_camera_shape": list(original_camera.shape),
            "canonical_camera_shape": list(canonical_camera.shape),
            "original_sh_shape": list(original_sh_tensor.shape),
            "selected_sh_shape": list(selected_sh_tensor.shape),
            "planes_shape": list(planes.shape),
            "used_same_planes_for_both_renders": True,
            "crop_pose_runs": 1,
            "dpr_lighting_estimation_runs": 1,
            "full_face_encoding_runs": 1,
            "croppose_detector": detector_info,
            "elapsed_seconds": elapsed,
            "device": str(device),
        }
        if device.type == "cuda":
            metadata["peak_gpu_memory_mb"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

        with open(
            output_root / "metadata" / "case_metadata" / f"{case_id}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        return status_row(
            case_id,
            input_path,
            success=True,
            status="SUCCESS",
            elapsed_seconds=elapsed,
            original_size=original_size,
            output_size=output_size,
            croppose_detector=detector_info.get("name", ""),
        )

    except Exception as exc:
        elapsed = time.time() - start
        error_message = f"{type(exc).__name__}: {exc}"
        failure_detail = {
            "case_id": case_id,
            "input_path": str(input_path),
            "status": stage,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        with open(
            output_root / "metadata" / "case_metadata" / f"{case_id}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(failure_detail, file, indent=2, ensure_ascii=False)
        return status_row(
            case_id,
            input_path,
            success=False,
            status=stage,
            error_message=error_message,
            elapsed_seconds=elapsed,
            original_size=original_size,
            output_size=output_size,
            croppose_detector=detector_info.get("name", ""),
        )
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(settings: dict[str, Any]) -> tuple[list[Path], int, Any, tuple[int, ...]]:
    missing = validate_relighting_root(settings["relighting_root"])
    if missing:
        raise FileNotFoundError(
            "PortraitRelighting checkout is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
            + ". Third-party model resources are not bundled. See "
            "README.md in the package root for acquisition and placement."
        )
    paths = scan_inputs(settings["input_dir"], settings["case_ids"])
    if not paths:
        raise RuntimeError(f"No supported input images found in {settings['input_dir']}")
    total_available_inputs = len(paths)
    if settings["limit"] is not None:
        paths = paths[: settings["limit"]]
    selected_sh, library_shape = load_selected_sh(
        settings["sh_index"], settings["example_lightings"]
    )
    return paths, total_available_inputs, selected_sh, library_shape


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args)
    paths, total_available_inputs, selected_sh_np, sh_library_shape = prepare_inputs(settings)

    print("relighting_root:", settings["relighting_root"])
    print("input_dir:", settings["input_dir"])
    print("output_root:", settings["output_root"])
    print("total_available_inputs:", total_available_inputs)
    print("selected_inputs:", len(paths))
    print("sh_index:", settings["sh_index"])
    print("sh_library_shape:", sh_library_shape)
    print("output_sets: original, fixedSH_origcam, fixedSH_fixedcam")
    if args.validate_only:
        print("VALIDATE_ONLY_PASS: no output directories created; no models imported or initialized")
        return

    output_root = settings["output_root"]
    if not settings["resume"] and not settings["overwrite"]:
        collisions = [path.stem for path in paths if case_outputs_present(output_root, path.stem)]
        if collisions:
            raise FileExistsError(
                "Output files already exist for selected cases. Use --resume to skip "
                "completed cases or --overwrite to regenerate them."
            )

    load_runtime(settings["relighting_root"])
    dirs = ensure_dirs(output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("resume:", settings["resume"])
    print("overwrite:", settings["overwrite"])
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    run_start = time.time()
    cropposer, relighting, dpr = initialize_models(device)
    cropposer.eval()
    relighting.eval()
    dpr.eval()
    warm_up_models(cropposer, relighting, dpr, device)

    selected_sh_tensor = torch.from_numpy(selected_sh_np).to(
        device=device,
        dtype=torch.float32,
    )
    canonical_camera = relighting.encoders.eg3d.args2cam(**FIXED_CAMERA_ARGS).to(
        device=device,
        dtype=torch.float32,
    )

    rows = []
    for idx, input_path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] start case_id={input_path.stem}")
        row = process_case(
            input_path=input_path,
            output_root=output_root,
            device=device,
            cropposer=cropposer,
            relighting=relighting,
            dpr=dpr,
            selected_sh_tensor=selected_sh_tensor,
            sh_index=settings["sh_index"],
            canonical_camera=canonical_camera,
            overwrite=settings["overwrite"],
            resume=settings["resume"],
        )
        rows.append(row)
        print(
            f"[{idx}/{len(paths)}] {row['status']} case_id={input_path.stem} "
            f"elapsed={row['elapsed_seconds']} error={row['error_message']}"
        )

    success_count = sum(1 for row in rows if row["success"] == 1)
    skipped_count = sum(1 for row in rows if row["status"] == "SKIPPED_ALREADY_EXISTS")
    failure_rows = [row for row in rows if row["success"] != 1]
    elapsed = time.time() - run_start

    status_path = dirs["metadata"] / "case_status.csv"
    failure_path = dirs["metadata"] / "failure_cases.csv"
    summary_path = dirs["metadata"] / "summary.json"
    manifest_path = dirs["metadata"] / "run_manifest.json"
    write_csv(status_path, rows)
    write_csv(failure_path, failure_rows)

    summary = {
        "relighting_root": str(settings["relighting_root"]),
        "input_dir": str(settings["input_dir"]),
        "output_root": str(output_root),
        "total_available_inputs": total_available_inputs,
        "selected_inputs": len(paths),
        "processed_or_available_success_count": success_count,
        "skipped_already_exists_count": skipped_count,
        "failure_count": len(failure_rows),
        "sh_source": str(settings["example_lightings"]),
        "sh_library_shape": list(sh_library_shape),
        "selected_sh_index": settings["sh_index"],
        "selected_sh_values": selected_sh_np.reshape(9).astype(float).tolist(),
        "fixed_camera_args": FIXED_CAMERA_ARGS,
        "output_image_sets": [
            "images/original/<case_id>.png",
            "images/fixedSH_origcam/<case_id>.png",
            "images/fixedSH_fixedcam/<case_id>.png",
        ],
        "device": str(device),
        "elapsed_seconds": elapsed,
        "no_classification_training": True,
        "no_model_weight_modification": True,
        "task_scope": "generate original, fixedSH+origcam, fixedSH+fixedcam images only",
    }
    if device.type == "cuda":
        summary["peak_gpu_memory_mb_at_end"] = (
            torch.cuda.max_memory_allocated() / 1024**2
        )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "config": str(settings["config_path"]),
        "args": {
            "relighting_root": str(settings["relighting_root"]),
            "input_dir": str(settings["input_dir"]),
            "output_root": str(output_root),
            "example_lightings": str(settings["example_lightings"]),
            "sh_index": settings["sh_index"],
            "limit": settings["limit"],
            "case_id": settings["case_ids"],
            "resume": settings["resume"],
            "overwrite": settings["overwrite"],
        },
        "input_cases": [
            {
                "case_id": path.stem,
                "input_path": str(path),
                "input_ext": path.suffix.lower().lstrip("."),
            }
            for path in paths
        ],
        "summary": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("case_status_csv:", status_path)
    print("failure_cases_csv:", failure_path)
    print("summary_json:", summary_path)
    print("run_manifest_json:", manifest_path)
    print("processed_or_available_success_count:", success_count)
    print("failure_count:", len(failure_rows))
    print("elapsed_seconds:", round(elapsed, 3))
    if len(failure_rows) == 0:
        print("BATCH_EXAMPLE_SH093_FIXEDLIGHT_PASS")
    else:
        raise RuntimeError(f"{len(failure_rows)} cases failed")


if __name__ == "__main__":
    main()
