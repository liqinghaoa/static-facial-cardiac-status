"""Generate OOF Grad-CAM PNGs for an R3DPR binary ResNet experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.binary_face_dataset import R3DPRBinaryFaceDataset, build_r3dpr_transforms
from models.resnet_binary__model import build_r3dpr_resnet_binary
from utils.oof_validation import expected_table, validate_oof
from utils.experiment_utils import load_yaml


CLASS_NAMES = {0: "Control", 1: "Patient"}
DEFAULT_OUTPUT_SUBDIR = "figures/gradcam_oof"
CHECKPOINT_FILENAMES = ("inner_best_macro_auc.pth", "best_macro_auc.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument(
        "--target-class",
        choices=("predicted", "true", "patient"),
        default="predicted",
        help="Logit used to calculate Grad-CAM for each sample.",
    )
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Optional OOF sample ID filter; may be supplied more than once.",
    )
    parser.add_argument(
        "--max-samples-per-fold",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit to process every OOF sample.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help="Merge this invocation's records into an existing manifest, replacing any matching sample IDs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replay-atol",
        type=float,
        default=1e-3,
        help="Maximum absolute OOF probability replay difference; predictions must always match.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_path(experiment_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else experiment_dir / path


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 150,
        }
    )


def final_class_name(index: int) -> str:
    try:
        return CLASS_NAMES[int(index)]
    except KeyError as error:
        raise ValueError(f"Expected binary class 0 or 1, got {index}") from error


def make_dataset_row(row: pd.Series, config: dict) -> dict:
    data = config["data"]
    table = pd.DataFrame(
        [
            {
                data["id_column"]: str(row["sample_id"]),
                data["group_id_column"]: str(row["patient_group_id"]),
                data["fold_column"]: int(row["fold"]),
                data["label_column"]: int(row["binary_label"]),
                data["sex_column"]: str(row.get("sex", "")),
            }
        ]
    )
    transform = build_r3dpr_transforms(
        "val",
        int(data["image_height"]),
        int(data["image_width"]),
        config["normalize"]["mean"],
        config["normalize"]["std"],
        False,
    )
    dataset = R3DPRBinaryFaceDataset(
        table=table,
        image_root=project_path(data["image_root"]),
        image_filename_template=str(data["image_filename_template"]),
        transform=transform,
        id_column=data["id_column"],
        group_id_column=data["group_id_column"],
        fold_column=data["fold_column"],
        label_column=data["label_column"],
        sex_column=data["sex_column"],
    )
    return dataset[0]


class GradCAM:
    """Grad-CAM for the output of the final ResNet residual stage."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        target_layer = model.layer4[-1]
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:  # noqa: ANN001
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:  # noqa: ANN001
        self.gradient = grad_output[0]

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def calculate(self, image: torch.Tensor, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        self.activation = None
        self.gradient = None
        logits = self.model(image)
        logits[0, int(target_class)].backward()
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture the final ResNet stage")
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=tuple(image.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        maximum = float(cam.max())
        if maximum > 0.0:
            cam /= maximum
        else:
            cam.fill(0.0)
        return logits.detach().cpu().numpy()[0], cam


def denormalize(image: torch.Tensor, config: dict) -> np.ndarray:
    mean = torch.tensor(config["normalize"]["mean"], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(config["normalize"]["std"], dtype=image.dtype).view(3, 1, 1)
    restored = image.detach().cpu() * std + mean
    return restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def save_heatmap(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    output_png: Path,
    row: pd.Series,
    target_class: int,
) -> None:
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * image_rgb + 0.45 * heatmap, 0.0, 1.0)
    true_class = final_class_name(int(row["binary_label"]))
    predicted_class = final_class_name(int(row["pred_class"]))
    target_name = final_class_name(target_class)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), constrained_layout=True)
    figure.suptitle(
        f"ID {row['sample_id']} | fold {int(row['fold'])} | true {true_class} | "
        f"predicted {predicted_class} | Patient probability {float(row['prob_patient']):.3f}",
        fontsize=8,
    )
    axes[0].imshow(image_rgb)
    axes[0].set_title("(a) Input image", loc="left", fontweight="bold")
    axes[1].imshow(overlay)
    axes[1].set_title(f"(b) Grad-CAM for {target_name} logit", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_axis_off()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(figure)


def target_from_row(row: pd.Series, mode: str) -> int:
    if mode == "predicted":
        return int(row["pred_class"])
    if mode == "true":
        return int(row["binary_label"])
    return 1


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    model = build_r3dpr_resnet_binary(
        str(model_cfg["backbone"]),
        pretrained="none",
        dropout=model_cfg.get("dropout"),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validate_requested_folds(folds: list[int] | None, n_folds: int) -> list[int]:
    requested = list(range(n_folds)) if folds is None else sorted(set(folds))
    if not requested or set(requested).difference(range(n_folds)):
        raise ValueError(f"fold must be in 0..{n_folds - 1}, got {requested}")
    return requested


def resolve_checkpoint_path(experiment_dir: Path, fold: int) -> Path:
    checkpoint_dir = experiment_dir / f"fold_{fold}" / "checkpoints"
    for filename in CHECKPOINT_FILENAMES:
        checkpoint_path = checkpoint_dir / filename
        if checkpoint_path.is_file():
            return checkpoint_path
    expected = ", ".join(str(checkpoint_dir / filename) for filename in CHECKPOINT_FILENAMES)
    raise FileNotFoundError(f"Missing selection checkpoint for fold {fold}; expected one of: {expected}")


def write_report(output_dir: Path, records: pd.DataFrame, args: argparse.Namespace) -> None:
    counts = records.groupby("fold").size().to_dict()
    report = [
        "# R3DPR OOF Grad-CAM",
        "",
        "- Each sample used the validation-selected checkpoint from its own outer fold.",
        f"- CAM target: `{args.target_class}`.",
        f"- Generated PNGs: {len(records)}.",
        f"- Per-fold counts: {counts}.",
        f"- Maximum probability replay difference: {records['max_abs_probability_difference'].max():.6g}.",
        "- The manifest records the input prediction, target logit, checkpoint, and PNG path.",
    ]
    (output_dir / "gradcam_generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def generate_gradcam(args: argparse.Namespace) -> Path:
    experiment_dir = project_path(args.experiment_dir)
    config_path = experiment_dir / "config_snapshot.yaml"
    oof_path = experiment_dir / "oof_predictions.csv"
    if not config_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError("experiment_dir must contain config_snapshot.yaml and oof_predictions.csv")
    config = load_yaml(config_path)
    oof = pd.read_csv(oof_path, dtype={"sample_id": "string", "patient_group_id": "string"})
    n_folds = int(config["data"]["n_folds"])
    validate_oof(oof, expected_table(config), n_folds)
    folds = validate_requested_folds(args.folds, n_folds)
    if args.max_samples_per_fold is not None and int(args.max_samples_per_fold) < 1:
        raise ValueError("max-samples-per-fold must be positive")
    if float(args.replay_atol) <= 0.0:
        raise ValueError("replay-atol must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only when necessary")
    device = torch.device(args.device)
    output_dir = output_path(experiment_dir, args.output_subdir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    records: list[dict[str, object]] = []
    requested_sample_ids = None if args.sample_ids is None else {str(sample_id) for sample_id in args.sample_ids}

    for fold in folds:
        fold_rows = oof.loc[oof["fold"].astype(int) == fold].sort_values("sample_id")
        if requested_sample_ids is not None:
            fold_rows = fold_rows.loc[fold_rows["sample_id"].astype(str).isin(requested_sample_ids)]
        if args.max_samples_per_fold is not None:
            fold_rows = fold_rows.head(int(args.max_samples_per_fold))
        checkpoint_path = resolve_checkpoint_path(experiment_dir, fold)
        model = load_model(config, checkpoint_path, device)
        gradcam = GradCAM(model)
        try:
            for row in fold_rows.itertuples(index=False):
                series = pd.Series(row._asdict())
                sample = make_dataset_row(series, config)
                image = sample["image"].unsqueeze(0).to(device)
                target_class = target_from_row(series, args.target_class)
                logits, cam = gradcam.calculate(image, target_class)
                probability = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
                predicted_class = int(probability.argmax())
                stored_probability = series[["prob_control", "prob_patient"]].to_numpy(dtype=float)
                probability_difference = float(np.max(np.abs(probability - stored_probability)))
                # CUDA kernels and library versions can differ from saved inference by around 1e-4.
                if predicted_class != int(series["pred_class"]) or probability_difference > float(args.replay_atol):
                    raise RuntimeError(
                        f"OOF replay mismatch for ID {series['sample_id']} in fold {fold}: "
                        f"replayed={probability.tolist()}, stored={stored_probability.tolist()}, "
                        f"max_abs_difference={probability_difference:.6g}"
                    )
                output_png = output_dir / f"fold_{fold}" / f"{series['sample_id']}_gradcam.png"
                save_heatmap(denormalize(sample["image"], config), cam, output_png, series, target_class)
                records.append(
                    {
                        "sample_id": str(series["sample_id"]),
                        "fold": fold,
                        "true_class": int(series["binary_label"]),
                        "predicted_class": predicted_class,
                        "prob_control": float(probability[0]),
                        "prob_patient": float(probability[1]),
                        "stored_prob_control": float(stored_probability[0]),
                        "stored_prob_patient": float(stored_probability[1]),
                        "max_abs_probability_difference": probability_difference,
                        "target_class": target_class,
                        "target_name": final_class_name(target_class),
                        "checkpoint_path": str(checkpoint_path),
                        "input_image_path": str(sample["image_path"]),
                        "output_png": str(output_png),
                    }
                )
        finally:
            gradcam.close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    manifest = pd.DataFrame(records)
    if requested_sample_ids is not None:
        generated_sample_ids = set(manifest["sample_id"].astype(str))
        missing_sample_ids = sorted(requested_sample_ids.difference(generated_sample_ids))
        if missing_sample_ids:
            raise ValueError(f"Requested sample IDs are absent from the selected OOF folds: {missing_sample_ids}")
    manifest_path = output_dir / "gradcam_manifest.csv"
    if args.append_manifest and manifest_path.is_file():
        existing = pd.read_csv(manifest_path, dtype={"sample_id": "string"})
        existing = existing.loc[~existing["sample_id"].astype(str).isin(manifest["sample_id"].astype(str))]
        manifest = pd.concat([existing, manifest], ignore_index=True)
    manifest = manifest.sort_values(["fold", "sample_id"], kind="stable").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    write_report(output_dir, manifest, args)
    return output_dir


def main() -> Path:
    args = parse_args()
    output_dir = generate_gradcam(args)
    print(f"GRADCAM_OUTPUT_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
