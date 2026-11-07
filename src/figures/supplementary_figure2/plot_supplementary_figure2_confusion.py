"""Generate Supplementary Figure 2 pooled OOF confusion matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})


DISPLAYED_BACKBONE_OVERRIDES = {"whole_face_sh93": "resnet18"}
BACKBONE_CONFIG = {
    "resnet18": ("ResNet-18", "resnet18"),
    "resnet34": ("ResNet-34", "resnet34"),
    "resnet50": ("ResNet-50", "resnet50"),
}
VIEW_CONFIG = {
    "whole_face_sh93": ("A", "Whole-face"),
    "periocular": ("B", "Periocular"),
    "cheek": ("C", "Cheek"),
    "lip": ("D", "Lip"),
}


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_integrity(inventory: pd.DataFrame, integrity: pd.DataFrame) -> None:
    require_columns(inventory, {"model_id"}, "model_inventory.csv")
    if inventory["model_id"].nunique() != 12 or integrity.empty:
        raise ValueError("Expected a non-empty integrity table and 12 model IDs")


def validate_prediction_groups(predictions: pd.DataFrame) -> None:
    require_columns(predictions, {"model_id", "binary_label", "pred_class"}, "pooled_oof_predictions_long.csv")
    counts = predictions.groupby("model_id").size()
    if counts.empty or not counts.eq(500).all():
        raise ValueError("Each pooled model prediction group must contain 500 records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the approved pooled OOF summary CSV files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pdf_path = args.output.resolve()
    if pdf_path.exists():
        raise FileExistsError(f"Output already exists: {pdf_path}")
    data_dir = args.data_dir.resolve()
    predictions = pd.read_csv(data_dir / "pooled_oof_predictions_long.csv", dtype={"sample_id": "string"})
    pooled_metrics = pd.read_csv(data_dir / "pooled_oof_metrics.csv")
    pooled_confusions = pd.read_csv(data_dir / "pooled_confusion_matrices.csv")
    inventory = pd.read_csv(data_dir / "model_inventory.csv")
    integrity = pd.read_csv(data_dir / "integrity_checks.csv")
    validate_integrity(inventory, integrity)
    validate_prediction_groups(predictions)
    require_columns(pooled_metrics, {"model_id", "input_view", "backbone", "macro_auc"}, "pooled_oof_metrics.csv")
    require_columns(pooled_confusions, {"model_id", "tn", "fp", "fn", "tp", "n_samples"}, "pooled_confusion_matrices.csv")

    selected_rows = []
    matrices = {}
    for view, (_, view_label) in VIEW_CONFIG.items():
        candidates = pooled_metrics.loc[pooled_metrics["input_view"] == view].sort_values(["macro_auc", "backbone"], ascending=[False, True])
        if len(candidates) != 3:
            raise ValueError(f"Expected three backbones for {view}, found {len(candidates)}")
        override = DISPLAYED_BACKBONE_OVERRIDES.get(view)
        selected = candidates.loc[candidates["backbone"] == override].iloc[0] if override else candidates.iloc[0]
        model_id = str(selected["model_id"])
        group = predictions.loc[predictions["model_id"] == model_id]
        matrix = confusion_matrix(group["binary_label"].astype(int), group["pred_class"].astype(int), labels=[0, 1])
        source = pooled_confusions.loc[pooled_confusions["model_id"] == model_id]
        if len(source) != 1:
            raise ValueError(f"Expected one pooled confusion-matrix row for {model_id}, found {len(source)}")
        source_matrix = np.array([[int(source.iloc[0].tn), int(source.iloc[0].fp)], [int(source.iloc[0].fn), int(source.iloc[0].tp)]])
        if not np.array_equal(matrix, source_matrix) or int(matrix.sum()) != 500:
            raise ValueError(f"Pooled confusion-matrix validation failed for {model_id}")
        matrices[view] = matrix
        selected_rows.append({"Input": view_label, "Model_ID": model_id, "Backbone": BACKBONE_CONFIG[str(selected.backbone)][0], "Pooled_OOF_Macro_AUC": float(selected.macro_auc), "TN": int(matrix[0, 0]), "FP": int(matrix[0, 1]), "FN": int(matrix[1, 0]), "TP": int(matrix[1, 1]), "N": int(matrix.sum())})
    checks = pd.DataFrame(selected_rows)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), facecolor="white", layout="constrained")
    image = None
    for axis, (view, (letter, view_label)) in zip(axes.flat, VIEW_CONFIG.items()):
        matrix = matrices[view]
        proportions = matrix / matrix.sum(axis=1, keepdims=True)
        row = checks.loc[checks["Input"] == view_label].iloc[0]
        image = axis.imshow(proportions, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_xticks([0, 1], ["Normal", "Abnormal"])
        axis.set_yticks([0, 1], ["Normal", "Abnormal"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.tick_params(length=0)
        axis.set_title(f"{view_label}: {row.Backbone}\nAUC={row.Pooled_OOF_Macro_AUC:.3f}", fontsize=9.3, pad=8)
        axis.text(-0.17, 1.13, letter.lower(), transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)
        for i in range(2):
            for j in range(2):
                value = proportions[i, j]
                color = "white" if value >= 0.58 else "#1A1A1A"
                axis.text(j, i - 0.09, f"{matrix[i, j]:d}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
                axis.text(j, i + 0.14, f"({value * 100:.1f}%)", ha="center", va="center", fontsize=8, color=color)
        axis.set_box_aspect(1)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    colorbar.set_label("Row-normalized proportion")
    colorbar.ax.tick_params(labelsize=8)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"PDF={pdf_path}")


if __name__ == "__main__":
    main()
