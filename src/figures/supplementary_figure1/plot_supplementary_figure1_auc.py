"""Generate Figure 3 outer-fold AUC stability curves for nested-direct experiments."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42})


BACKBONE_CONFIG = {
    "resnet18": ("ResNet-18", "resnet18"),
    "resnet34": ("ResNet-34", "resnet34"),
    "resnet50": ("ResNet-50", "resnet50"),
}
VIEW_CONFIG = {
    "wholeface": ("A", "Whole-face"),
    "periocular": ("B", "Periocular"),
    "cheek": ("C", "Cheek"),
    "lip": ("D", "Lip"),
}
STYLES = {
    "resnet18": ("#0072B2", "-", "o"),
    "resnet34": ("#D55E00", "--", "s"),
    "resnet50": ("#CC79A7", "-.", "^"),
}


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_integrity(inventory: pd.DataFrame, integrity: pd.DataFrame) -> None:
    if inventory.empty or integrity.empty:
        raise ValueError("Integrity tables must not be empty")
    require_columns(inventory, {"model_id"}, "model_inventory.csv")
    if inventory["model_id"].nunique() != 12:
        raise ValueError("Expected 12 unique model IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the three approved summary CSV files.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pdf_path = args.output.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        raise FileExistsError(f"Output already exists: {pdf_path}")
    foldwise = pd.read_csv(args.data_dir / "foldwise_metrics_long.csv")
    inventory = pd.read_csv(args.data_dir / "model_inventory.csv")
    integrity = pd.read_csv(args.data_dir / "integrity_checks.csv")
    validate_integrity(inventory, integrity)
    require_columns(foldwise, {"model_id", "input_view", "backbone", "fold", "selected_epoch", "macro_auc"}, "foldwise_metrics_long.csv")
    if len(foldwise) != 60:
        raise ValueError(f"Expected 60 fold rows for 12 models, found {len(foldwise)}")
    records = []
    for view, (_, view_label) in VIEW_CONFIG.items():
        for backbone, (backbone_label, _) in BACKBONE_CONFIG.items():
            group = foldwise.loc[(foldwise["input_view"] == view) & (foldwise["backbone"] == backbone)].sort_values("fold")
            if len(group) != 5 or set(group["fold"].astype(int)) != set(range(5)):
                raise ValueError(f"{view}/{backbone} does not have exactly one row for outer folds 0-4")
            for row in group.itertuples(index=False):
                records.append({"Input": view_label, "Backbone": backbone_label, "Fold": int(row.fold) + 1, "Outer_Test_Macro_AUC": float(row.macro_auc), "Selected_Epoch": int(row.selected_epoch)})
    checks = pd.DataFrame(records)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), facecolor="white", layout="constrained")
    for axis, (view, (letter, view_label)) in zip(axes.flat, VIEW_CONFIG.items()):
        for backbone, (backbone_label, _) in BACKBONE_CONFIG.items():
            color, linestyle, marker = STYLES[backbone]
            group = checks.loc[(checks["Input"] == view_label) & (checks["Backbone"] == backbone_label)].sort_values("Fold")
            axis.plot(group["Fold"], group["Outer_Test_Macro_AUC"], color=color, linestyle=linestyle, linewidth=1.5, marker=marker, markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, label=backbone_label, zorder=2)
        axis.set_xlim(0.8, 5.2)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(range(1, 6))
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.set_xlabel("Outer fold")
        axis.set_ylabel("AUC")
        axis.set_title(view_label, fontsize=10, pad=8)
        axis.text(-0.15, 1.12, letter.lower(), transform=axis.transAxes, fontsize=12, fontweight="bold", va="top", clip_on=False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
        axis.legend(loc="lower right", fontsize=7.3, handlelength=1.7)
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"PDF={pdf_path}")


if __name__ == "__main__":
    main()
