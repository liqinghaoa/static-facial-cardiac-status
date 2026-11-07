"""Plot two paired OOF ROC comparisons from the same 330-person cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def load_predictions(path: Path, score_column: str, model_name: str,
                    fold_column: str = "fold") -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str, "patient_group_id": str})
    required = {"ID", "patient_group_id", "binary_label", fold_column, score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame.ID.duplicated().any() or len(frame) != 330:
        raise ValueError(f"{path} must contain 330 unique IDs")
    return frame[["ID", "patient_group_id", "binary_label", fold_column, score_column]].rename(
        columns={"binary_label": f"{model_name}_label", fold_column: f"{model_name}_fold",
                 score_column: model_name}
    )


def load_ci(path: Path, key_column: str, key: str) -> tuple[float, float, float]:
    intervals = pd.read_csv(path)
    row = intervals.loc[intervals[key_column].astype(str) == key]
    if len(row) != 1:
        raise ValueError(f"Expected one CI row for {key} in {path}")
    row = row.iloc[0]
    return float(row.estimate), float(row.ci_lower), float(row.ci_upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic-dir", type=Path, required=True)
    parser.add_argument("--basic-fusion-dir", type=Path, required=True)
    parser.add_argument("--clinical-dir", type=Path, required=True)
    parser.add_argument("--clinical-fusion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    basic = load_predictions(args.basic_dir / "oof_predictions.csv", "L_clinical_OOF", "basic")
    basic_fusion = load_predictions(args.basic_fusion_dir / "paired_oof_predictions_330.csv", "L_fusion_OOF", "basic_fusion", "clinical_fold")
    clinical = load_predictions(args.clinical_dir / "oof_predictions_330.csv", "L_clinical5_OOF", "clinical")
    clinical_fusion = load_predictions(args.clinical_fusion_dir / "oof_predictions_330.csv", "L_clinical5_face_OOF", "clinical_fusion")

    frames = [basic, basic_fusion, clinical, clinical_fusion]
    reference = basic[["ID", "patient_group_id", "basic_label"]].rename(columns={"basic_label": "binary_label"})
    scores = reference.copy()
    for frame, name in zip(frames, ("basic", "basic_fusion", "clinical", "clinical_fusion")):
        check = frame[["ID", "patient_group_id", f"{name}_label"]].rename(columns={f"{name}_label": "binary_label"})
        check = reference.merge(check, on=["ID", "patient_group_id", "binary_label"], how="outer", indicator=True)
        if len(check) != 330 or not check._merge.eq("both").all():
            raise ValueError(f"ID/group/label set differs for {name}")
        scores = scores.merge(frame[["ID", name]], on="ID", how="left", validate="one_to_one", sort=False)
    if scores.isna().any().any() or scores.ID.nunique() != 330:
        raise ValueError("Prediction merge is incomplete")

    metrics = {
        "basic": load_ci(args.basic_dir / "oof_metrics_cluster_bootstrap_ci.csv", "metric", "roc_auc"),
        "basic_fusion": load_ci(args.basic_fusion_dir / "auc_paired_cluster_bootstrap_ci.csv", "comparison", "fusion"),
        "clinical": load_ci(args.clinical_fusion_dir / "auc_paired_cluster_bootstrap_ci.csv", "comparison", "clinical5"),
        "clinical_fusion": load_ci(args.clinical_fusion_dir / "auc_paired_cluster_bootstrap_ci.csv", "comparison", "clinical5_face"),
    }
    y = scores.binary_label.to_numpy(int)
    for name, (reported, _, _) in metrics.items():
        computed = float(roc_auc_score(y, scores[name].to_numpy(float)))
        if not np.isclose(computed, reported, atol=1e-10, rtol=0):
            raise ValueError(f"AUC mismatch for {name}: computed={computed}, reported={reported}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "basic": "Basic",
        "basic_fusion": "Basic + face",
        "clinical": "Clinical",
        "clinical_fusion": "Clinical + face",
    }
    colors = {"basic": "#0072B2", "basic_fusion": "#D55E00",
              "clinical": "#0072B2", "clinical_fusion": "#D55E00"}
    groups = [("basic", "basic_fusion"), ("clinical", "clinical_fusion")]
    titles = ["a  Basic information models", "b  Clinical indicator models"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        # Set the physical page width to the manuscript's 160 mm, so pt sizes
        # remain unchanged when the PDF is included at textwidth.
        fig, axes = plt.subplots(1, 2, figsize=(160 / 25.4, 90 / 25.4),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.91, wspace=.19)
        for ax, pair, title in zip(axes, groups, titles):
            for name in pair:
                fpr, tpr, _ = roc_curve(y, scores[name].to_numpy(float))
                auc, low, high = metrics[name]
                ax.plot(fpr, tpr, color=colors[name], linewidth=1.3,
                        label=f"{labels[name]}: AUC {auc:.3f}\n95% CI {low:.3f}–{high:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=.8, label="Chance")
            ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("1 - Specificity", fontsize=8)
            ax.legend(loc="lower right", fontsize=7.5, frameon=False,
                      handlelength=1.5, handletextpad=.5, labelspacing=.5,
                      borderaxespad=.5)
        axes[0].set_ylabel("Sensitivity", fontsize=8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
    print(f"ROC_PDF={args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
