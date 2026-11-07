"""Draw the final Figure 2 four-panel ROC comparison PDF."""

from pathlib import Path
import csv
import re
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


REGIONS = ["全脸", "眼周", "面颊", "口唇"]
DEPTHS = [18, 34, 50]
STEMS = {
    "全脸": "256x320_relight_neutral_front",
    "眼周": "224x224_manual_shift_eye_roi",
    "面颊": "224x224_manual_shift_cheek_roi",
    "口唇": "224x224_manual_shift_lip_roi",
}


def load_results(results_tex: Path, source: Path):
    text = results_tex.read_text(encoding="utf-8")
    entries = re.findall(
        r"^\s*(全脸|眼周|面颊|口唇) & ResNet-(18|34|50) & ([0-9.]+)",
        text,
        re.MULTILINE,
    )
    assert len(entries) == 12
    expected = {(region, int(depth)): value for region, depth, value in entries}
    curves = {}
    reference = None
    for region in REGIONS:
        for depth in DEPTHS:
            folder = (
                f"R3DPR_ResNet{depth}_ControlVsPatient_Binary_{STEMS[region]}"
                "_fullbaseline_ls005_nested_direct_5fold"
            )
            with (source / folder / "oof_predictions.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            records = {
                row["sample_id"]: (int(row["binary_label"]), int(row["fold"]))
                for row in rows
            }
            assert len(rows) == len(records) == 500
            assert sum(label for label, _ in records.values()) == 385
            if reference is None:
                reference = records
            assert records == reference
            labels = np.array([int(row["binary_label"]) for row in rows])
            abnormal = np.array([float(row["prob_patient"]) for row in rows])
            normal = np.array([float(row["prob_control"]) for row in rows])
            assert np.isfinite(abnormal).all()
            assert np.allclose(abnormal + normal, 1, atol=1e-6)
            auc = roc_auc_score(labels, abnormal)
            macro_auc = (auc + roc_auc_score(1 - labels, normal)) / 2
            assert abs(auc - macro_auc) < 1e-12
            assert f"{macro_auc:.4f}" == expected[(region, depth)]
            fpr, tpr, _ = roc_curve(labels, abnormal, pos_label=1)
            curves[(region, depth)] = (fpr, tpr, macro_auc)
    return curves


def draw_roc(ax, curves, region, scale=1.0):
    # Okabe-Ito-inspired colors with line styles that remain distinguishable
    # under red-green color-vision deficiency and grayscale reproduction.
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    linestyles = ["-", "--", "-."]
    for depth, color, linestyle in zip(DEPTHS, colors, linestyles):
        fpr, tpr, auc = curves[(region, depth)]
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, lw=1.05 * scale,
                label=f"ResNet-{depth} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#8A8A8A", lw=0.7 * scale, ls=":", zorder=0)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
           xlabel="1 - Specificity", ylabel="Sensitivity")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=6.0 * scale, length=2.3 * scale, pad=2 * scale)
    ax.xaxis.label.set_size(6.7 * scale)
    ax.yaxis.label.set_size(6.7 * scale)
    ax.xaxis.labelpad = 2.3 * scale
    ax.yaxis.labelpad = 2.3 * scale
    ax.grid(color="#E5E5E5", lw=0.55 * scale)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75 * scale)
    ax.legend(loc="lower right", frameon=False, fontsize=5.9 * scale,
              handlelength=1.5, handletextpad=0.4, borderaxespad=0.3,
              labelspacing=0.28, borderpad=0.15)
    titles = {"全脸": "Whole-face", "眼周": "Periocular", "面颊": "Cheek", "口唇": "Lip"}
    ax.set_title(titles[region], fontsize=8.2 * scale, pad=3.5 * scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-root", type=Path, required=True,
                        help="Directory containing the 12 nested-direct result folders.")
    parser.add_argument("--results-tex", type=Path, required=True,
                        help="LaTeX table containing the reported 12 AUC values.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "Figure2_classification.pdf"
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.75,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    curves = load_results(args.results_tex, args.oof_root)
    # The final Figure 2 contains only the four region-specific ROC panels.
    roc_figure, axes = plt.subplots(2, 2, figsize=(5.6, 4.8))
    roc_figure.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.93,
                               hspace=0.34, wspace=0.31)
    for ax, region in zip(axes.flat, REGIONS):
        draw_roc(ax, curves, region, scale=1.25)
    roc_figure.savefig(target, bbox_inches="tight", pad_inches=0.04)
    plt.close(roc_figure)
    print(target)


if __name__ == "__main__":
    main()
