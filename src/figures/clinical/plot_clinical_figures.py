"""Render the final Figure 4 from paired observations and saved statistics.

No correlations, hypothesis tests, or bootstrap intervals are re-estimated here.
The source workbook is opened read-only and is never modified.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import openpyxl


OUT: Path | None = None
SOURCE: Path | None = None
BLUE = "#355C8C"
METRICS = [
    ("BNP", "BNP (pg/mL)", "最近BNP申请时间", "BNP", "pg/mL", 2, True),
    ("LVEF", "LVEF (%)", "最近LVEF申请时间", "LVEF", "%", 1, False),
    ("血红蛋白", "血红蛋白 (g/L)", "最近Hemoglobin申请时间", "Hemoglobin", "g/L", 1, False),
    ("白蛋白", "白蛋白 (g/L)", "最近Albumin申请时间", "Albumin", "g/L", 1, False),
    ("eGFR", "eGFR (原表单位)", "最近eGFR申请时间", "eGFR", "mL/min/1.73 m²", 2, False),
    ("肌酐", "肌酐 (μmol/L)", "最近Creatinine申请时间", "Creatinine", "μmol/L", 1, True),
    ("全血超敏CRP", "全血超敏CRP (mg/L)", "最近Whole-blood hs-CRP申请时间", "Whole-blood hs-CRP", "mg/L", 2, True),
    ("高敏肌钙蛋白T", "高敏肌钙蛋白T (μg/L)", "最近hs-cTnT申请时间", "hs-cTnT", "μg/L", 3, True),
]
META = {m[0]: m for m in METRICS}
MAIN_NAMES = ["BNP", "高敏肌钙蛋白T", "白蛋白", "LVEF"]


def load_data():
    if SOURCE is None:
        raise ValueError("Set --source to the approved clinical workbook.")
    book = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    def records(name):
        rows = list(book[name].values)
        return [dict(zip(rows[0], row)) for row in rows[1:]]
    participants = records("筛选后数据")
    stats_rows = list(book["相关性结果"].values)[6:14]
    stats = {
        r[0]: dict(n=int(r[1]), rho=float(r[2]), low=float(r[3]), high=float(r[4]),
                   p=float(r[5]), q=float(r[6])) for r in stats_rows
    }
    book.close()
    assert len(participants) == len({r["患者ID"] for r in participants}) == 483
    for name, column, datecol, *_ in METRICS:
        rows = [r for r in participants if r[column] is not None]
        assert len(rows) == stats[name]["n"]
        assert stats[name]["low"] < stats[name]["rho"] < stats[name]["high"]
        for r in rows:
            assert np.isfinite([r[column], r["模型分数"]]).all()
            assert abs(r["模型分数"] - (r["logit_patient"] - r["logit_control"])) < 1e-10
            gap = (r["照片时间"] - r[datecol]).total_seconds() / 86400
            assert 0 <= gap <= (30 if name == "LVEF" else 7)
    assert sum(s["q"] < .05 for s in stats.values()) == 5
    return participants, stats


def style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": True,
        "savefig.facecolor": "white",
    })


def save(figure, basename, *, final_size=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if final_size:
        figure.savefig(OUT / f"{basename}.pdf", dpi=600, facecolor="white")
    else:
        for ext in ("pdf", "png"):
            figure.savefig(OUT / f"{basename}.{ext}", dpi=600,
                           bbox_inches="tight", pad_inches=.08)
    plt.close(figure)
    print(f"FIGURE={basename}")


def q_math(value):
    if value >= .001:
        return f"{value:.3f}"
    base, exponent = f"{value:.2e}".split("e")
    return base + r"\times10^{" + str(int(exponent)) + "}"


def relationships(participants, stats, names, basename):
    style()
    final_size = basename == "Figure4_individual_clinical_associations"
    figure, axes = plt.subplots(
        2, 2,
        figsize=(160 / 25.4, 130 / 25.4) if final_size else (7.2, 5.5),
        layout="constrained",
    )
    y = np.array([r["模型分数"] for r in participants])
    ylim = (float(y.min()) - .35, float(y.max()) + .35)
    for letter, axis, name in zip("abcd" if final_size else "ABCD", axes.flat, names):
        _, col, _, label, unit, _, logx = META[name]
        rows = [r for r in participants if r[col] is not None]
        x = np.array([r[col] for r in rows])
        score = np.array([r["模型分数"] for r in rows])
        s = stats[name]
        axis.scatter(x, score, s=8, alpha=.32, color=BLUE, linewidths=0, zorder=3)
        if logx:
            assert (x > 0).all()
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(*ylim)
        axis.set_xlabel(f"{label} ({unit})")
        axis.set_ylabel("Facial score")
        if final_size:
            axis.tick_params(axis="both", labelsize=7.5)
        axis.grid(axis="y", color="#E8EDF2", linewidth=.55, zorder=0)
        axis.set_axisbelow(True)
        axis.set_title(f"{letter}  {label}", loc="left", fontsize=10, fontweight="bold", pad=33)
        text = (f"n = {s['n']};  " + r"$\rho$" + f" = {s['rho']:.3f};  " +
                "$q = " + q_math(s["q"]) + "$\n" +
                f"95% CI: {s['low']:.3f} to {s['high']:.3f}")
        axis.text(0, 1.018, text, transform=axis.transAxes, va="bottom", ha="left",
                  fontsize=7.5 if final_size else 7.2,
                  linespacing=1.4, clip_on=False)
    figure.set_constrained_layout_pads(w_pad=.05, h_pad=.07, hspace=.10, wspace=.08)
    save(figure, basename, final_size=final_size)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=("figure4",),
                        help="Render the final Figure 4.")
    parser.add_argument("--source", type=Path, required=True,
                        help="Approved paired clinical workbook; it is opened read-only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    global SOURCE, OUT
    SOURCE = args.source.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    participants, stats = load_data()
    relationships(participants, stats, MAIN_NAMES, "Figure4_individual_clinical_associations")


if __name__ == "__main__":
    main()
