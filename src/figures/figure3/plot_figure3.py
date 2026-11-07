"""Plot the eight reported correlations without recalculating statistics."""

from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


SOURCE: Path | None = None
OUT: Path | None = None
NAMES = {
    "BNP": "BNP",
    "LVEF": "LVEF",
    "Hemoglobin": "Hemoglobin",
    "Albumin": "Albumin",
    "eGFR": "eGFR",
    "Creatinine": "Creatinine",
    "Whole-blood hs-CRP": "Whole-blood hs-CRP",
    "hs-cTnT": "hs-cTnT",
}
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−", "0123456789-+-")


def number(text):
    value = text.translate(SUPERSCRIPTS)
    if "×" in value:
        base, exponent = re.split(r"\s*×\s*10", value)
        return float(base) * 10 ** int(exponent)
    return float(value)


def q_label(text):
    value = text.translate(SUPERSCRIPTS)
    if "×" in value:
        base, exponent = re.split(r"\s*×\s*10", value)
        return base + r"$\times10^{" + exponent + "}$"
    return text


def load_data():
    if SOURCE is None:
        raise ValueError("Set --source to the approved correlation-results document.")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(SOURCE) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    rows = []
    for table in document.findall(".//w:tbl", ns):
        for row in table.findall("w:tr", ns):
            cells = [
                "".join(t.text or "" for t in cell.findall(".//w:t", ns))
                for cell in row.findall("w:tc", ns)
            ]
            if len(cells) == 6 and cells[0] in NAMES:
                low, high = cells[3].split(" to ")
                rows.append({
                    "source_name": cells[0], "name": NAMES[cells[0]],
                    "n": int(cells[1]), "rho": number(cells[2]),
                    "low": number(low), "high": number(high),
                    "p": cells[4], "q": cells[5],
                })
    assert len(rows) == len({r["name"] for r in rows}) == 8
    assert [r["source_name"] for r in rows] == list(NAMES)
    assert all(0 < r["n"] <= 483 and r["low"] < r["rho"] < r["high"] for r in rows)
    assert sum(number(r["q"]) < 0.05 for r in rows) == 5
    return rows


def render(rows):
    if OUT is None:
        raise ValueError("Output directory is not configured")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9,
        "axes.unicode_minus": False,
        "axes.labelsize": 9, "xtick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2, 3.5), facecolor="white")
    ax = fig.add_axes([0.245, 0.19, 0.46, 0.70])
    positions = np.arange(8)
    estimates = np.array([r["rho"] for r in rows])
    low = np.array([r["low"] for r in rows])
    high = np.array([r["high"] for r in rows])
    for i in positions:
        if i % 2 == 0:
            ax.axhspan(i - 0.47, i + 0.47, color="#F3F6F9", zorder=0)
    ax.axvline(0, color="#777777", lw=1, ls=(0, (3, 3)), zorder=1)
    ax.errorbar(estimates, positions, xerr=[estimates - low, high - estimates],
                fmt="o", markersize=5, color="#355C8C", ecolor="#355C8C",
                elinewidth=1.4, capsize=3, capthick=1.1, zorder=3)
    ax.set_xlim(-0.40, 0.50)
    ax.set_ylim(7.65, -0.85)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_yticks([])
    ax.set_xlabel("Spearman ρ", labelpad=7)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="x", length=3, color="#777777")

    # Align annotation columns to each estimate without sharing the effect-size axis.
    y_transform = ax.get_yaxis_transform()
    for y, row in enumerate(rows):
        ax.text(-0.44, y, row["name"], transform=y_transform,
                ha="left", va="center", clip_on=False)
        ax.text(1.16, y, str(row["n"]), transform=y_transform,
                ha="center", va="center", clip_on=False)
        ax.text(1.52, y, q_label(row["q"]), transform=y_transform,
                ha="center", va="center", clip_on=False)
    header_y = -0.72
    for x, text, align in [(-0.44, "Clinical marker", "left"),
                           (0.50, "Spearman ρ (95% CI)", "center"),
                           (1.16, "n", "center"),
                           (1.52, "FDR-adjusted q", "center")]:
        ax.text(x, header_y, text, transform=y_transform,
                ha=align, va="center", clip_on=False)

    for ext in ["pdf", "png"]:
        path = OUT / f"Figure3_clinical_associations.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.07)
        print(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Approved document containing the aggregate correlation table.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    global SOURCE, OUT
    SOURCE = args.source.resolve()
    OUT = args.output_dir.resolve()
    render(load_data())


if __name__ == "__main__":
    main()
