"""Build a patient-grouped, sex-stratified 5-fold split for binary NYHA tasks.

The binary label is defined as:
  - 0: normal (raw NYHA == 0)
  - 1: abnormal (raw NYHA in {1, 2, 3, 4})
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


CSV_ENCODING = "utf-8-sig"
N_SPLITS = 5
RANDOM_SEED = 2026
NYHA_VALUES = (0, 1, 2, 3, 4)
RAW_NYHA_BALANCE_WEIGHT = 8.0
OUTPUT_COLUMNS = [
    "ID",
    "patient_group_id",
    "SEX",
    "sex_name",
    "NYHA",
    "label_2class",
    "label_2class_name",
    "binary_label",
    "stratum",
    "fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def normalize_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    identifier = str(value).strip()
    if not identifier:
        return None
    if re.fullmatch(r"\d+\.0", identifier):
        identifier = identifier[:-2]
    return identifier


def load_and_validate_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {path}")

    frame = pd.read_csv(path, dtype={"ID": "string"}, encoding=CSV_ENCODING)
    required_columns = {"ID", "SEX", "NYHA"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    clean = frame.loc[:, ["ID", "SEX", "NYHA"]].copy()
    clean["ID"] = clean["ID"].map(normalize_identifier)
    clean["SEX"] = pd.to_numeric(clean["SEX"], errors="coerce")
    clean["NYHA"] = pd.to_numeric(clean["NYHA"], errors="coerce")

    issues: list[str] = []
    if clean["ID"].isna().any():
        issues.append("missing ID")
    if (~clean["SEX"].isin((0, 1))).any():
        issues.append("SEX must be 0 or 1")
    if (~clean["NYHA"].isin(NYHA_VALUES)).any():
        issues.append("NYHA must be an integer from 0 to 4")
    duplicate_ids = clean["ID"].duplicated(keep=False) & clean["ID"].notna()
    if duplicate_ids.any():
        duplicate_text = ", ".join(sorted(clean.loc[duplicate_ids, "ID"].unique()))
        issues.append(f"duplicate ID values: {duplicate_text}")
    if issues:
        raise ValueError("Invalid labels: " + "; ".join(issues))

    clean["SEX"] = clean["SEX"].astype("int64")
    clean["NYHA"] = clean["NYHA"].astype("int64")
    return clean.reset_index(drop=True)


def add_split_columns(frame: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    result = frame.copy()
    result["patient_group_id"] = result["ID"].str.replace(r"-\d+$", "", regex=True)
    inconsistent_sex = result.groupby("patient_group_id")["SEX"].nunique()
    inconsistent_sex = inconsistent_sex[inconsistent_sex > 1]
    if not inconsistent_sex.empty:
        groups = ", ".join(inconsistent_sex.index.astype(str).tolist())
        raise ValueError(f"Inconsistent SEX within patient groups: {groups}")

    result["label_2class"] = (result["NYHA"] != 0).astype("int64")
    result["binary_label"] = result["label_2class"]
    result["label_2class_name"] = result["label_2class"].map(
        {0: "normal", 1: "abnormal"}
    )
    result["sex_name"] = result["SEX"].map({0: "female", 1: "male"})
    result["stratum"] = (
        result["label_2class"].astype(str) + "_" + result["SEX"].astype(str)
    )

    if result["patient_group_id"].nunique() < n_splits:
        raise ValueError("Fewer patient groups than requested folds")
    small_strata = result["stratum"].value_counts()
    small_strata = small_strata[small_strata < n_splits]
    if not small_strata.empty:
        warnings.warn(
            "Some label_2class x SEX strata have fewer samples than folds: "
            + ", ".join(f"{name}={count}" for name, count in small_strata.items()),
            stacklevel=2,
        )
    return result


def assign_folds(
    frame: pd.DataFrame, n_splits: int, random_seed: int
) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed
    )
    result = frame.copy()
    result["fold"] = -1
    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            X=result.index.to_numpy(),
            y=result["stratum"],
            groups=result["patient_group_id"],
        )
    ):
        result.loc[validation_indices, "fold"] = fold
    if (result["fold"] < 0).any():
        raise RuntimeError("Some rows were not assigned to a fold")
    return refine_stratum_balance(result, n_splits, random_seed)


def refine_stratum_balance(
    frame: pd.DataFrame, n_splits: int, random_seed: int
) -> pd.DataFrame:
    """Improve label x sex balance while moving whole patient groups only."""
    result = frame.copy()
    strata = ("0_0", "0_1", "1_0", "1_1")
    grouped: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for patient_group_id, group in result.groupby("patient_group_id", sort=True):
        vector = np.array(
            [(group["stratum"] == stratum).sum() for stratum in strata],
            dtype=float,
        )
        nyha_vector = np.array(
            [(group["NYHA"] == nyha).sum() for nyha in NYHA_VALUES],
            dtype=float,
        )
        grouped.append((str(patient_group_id), vector, nyha_vector, len(group)))

    assignments = np.array(
        [
            int(
                result.loc[
                    result["patient_group_id"] == patient_group_id, "fold"
                ].iloc[0]
            )
            for patient_group_id, _, _, _ in grouped
        ],
        dtype=int,
    )
    targets = np.array(
        [float((result["stratum"] == stratum).sum()) / n_splits for stratum in strata]
    )
    nyha_targets = np.array(
        [float((result["NYHA"] == nyha).sum()) / n_splits for nyha in NYHA_VALUES]
    )
    size_target = len(result) / n_splits
    fold_strata = np.zeros((n_splits, len(strata)), dtype=float)
    fold_nyha = np.zeros((n_splits, len(NYHA_VALUES)), dtype=float)
    fold_sizes = np.zeros(n_splits, dtype=float)
    for assignment, (_, vector, nyha_vector, size) in zip(assignments, grouped):
        fold_strata[assignment] += vector
        fold_nyha[assignment] += nyha_vector
        fold_sizes[assignment] += size

    def score() -> float:
        stratum_score = np.sum(
            (fold_strata - targets) ** 2 / np.maximum(targets, 1.0)
        )
        raw_nyha_score = RAW_NYHA_BALANCE_WEIGHT * np.sum(
            (fold_nyha - nyha_targets) ** 2 / np.maximum(nyha_targets, 1.0)
        )
        size_score = 10.0 * np.sum(
            (fold_sizes - size_target) ** 2 / max(size_target, 1.0)
        )
        return float(stratum_score + raw_nyha_score + size_score)

    def move(group_index: int, source: int, target: int) -> None:
        _, vector, nyha_vector, size = grouped[group_index]
        fold_strata[source] -= vector
        fold_strata[target] += vector
        fold_nyha[source] -= nyha_vector
        fold_nyha[target] += nyha_vector
        fold_sizes[source] -= size
        fold_sizes[target] += size
        assignments[group_index] = target

    rng = np.random.default_rng(random_seed)
    for _ in range(50):
        improved = False
        for group_index in rng.permutation(len(grouped)):
            source = int(assignments[group_index])
            baseline = score()
            best_target = source
            best_score = baseline
            for target in range(n_splits):
                if target == source:
                    continue
                move(group_index, source, target)
                candidate_score = score()
                move(group_index, target, source)
                if candidate_score < best_score - 1e-12:
                    best_target = target
                    best_score = candidate_score
            if best_target != source:
                move(group_index, source, best_target)
                improved = True

        # Equal-size group swaps preserve fold sizes while improving strata.
        order = rng.permutation(len(grouped))
        for left_position, left_index in enumerate(order):
            left_fold = int(assignments[left_index])
            _, left_vector, left_nyha, left_size = grouped[left_index]
            for right_index in order[left_position + 1 :]:
                right_fold = int(assignments[right_index])
                _, right_vector, right_nyha, right_size = grouped[right_index]
                if left_fold == right_fold or left_size != right_size:
                    continue
                baseline = score()
                fold_strata[left_fold] += right_vector - left_vector
                fold_strata[right_fold] += left_vector - right_vector
                fold_nyha[left_fold] += right_nyha - left_nyha
                fold_nyha[right_fold] += left_nyha - right_nyha
                candidate_score = score()
                if candidate_score < baseline - 1e-12:
                    assignments[left_index] = right_fold
                    assignments[right_index] = left_fold
                    improved = True
                    left_fold = right_fold
                    left_vector = right_vector
                    left_nyha = right_nyha
                    left_size = right_size
                    break
                fold_strata[left_fold] += left_vector - right_vector
                fold_strata[right_fold] += right_vector - left_vector
                fold_nyha[left_fold] += left_nyha - right_nyha
                fold_nyha[right_fold] += right_nyha - left_nyha
        if not improved:
            break

    group_to_fold = {
        patient_group_id: int(fold)
        for (patient_group_id, _, _, _), fold in zip(grouped, assignments)
    }
    result["fold"] = result["patient_group_id"].map(group_to_fold).astype("int64")
    return result


def validate_split(frame: pd.DataFrame, n_splits: int) -> None:
    if len(frame) != frame["ID"].nunique():
        raise RuntimeError("Output contains duplicate ID values")
    expected_folds = set(range(n_splits))
    actual_folds = set(frame["fold"].unique())
    if actual_folds != expected_folds:
        raise RuntimeError(f"Unexpected fold assignments: {sorted(actual_folds)}")
    leaked_groups = frame.groupby("patient_group_id")["fold"].nunique()
    leaked_groups = leaked_groups[leaked_groups > 1]
    if not leaked_groups.empty:
        groups = ", ".join(leaked_groups.index.astype(str).tolist())
        raise RuntimeError(f"Patient-group leakage across folds: {groups}")
    expected_label = (frame["NYHA"] != 0).astype("int64")
    if not expected_label.equals(frame["label_2class"]):
        raise RuntimeError("Binary label mapping does not match raw NYHA")


def build_summary(frame: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for fold in range(n_splits):
        subset = frame.loc[frame["fold"] == fold]
        row: dict[str, int] = {"fold": fold, "total_samples": len(subset)}
        for label, name in ((0, "normal"), (1, "abnormal")):
            row[f"class_{name}"] = int((subset["label_2class"] == label).sum())
        for sex in (0, 1):
            row[f"SEX_{sex}"] = int((subset["SEX"] == sex).sum())
        for nyha in NYHA_VALUES:
            row[f"NYHA_{nyha}"] = int((subset["NYHA"] == nyha).sum())
        for stratum in ("0_0", "0_1", "1_0", "1_1"):
            row[f"stratum_{stratum}"] = int((subset["stratum"] == stratum).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, n_splits: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.loc[:, OUTPUT_COLUMNS].sort_values(
        ["fold", "patient_group_id", "ID"], kind="stable"
    )
    stem = "nyha_2class_sex_stratified_group_5fold"
    ordered.to_csv(output_dir / f"{stem}.csv", index=False, encoding=CSV_ENCODING)
    for fold in range(n_splits):
        ordered.loc[ordered["fold"] != fold].to_csv(
            output_dir / f"{stem}_fold_{fold}_train.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
        ordered.loc[ordered["fold"] == fold].to_csv(
            output_dir / f"{stem}_fold_{fold}_val.csv",
            index=False,
            encoding=CSV_ENCODING,
        )
    summary.to_csv(
        output_dir / f"{stem}_summary.csv", index=False, encoding=CSV_ENCODING
    )

    report_lines = [
        "# Binary NYHA 5-fold split quality report",
        "",
        "- Binary label: 0 = NYHA 0 (normal), 1 = NYHA 1-4 (abnormal)",
        "- Stratification: label_2class x SEX",
        "- Grouping: patient_group_id",
        "- Patient-group leakage: none",
        "",
        "## Fold summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / f"{stem}_quality_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.n_splits != N_SPLITS:
        raise ValueError("This script is fixed to 5-fold splitting")

    labels = load_and_validate_labels(args.labels_csv)
    prepared = add_split_columns(labels, args.n_splits)
    split = assign_folds(prepared, args.n_splits, args.random_seed)
    validate_split(split, args.n_splits)
    summary = build_summary(split, args.n_splits)
    write_outputs(split, summary, args.output_dir, args.n_splits)

    print(f"Generated {len(split)} rows across {args.n_splits} folds.")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
