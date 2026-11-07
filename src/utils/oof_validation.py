"""Shared OOF coverage and label checks for the current experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = ["macro_auc", "macro_f1", "balanced_accuracy", "sensitivity", "specificity"]

METRIC_LABELS = {
    "macro_auc": "Macro-AUC",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}

def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def expected_table(config: dict) -> pd.DataFrame:
    data = config["data"]
    table = pd.read_csv(project_path(data["table_csv"]), dtype={data["id_column"]: "string", data["group_id_column"]: "string"})
    table = table.rename(columns={data["id_column"]: "sample_id", data["group_id_column"]: "patient_group_id", data["fold_column"]: "fold", data["label_column"]: "binary_label"})
    table["sample_id"] = table["sample_id"].astype(str)
    table["patient_group_id"] = table["patient_group_id"].astype(str)
    table["fold"] = table["fold"].astype(int)
    table["binary_label"] = table["binary_label"].astype(int)
    return table.loc[:, ["sample_id", "patient_group_id", "fold", "binary_label"]]

def validate_oof(oof: pd.DataFrame, expected: pd.DataFrame, n_folds: int) -> None:
    required = {"sample_id", "patient_group_id", "fold", "binary_label", "logit_control", "logit_patient", "prob_control", "prob_patient", "pred_class", "image_path", "selected_epoch", "checkpoint_path"}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF lacks required columns: {missing}")
    oof = oof.copy()
    oof["sample_id"] = oof["sample_id"].astype(str)
    oof["patient_group_id"] = oof["patient_group_id"].astype(str)
    if oof["sample_id"].duplicated().any():
        raise ValueError("OOF contains duplicate sample IDs")
    if len(oof) != len(expected) or set(oof["sample_id"]) != set(expected["sample_id"]):
        raise ValueError(f"OOF ID coverage failed: observed={len(oof)}, expected={len(expected)}")
    if set(oof["fold"].astype(int).unique()) != set(range(n_folds)):
        raise ValueError("OOF fold coverage is incomplete")
    expected_indexed = expected.set_index("sample_id")
    joined = oof.set_index("sample_id").join(expected_indexed, rsuffix="_expected", how="left")
    for column in ("patient_group_id", "fold", "binary_label"):
        observed = joined[column].astype(str).to_numpy() if column == "patient_group_id" else joined[column].astype(int).to_numpy()
        expected_values = joined[f"{column}_expected"].astype(str).to_numpy() if column == "patient_group_id" else joined[f"{column}_expected"].astype(int).to_numpy()
        if not np.array_equal(observed, expected_values):
            raise ValueError(f"OOF {column} values do not match the source R3DPR CSV")
    probability = oof[["prob_control", "prob_patient"]].to_numpy(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("OOF probabilities are invalid")
    if not np.array_equal(oof["pred_class"].to_numpy(int), probability.argmax(axis=1)):
        raise ValueError("OOF predictions are not softmax argmax")
    expected_counts = expected.groupby("fold").size().to_dict()
    observed_counts = oof.groupby("fold").size().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"OOF fold sample counts differ from source table: {observed_counts} != {expected_counts}")
