"""Current fixed-logit fusion alignment, checks and paired evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

import yaml

from scipy.special import expit

from sklearn.metrics import roc_auc_score

from utils.clinical_logistic import basic_metrics, calibration_metrics, resolve_path, save_csv, save_json

SCORES = {"clinical": "L_clinical_OOF", "face": "L_face_OOF", "fusion": "L_fusion_OOF"}

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["fusion"] != {
        "clinical_weight": 0.5, "face_weight": 0.5, "intercept": 0.0,
        "face_score": "logit_patient_minus_logit_control", "fit_parameters": False,
    }:
        raise ValueError("This experiment fixes fusion to 0.5*clinical + 0.5*face with zero intercept")
    e = cfg["evaluation"]
    if e["classification_threshold"] != 0.5 or e["bootstrap_unit"] != "patient_group_id" or e["bootstrap_refit"]:
        raise ValueError("Expected threshold 0.5 and fixed-prediction patient-group bootstrap")
    if e["bootstrap_replicates"] < 1 or not 0 < e["confidence_level"] < 1:
        raise ValueError("Invalid bootstrap settings")
    return cfg

def validate_identifiers(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in [id_column, "patient_group_id"]:
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"Missing identifier in {column}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].isin(["", "nan", "None"]).any():
            raise ValueError(f"Empty identifier in {column}")
    if frame[id_column].duplicated().any():
        raise ValueError(f"Duplicated {id_column}")
    for column, choices in [("binary_label", [0, 1]), ("fold", list(range(5)))]:
        if not frame[column].isin(choices).all():
            raise ValueError(f"Invalid {column}")
    if (frame.groupby("patient_group_id").fold.nunique() > 1).any():
        raise ValueError("Patient group crosses source folds")
    return frame

def align_and_fuse(clinical: pd.DataFrame, face: pd.DataFrame) -> pd.DataFrame:
    clinical = validate_identifiers(clinical, "ID")
    face = validate_identifiers(face, "sample_id")
    if not np.array_equal(clinical.fold, clinical.prediction_model_fold):
        raise ValueError("Clinical OOF model fold differs from the saved source fold")
    clinical = clinical.rename(columns={"fold": "clinical_fold"}).drop(columns=["prediction_model_fold", "predicted_label"], errors="ignore")
    selected = face[["sample_id", "patient_group_id", "fold", "binary_label", "sex",
                     "logit_control", "logit_patient", "prob_patient"]].rename(columns={
        "sample_id": "ID", "patient_group_id": "face_patient_group_id", "fold": "face_fold",
        "binary_label": "face_binary_label", "sex": "face_SEX", "prob_patient": "saved_face_probability",
    })
    merged = clinical.merge(selected, on="ID", how="left", validate="one_to_one", sort=False, indicator=True)
    if len(merged) != len(clinical) or not (merged._merge == "both").all():
        raise ValueError("Not all clinical IDs have a face OOF prediction")
    if not np.array_equal(merged.ID, clinical.ID):
        raise ValueError("ID order changed while aligning predictions")
    for left, right in [("patient_group_id", "face_patient_group_id"),
                        ("binary_label", "face_binary_label"), ("SEX", "face_SEX")]:
        if not np.array_equal(merged[left], merged[right]):
            raise ValueError(f"Source mismatch: {left}")
    if "original_500_fold" in merged and not np.array_equal(merged.original_500_fold, merged.face_fold):
        raise ValueError("Face OOF fold differs from the original 500-record fold")
    numeric = ["L_clinical_OOF", "p_clinical_OOF", "logit_control", "logit_patient", "saved_face_probability"]
    if not np.isfinite(merged[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite input predictions")
    np.testing.assert_allclose(expit(merged.L_clinical_OOF), merged.p_clinical_OOF, atol=1e-12, rtol=1e-12)
    merged["L_face_OOF"] = merged.logit_patient - merged.logit_control
    merged["p_face_OOF"] = expit(merged.L_face_OOF)
    np.testing.assert_allclose(merged.p_face_OOF, merged.saved_face_probability, atol=2e-7, rtol=2e-7)
    merged["L_fusion_OOF"] = 0.5 * merged.L_clinical_OOF + 0.5 * merged.L_face_OOF
    merged["p_fusion_OOF"] = expit(merged.L_fusion_OOF)
    for name, column in SCORES.items():
        merged[f"predicted_{name}"] = (merged[column] >= 0).astype(int)
    merged["face_fold"] = merged.face_fold.astype(int)
    return merged.drop(columns=["_merge", "face_patient_group_id", "face_binary_label", "face_SEX", "saved_face_probability"])

def verify_face_oof_manifests(face: pd.DataFrame, experiment: Path) -> None:
    # Epoch selection also must exclude the outer held-out patient groups.
    for fold in range(5):
        manifest = pd.read_csv(experiment / f"fold_{fold}" / "inner_split.csv",
                               dtype={"ID": str, "patient_group_id": str})
        if set(manifest.role) != {"inner_train", "inner_validation", "outer_test"}:
            raise ValueError(f"Unexpected manifest roles in face fold {fold}")
        heldout = manifest.loc[manifest.role == "outer_test"]
        development = manifest.loc[manifest.role != "outer_test"]
        expected = face.loc[face.fold == fold]
        if set(heldout.ID) != set(expected.sample_id):
            raise ValueError(f"Face manifest/OOF IDs disagree in fold {fold}")
        if set(heldout.patient_group_id) & set(development.patient_group_id):
            raise ValueError(f"Face held-out group used in fitting/selection, fold {fold}")
        if set(heldout.ID) & set(development.ID):
            raise ValueError(f"Face held-out record used in fitting/selection, fold {fold}")

def load_inputs(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    clinical = pd.read_csv(resolve_path(d["clinical_predictions"]), dtype={"ID": str, "patient_group_id": str})
    experiment = resolve_path(d["face_experiment"])
    face = pd.read_csv(experiment / "oof_predictions.csv", dtype={"sample_id": str, "patient_group_id": str})
    if len(face) != d["expected_face_source_records"]:
        raise ValueError("Unexpected face source cohort size")
    combined = align_and_fuse(clinical, face)
    if len(combined) != d["expected_records"] or combined.patient_group_id.nunique() != d["expected_groups"]:
        raise ValueError("Unexpected paired cohort size")
    if combined.binary_label.value_counts().reindex([0, 1], fill_value=0).tolist() != d["expected_class_counts"]:
        raise ValueError("Unexpected paired class counts")
    verify_face_oof_manifests(face, experiment)
    return combined

def paired_bootstrap(scores: pd.DataFrame, settings: dict) -> pd.DataFrame:
    y = scores.binary_label.to_numpy(int)
    logits = {name: scores[column].to_numpy(float) for name, column in SCORES.items()}
    points = {name: float(roc_auc_score(y, values)) for name, values in logits.items()}
    points["fusion_minus_clinical"] = points["fusion"] - points["clinical"]
    points["fusion_minus_face"] = points["fusion"] - points["face"]
    draws = {name: [] for name in points}
    codes, groups = pd.factorize(scores.patient_group_id, sort=True)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    accepted = 0
    for _ in range(settings["bootstrap_replicates"]):
        multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = multiplicity[codes]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        aucs = {name: float(roc_auc_score(y, values, sample_weight=weights)) for name, values in logits.items()}
        aucs["fusion_minus_clinical"] = aucs["fusion"] - aucs["clinical"]
        aucs["fusion_minus_face"] = aucs["fusion"] - aucs["face"]
        for name, value in aucs.items():
            draws[name].append(value)
        accepted += 1
    if accepted < 0.95 * settings["bootstrap_replicates"]:
        raise ValueError("Too few valid paired bootstrap samples")
    tail = (1 - settings["confidence_level"]) / 2
    rows = []
    for name, values in draws.items():
        low, high = np.quantile(values, [tail, 1 - tail])
        rows.append({"comparison": name, "estimate": points[name], "ci_lower": low, "ci_upper": high,
                     "confidence_level": settings["confidence_level"], "valid_replicates": accepted,
                     "unit": "patient_group_id", "paired": True, "refit_models": False,
                     "method": "percentile_bootstrap_conditional_on_fixed_source_OOF_predictions"})
    return pd.DataFrame(rows)

def metrics_table(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    y = scores.binary_label.to_numpy(int)
    rows = []
    for name, column in SCORES.items():
        logit = scores[column].to_numpy(float)
        rows.append({"model": name, "records": len(scores), "patient_groups": scores.patient_group_id.nunique(),
                     **basic_metrics(y, logit, cfg["evaluation"]["classification_threshold"]),
                     **calibration_metrics(y, logit, cfg["calibration"])})
    return pd.DataFrame(rows)
