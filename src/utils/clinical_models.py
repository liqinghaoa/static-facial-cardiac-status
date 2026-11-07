"""Shared five-variable clinical model functions used by the 330-person analysis. No 340-person entry point."""

from __future__ import annotations

import warnings

import numpy as np

import pandas as pd

from scipy.special import expit

from sklearn.compose import ColumnTransformer

from sklearn.exceptions import ConvergenceWarning

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from utils.clinical_logistic import basic_metrics, calibration_metrics, make_logistic, resolve_path

from utils.fixed_fusion import validate_identifiers, verify_face_oof_manifests

VARIANTS = {"clinical5": ["SEX", "Age_y", "BMI_kg_m2", "BNP_pg_mL", "hs_cTnT_source_unit"]}

def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]
    data = pd.read_excel(resolve_path(d["workbook"]), sheet_name=d["sheet"], engine="openpyxl",
                         dtype={"ID": str, "patient_group_id": str})
    data = validate_identifiers(data, "ID").reset_index(drop=True)
    features = VARIANTS["clinical5"]
    for column in features + ["NYHA", "original_500_fold", "source_460_fold"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[features + ["NYHA", "original_500_fold", "source_460_fold"]].to_numpy(float)).all():
        raise ValueError("Non-finite required data; no imputation is allowed")
    if not data.SEX.isin([0, 1]).all() or not data.NYHA.isin(range(5)).all():
        raise ValueError("Invalid sex/NYHA coding")
    if not np.array_equal((data.NYHA > 0).astype(int), data.binary_label):
        raise ValueError("Binary label does not match NYHA mapping")
    if (data[features[1:]] <= 0).any().any():
        raise ValueError("Age, BMI and biomarkers must retain the positive source values")
    if len(data) != d["expected_records"] or data.patient_group_id.nunique() != d["expected_groups"]:
        raise ValueError("Unexpected clinical cohort size")
    if data.binary_label.value_counts().reindex([0, 1], fill_value=0).tolist() != d["expected_class_counts"]:
        raise ValueError("Unexpected outcome distribution")
    if (data.groupby("patient_group_id").binary_label.nunique() > 1).any():
        raise ValueError("Outcome differs within patient groups")
    for fold in d["folds"]:
        subset = data.loc[data.fold == fold]
        if len(subset) != d["expected_records_per_fold"] or subset.patient_group_id.nunique() != d["expected_groups_per_fold"]:
            raise ValueError(f"Unexpected fold size: {fold}")
        if subset.binary_label.value_counts().reindex([0, 1], fill_value=0).tolist() != d["expected_test_class_counts"][fold]:
            raise ValueError(f"Unexpected class counts: fold {fold}")
    face_root = resolve_path(d["face_experiment"])
    face = pd.read_csv(face_root / "oof_predictions.csv", dtype={"sample_id": str, "patient_group_id": str})
    face = validate_identifiers(face, "sample_id")
    if len(face) != d["expected_face_source_records"]:
        raise ValueError("Unexpected face source size")
    verify_face_oof_manifests(face, face_root)
    # Check source alignment before any clinical fit or experiment-directory creation.
    align_face(data, face)
    return data, face

def fit_model(train: pd.DataFrame, features: list[str], settings: dict) -> tuple[Pipeline, dict]:
    continuous = [column for column in features if column != "SEX"]
    preprocessor = ColumnTransformer([
        ("sex", "passthrough", ["SEX"]), ("continuous", StandardScaler(), continuous),
    ], remainder="drop")
    model = Pipeline([("preprocessing", preprocessor), ("logistic", make_logistic(settings))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[features], train.binary_label.to_numpy(int))
    regression = model.named_steps["logistic"]
    if not np.array_equal(regression.classes_, [0, 1]):
        raise ValueError("Unexpected class order")
    scaler = model.named_steps["preprocessing"].named_transformers_["continuous"]
    mean, scale = np.r_[0.0, scaler.mean_], np.r_[1.0, scaler.scale_]
    beta_scaled = regression.coef_[0]
    beta = beta_scaled / scale
    info = {
        "features": features, "intercept": float(regression.intercept_[0] - np.dot(beta, mean)),
        "coefficients": dict(zip(features, beta.tolist())),
        "preprocessing_mean": dict(zip(features, mean.tolist())),
        "preprocessing_scale": dict(zip(features, scale.tolist())),
        "standardized_intercept": float(regression.intercept_[0]),
        "standardized_coefficients": dict(zip(features, beta_scaled.tolist())),
        "n_train_records": len(train), "n_train_patient_groups": train.patient_group_id.nunique(),
        "solver_iterations": int(regression.n_iter_[0]), "biomarker_transformation": "none",
    }
    return model, info

def fit_fold(data: pd.DataFrame, fold: int, variant: str, cfg: dict):
    train, test = data.loc[data.fold != fold], data.loc[data.fold == fold]
    if set(train.patient_group_id) & set(test.patient_group_id):
        raise ValueError("Patient group leakage between train and test")
    features = VARIANTS[variant]
    model, info = fit_model(train, features, cfg["model"])
    logit = model.decision_function(test[features])
    raw = info["intercept"] + test[features].to_numpy(float) @ np.array([info["coefficients"][c] for c in features])
    np.testing.assert_allclose(raw, logit, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(expit(logit), model.predict_proba(test[features])[:, 1], atol=1e-12, rtol=1e-12)
    if not np.isfinite(logit).all():
        raise ValueError("Non-finite fold prediction")
    prediction = test.copy()
    prediction[f"L_{variant}_OOF"] = logit
    prediction[f"p_{variant}_OOF"] = expit(logit)
    prediction[f"predicted_{variant}"] = (logit >= 0).astype(int)
    return model, info, prediction

def align_face(data: pd.DataFrame, face: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "patient_group_id", "binary_label", "sex", "fold",
               "logit_control", "logit_patient", "prob_patient"]
    subset = face[columns].rename(columns={
        "sample_id": "ID", "patient_group_id": "face_patient_group_id", "binary_label": "face_binary_label",
        "sex": "face_SEX", "fold": "face_fold", "prob_patient": "saved_face_probability",
    })
    paired = data.merge(subset, how="left", on="ID", validate="one_to_one", sort=False, indicator=True)
    if len(paired) != len(data) or not (paired._merge == "both").all() or not np.array_equal(paired.ID, data.ID):
        raise ValueError("Incomplete or misordered face OOF match")
    for a, b in [("patient_group_id", "face_patient_group_id"), ("binary_label", "face_binary_label"),
                 ("SEX", "face_SEX"), ("original_500_fold", "face_fold")]:
        if not np.array_equal(paired[a], paired[b]):
            raise ValueError(f"Face/clinical source mismatch: {a}")
    paired["L_face_OOF"] = paired.logit_patient - paired.logit_control
    paired["p_face_OOF"] = expit(paired.L_face_OOF)
    if not np.isfinite(paired[["logit_patient", "logit_control", "L_face_OOF"]].to_numpy(float)).all():
        raise ValueError("Non-finite face scores")
    np.testing.assert_allclose(paired.p_face_OOF, paired.saved_face_probability, atol=2e-7, rtol=2e-7)
    paired["face_fold"] = paired.face_fold.astype(int)
    return paired.drop(columns=["_merge", "face_patient_group_id", "face_binary_label", "face_SEX", "saved_face_probability"])

def evaluate(y: np.ndarray, logit: np.ndarray, cfg: dict) -> dict:
    return {**basic_metrics(y, logit, cfg["evaluation"]["classification_threshold"]),
            **calibration_metrics(y, logit, cfg["model"])}

def coefficient_rows(model_name: str, info: dict) -> list[dict]:
    rows = [{"model": model_name, "term": "intercept", "coefficient": info["intercept"], "odds_ratio": np.nan,
             "unit": "raw-feature log odds intercept"}]
    units = {"SEX": "male vs female", "Age_y": "per 1 year", "BMI_kg_m2": "per 1 kg/m^2",
             "BNP_pg_mL": "per 1 pg/mL", "hs_cTnT_source_unit": "per 1 source unit"}
    for name, value in info["coefficients"].items():
        rows.append({"model": model_name, "term": name, "coefficient": value,
                     "odds_ratio": float(np.exp(value)), "unit": units[name]})
    return rows

EXPECTED_CLINICAL_STRATEGY = {'model': {'type': 'unpenalized_logistic_regression', 'solver': 'lbfgs', 'fit_intercept': True, 'class_weight': None, 'max_iter': 5000, 'tolerance': 1e-10, 'sex_encoding': {'female': 0, 'male': 1}, 'positive_class': 1, 'standardization': 'training_fold_only', 'missing_values': 'error', 'nonlinear_terms': False, 'interactions': False, 'variable_selection': False, 'hyperparameter_search': False, 'biomarker_transformation': 'none'}, 'fusion': {'clinical_weight': 0.5, 'face_weight': 0.5, 'intercept': 0.0, 'face_score': 'logit_patient_minus_logit_control', 'fit_parameters': False}, 'evaluation': {'classification_threshold': 0.5, 'bootstrap_replicates': 2000, 'bootstrap_seed': 2026, 'confidence_level': 0.95, 'bootstrap_unit': 'patient_group_id', 'bootstrap_refit': False, 'calibration_bins': 10}, 'final_model': {'fit_on_all_records_after_oof': True, 'use_full_data_predictions_for_evaluation': False}}
