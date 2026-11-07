"""Shared Logistic regression, evaluation and fixed-fold execution for the current 330-person analysis."""

from __future__ import annotations

import argparse

import json

import logging

import platform

import sys

import warnings

from datetime import datetime

from pathlib import Path

from typing import Any

import joblib

import numpy as np

import pandas as pd

import scipy

import sklearn

import yaml

from scipy.optimize import brentq

from scipy.special import expit

from sklearn.compose import ColumnTransformer

from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ["SEX", "Age_y", "BMI_kg_m2"]

CONTINUOUS = ["Age_y", "BMI_kg_m2"]

IDENTIFIERS = ["ID", "patient_group_id", "fold", "binary_label"]

SNAPSHOT_COLUMNS = [
    "ID", "patient_group_id", "fold", "original_500_fold", "SEX", "sex_name",
    "Age_y", "Height_cm", "Weight_kg", "BMI_kg_m2", "NYHA", "binary_label",
    "binary_name",
]

LOG = logging.getLogger("clinical_logistic")

def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    m, e, d = cfg["model"], cfg["evaluation"], cfg["data"]
    if m["features"] != FEATURES or m["continuous_features"] != CONTINUOUS:
        raise ValueError("This experiment requires exactly SEX, Age_y, BMI_kg_m2")
    expected = {
        "type": "unpenalized_logistic_regression", "solver": "lbfgs",
        "fit_intercept": True, "class_weight": None, "positive_class": 1,
        "standardization": "training_fold_only", "missing_values": "error",
        "nonlinear_terms": False, "interactions": False,
        "variable_selection": False, "hyperparameter_search": False,
        "sex_encoding": {"female": 0, "male": 1},
    }
    for key, value in expected.items():
        if m[key] != value:
            raise ValueError(f"Unsupported model setting: {key}={m[key]!r}")
    if [d[k] for k in ("id_column", "group_column", "fold_column", "outcome_column")] != IDENTIFIERS:
        raise ValueError("Column definitions do not match this experiment")
    if d["folds"] != list(range(5)):
        raise ValueError("Expected the existing folds 0,1,2,3,4")
    if not 0 < e["classification_threshold"] < 1:
        raise ValueError("Classification threshold must be inside (0,1)")
    if e["bootstrap_unit"] != "patient_group_id" or e["bootstrap_refit"]:
        raise ValueError("Only fixed-OOF patient-group bootstrap is implemented")
    if e["bootstrap_replicates"] < 1 or not 0 < e["confidence_level"] < 1:
        raise ValueError("Invalid bootstrap settings")
    if cfg["final_model"] != {
        "fit_on_all_records_after_oof": True,
        "use_full_data_predictions_for_evaluation": False,
    }:
        raise ValueError("Final model must be fitted after OOF and not evaluated in sample")
    return cfg





def make_logistic(settings: dict[str, Any]) -> LogisticRegression:
    kwargs = dict(
        solver="lbfgs", fit_intercept=True, class_weight=None,
        max_iter=int(settings["max_iter"]), tol=float(settings["tolerance"]),
    )
    # scikit-learn >=1.8 replaces penalty=None with C=np.inf. Both mean no penalty.
    version = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
    if version >= (1, 8):
        kwargs.update(C=np.inf, l1_ratio=0.0)
    else:
        kwargs.update(penalty=None if version >= (1, 2) else "none")
    return LogisticRegression(**kwargs)

def fit_model(train: pd.DataFrame, settings: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    preprocessing = ColumnTransformer([
        ("sex", "passthrough", ["SEX"]),
        ("continuous", StandardScaler(), CONTINUOUS),
    ], remainder="drop")
    model = Pipeline([("preprocessing", preprocessing), ("logistic", make_logistic(settings))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[FEATURES], train.binary_label.to_numpy(int))
    regression = model.named_steps["logistic"]
    if not np.array_equal(regression.classes_, [0, 1]):
        raise ValueError("Unexpected class order")
    scaler = model.named_steps["preprocessing"].named_transformers_["continuous"]
    mean = np.r_[0.0, scaler.mean_]
    scale = np.r_[1.0, scaler.scale_]
    standardized = regression.coef_[0].astype(float)
    raw = standardized / scale
    intercept = float(regression.intercept_[0] - np.dot(raw, mean))
    if not np.isfinite(np.r_[intercept, raw]).all():
        raise ValueError("Non-finite model coefficients")
    coefficients = {
        "features": FEATURES, "intercept": intercept,
        "coefficients": dict(zip(FEATURES, raw.tolist())),
        "preprocessing_mean": dict(zip(FEATURES, mean.tolist())),
        "preprocessing_scale": dict(zip(FEATURES, scale.tolist())),
        "standardized_intercept": float(regression.intercept_[0]),
        "standardized_coefficients": dict(zip(FEATURES, standardized.tolist())),
        "n_train_records": len(train),
        "n_train_patient_groups": train.patient_group_id.nunique(),
        "solver_iterations": int(regression.n_iter_[0]),
    }
    return model, coefficients

def predict_records(model: Pipeline, test: pd.DataFrame, coefficients: dict[str, Any],
                    threshold: float) -> pd.DataFrame:
    out = test[[c for c in SNAPSHOT_COLUMNS if c in test.columns]].copy()
    logit = model.decision_function(test[FEATURES])
    raw_beta = np.array([coefficients["coefficients"][f] for f in FEATURES])
    raw_logit = coefficients["intercept"] + test[FEATURES].to_numpy(float) @ raw_beta
    np.testing.assert_allclose(logit, raw_logit, rtol=1e-10, atol=1e-9)
    probability = model.predict_proba(test[FEATURES])[:, 1]
    np.testing.assert_allclose(probability, expit(logit), rtol=1e-12, atol=1e-12)
    if not np.isfinite(logit).all():
        raise ValueError("Non-finite prediction")
    out["L_clinical_OOF"] = logit
    out["p_clinical_OOF"] = probability
    out["predicted_label"] = (probability >= threshold).astype(int)
    out["prediction_model_fold"] = test.fold.to_numpy(int)
    return out

def fit_fold(data: pd.DataFrame, fold: int, cfg: dict[str, Any]):
    train, test = data.loc[data.fold != fold], data.loc[data.fold == fold]
    if set(train.patient_group_id) & set(test.patient_group_id):
        raise ValueError("Patient leakage in train/test split")
    model, coefficients = fit_model(train, cfg["model"])
    predictions = predict_records(model, test, coefficients, cfg["evaluation"]["classification_threshold"])
    return model, coefficients, predictions

def basic_metrics(y: np.ndarray, logit: np.ndarray, threshold: float,
                  weights: np.ndarray | None = None) -> dict[str, float]:
    y, logit = np.asarray(y, int), np.asarray(logit, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    prob = expit(logit)
    pred = prob >= threshold
    tn = float(w[(y == 0) & ~pred].sum())
    fp = float(w[(y == 0) & pred].sum())
    fn = float(w[(y == 1) & ~pred].sum())
    tp = float(w[(y == 1) & pred].sum())

    def ratio(a, b):
        return float(a / b) if b else float("nan")

    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    f1_positive, f1_negative = ratio(2 * tp, 2 * tp + fp + fn), ratio(2 * tn, 2 * tn + fp + fn)
    return {
        "roc_auc": float(roc_auc_score(y, logit, sample_weight=w)),
        "brier_score": float(np.average((prob - y) ** 2, weights=w)),
        "log_loss": float(np.average(np.logaddexp(0, logit) - y * logit, weights=w)),
        "accuracy": ratio(tp + tn, w.sum()),
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
        "f1_patient": f1_positive, "macro_f1": (f1_positive + f1_negative) / 2,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

def calibration_metrics(y: np.ndarray, logit: np.ndarray, settings: dict[str, Any]) -> dict[str, float]:
    y, logit = np.asarray(y, int), np.asarray(logit, float)
    # Offset model: only the intercept is estimated and slope remains fixed at 1.
    intercept_only = brentq(
        lambda a: float((expit(logit + a) - y).sum()),
        -float(logit.max()) - 50, -float(logit.min()) + 50,
    )
    if np.ptp(logit) <= 1e-12:
        return {"calibration_in_the_large": float(intercept_only),
                "calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    # Evaluation only: these coefficients never modify the saved OOF predictions.
    calibration = make_logistic(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibration.fit(logit[:, None], y)
    return {
        "calibration_in_the_large": float(intercept_only),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }

def evaluate(predictions: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    y, logit = predictions.binary_label.to_numpy(int), predictions.L_clinical_OOF.to_numpy(float)
    return {
        **basic_metrics(y, logit, cfg["evaluation"]["classification_threshold"]),
        **calibration_metrics(y, logit, cfg["model"]),
    }

def bootstrap_intervals(oof: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    e = cfg["evaluation"]
    inverse, groups = pd.factorize(oof.patient_group_id, sort=True)
    rng = np.random.default_rng(e["bootstrap_seed"])
    y, logit = oof.binary_label.to_numpy(int), oof.L_clinical_OOF.to_numpy(float)
    estimate = basic_metrics(y, logit, e["classification_threshold"])
    names = [k for k in estimate if k not in {"tn", "fp", "fn", "tp"}]
    draws: dict[str, list[float]] = {k: [] for k in names}
    accepted = 0
    for _ in range(e["bootstrap_replicates"]):
        counts = np.bincount(rng.integers(0, len(groups), size=len(groups)), minlength=len(groups))
        weights = counts[inverse]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        metrics = basic_metrics(y, logit, e["classification_threshold"], weights)
        accepted += 1
        for name in names:
            draws[name].append(metrics[name])
    if accepted < 0.95 * e["bootstrap_replicates"]:
        raise ValueError("Too many bootstrap resamples lacked both outcome classes")
    tail = (1 - e["confidence_level"]) / 2
    rows = []
    for name, values in draws.items():
        finite = np.asarray(values)[np.isfinite(values)]
        low, high = np.quantile(finite, [tail, 1 - tail]) if len(finite) else [np.nan, np.nan]
        rows.append({
            "metric": name, "estimate": estimate[name], "ci_lower": low, "ci_upper": high,
            "confidence_level": e["confidence_level"], "valid_replicates": len(finite),
            "requested_replicates": e["bootstrap_replicates"],
            "unit": "patient_group_id", "refit_models": False,
            "method": "percentile_bootstrap_conditional_on_fixed_OOF_predictions",
        })
    return pd.DataFrame(rows)

def cohort_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ["all", "Control", "Patient"]:
        frame = data if scope == "all" else data.loc[data.binary_label == int(scope == "Patient")]
        row = {"scope": scope, "records": len(frame), "patient_groups": frame.patient_group_id.nunique(),
               "female": int((frame.SEX == 0).sum()), "male": int((frame.SEX == 1).sum()),
               "age_under_18_records": int((frame.Age_y < 18).sum())}
        for column in ["Age_y", "BMI_kg_m2"]:
            for statistic, value in [("min", frame[column].min()), ("max", frame[column].max()),
                                      ("mean", frame[column].mean()), ("sd", frame[column].std()),
                                      ("median", frame[column].median())]:
                row[f"{column}_{statistic}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)

def fold_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in sorted(data.fold.unique()):
        test, train = data.loc[data.fold == fold], data.loc[data.fold != fold]
        row = {"fold": int(fold), "train_records": len(train), "test_records": len(test),
               "train_patient_groups": train.patient_group_id.nunique(),
               "test_patient_groups": test.patient_group_id.nunique(),
               "train_control": int((train.binary_label == 0).sum()),
               "train_patient": int((train.binary_label == 1).sum()),
               "test_control": int((test.binary_label == 0).sum()),
               "test_patient": int((test.binary_label == 1).sum()),
               "test_female": int((test.SEX == 0).sum()), "test_male": int((test.SEX == 1).sum())}
        for label in [0, 1]:
            for sex in [0, 1]:
                row[f"test_label_{label}_sex_{sex}"] = int(((test.binary_label == label) & (test.SEX == sex)).sum())
        for nyha in range(5):
            row[f"test_NYHA_{nyha}"] = int((test.NYHA == nyha).sum())
        rows.append(row)
    return pd.DataFrame(rows)

def coefficient_rows(name: str, info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"model": name, "term": "intercept", "coefficient": info["intercept"], "odds_ratio": np.nan,
             "unit": "log odds at all raw predictors equal to zero"}]
    units = {"SEX": "male (1) versus female (0)", "Age_y": "per 1 year", "BMI_kg_m2": "per 1 kg/m^2"}
    for term, value in info["coefficients"].items():
        rows.append({"model": name, "term": term, "coefficient": value,
                     "odds_ratio": float(np.exp(value)), "unit": units[term]})
    return rows

def save_json(path: Path, payload: Any) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def save_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")

def make_figure(oof: pd.DataFrame, pooled: dict[str, float], output: Path, bins: int) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = oof[["binary_label", "p_clinical_OOF"]].copy()
    table["bin"] = pd.qcut(table.p_clinical_OOF, q=bins, duplicates="drop")
    grouped = table.groupby("bin", observed=True).agg(
        n=("binary_label", "size"), predicted_probability=("p_clinical_OOF", "mean"),
        observed_fraction=("binary_label", "mean"),
    ).reset_index()
    grouped["bin"] = grouped["bin"].astype(str)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
        for fold, frame in oof.groupby("fold"):
            fpr, tpr, _ = roc_curve(frame.binary_label, frame.L_clinical_OOF)
            axes[0].plot(fpr, tpr, alpha=0.45, linewidth=1, label=f"Fold {fold}")
        fpr, tpr, _ = roc_curve(oof.binary_label, oof.L_clinical_OOF)
        axes[0].plot(fpr, tpr, color="#163D67", linewidth=2, label=f"Pooled OOF AUC = {pooled['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Clinical model: OOF ROC",
                    xlim=(0, 1), ylim=(0, 1))
        axes[0].legend(loc="lower right", fontsize=8, frameon=False)
        axes[1].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1, label="Ideal calibration")
        axes[1].plot(grouped.predicted_probability, grouped.observed_fraction, "o-", color="#163D67",
                     label=f"Quantile bins (n={len(oof)})")
        p = np.linspace(oof.p_clinical_OOF.min(), oof.p_clinical_OOF.max(), 200)
        lp = np.log(p / (1 - p))
        axes[1].plot(p, expit(pooled["calibration_intercept"] + pooled["calibration_slope"] * lp),
                     color="#BF6831", label="Logistic calibration fit")
        axes[1].set(xlabel="Predicted probability", ylabel="Observed outcome fraction",
                    title="Clinical model: OOF calibration", xlim=(0, 1), ylim=(0, 1))
        axes[1].legend(loc="lower right", fontsize=8, frameon=False)
        fig.savefig(output / "oof_roc_calibration.png", dpi=220)
        fig.savefig(output / "oof_roc_calibration.pdf")
        plt.close(fig)
    return grouped



def run_fixed_fivefold(load_data, write_report) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="Must be a new directory; never overwrite an existing experiment")
    parser.add_argument("--preflight", action="store_true", help="Validate input and fit fold 0 without writing outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    cfg = load_config(resolve_path(args.config))
    data = load_data(cfg)
    LOG.info("Input valid: records=%d patient_groups=%d fold_sizes=%s age_range=%s..%s under18=%d",
             len(data), data.patient_group_id.nunique(), data.fold.value_counts().sort_index().tolist(),
             data.Age_y.min(), data.Age_y.max(), (data.Age_y < 18).sum())
    if args.preflight:
        model, info, predictions = fit_fold(data, 0, cfg)
        evaluate(predictions, cfg)
        LOG.info("PREFLIGHT PASSED: fold=0 train=%d test=%d iterations=%d; raw logit/probability checks passed; no files written",
                 info["n_train_records"], len(predictions), info["solver_iterations"])
        return
    output = resolve_path(args.output_dir or cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(handler)
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
             "experiment_dir": str(output), "input_workbook": str(resolve_path(cfg["data"]["workbook"])),
             "completed_folds": [], "completed_oof_records": 0,
             "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
             "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                          "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    try:
        (output / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        save_json(output / "run_info.json", state)
        save_csv(output / "input_snapshot.csv", data[[c for c in SNAPSHOT_COLUMNS if c in data.columns]])
        summary = fold_summary(data)
        save_csv(output / "fold_summary.csv", summary)
        save_csv(output / "cohort_summary.csv", cohort_summary(data))
        predictions, metric_rows, coefficients = [], [], []
        LOG.info("FULL RUN STARTED: fixed fivefold CV; output=%s", output)
        for fold in cfg["data"]["folds"]:
            model, info, test_predictions = fit_fold(data, fold, cfg)
            directory = output / f"fold_{fold}"
            directory.mkdir()
            joblib.dump(model, directory / "model.joblib")
            save_json(directory / "coefficients_raw_units.json", info)
            save_csv(directory / "train_ids.csv", data.loc[data.fold != fold, IDENTIFIERS])
            save_csv(directory / "test_predictions.csv", test_predictions)
            metrics = {**summary.loc[summary.fold == fold].iloc[0].to_dict(), **evaluate(test_predictions, cfg)}
            metric_rows.append(metrics)
            predictions.append(test_predictions)
            coefficients.extend(coefficient_rows(f"fold_{fold}", info))
            state["completed_folds"].append(fold)
            state["completed_oof_records"] += len(test_predictions)
            save_json(output / "run_info.json", state)
            save_csv(output / "metrics_by_fold.csv", pd.DataFrame(metric_rows))
            LOG.info("fold=%d train=%d test=%d iterations=%d OOF_AUC=%.4f Brier=%.4f",
                     fold, info["n_train_records"], len(test_predictions), info["solver_iterations"],
                     metrics["roc_auc"], metrics["brier_score"])
        oof = pd.concat(predictions).sort_index()
        if len(oof) != len(data) or not oof.index.equals(data.index) or oof.ID.duplicated().any():
            raise ValueError("OOF coverage/order is invalid")
        if not np.array_equal(oof.ID, data.ID) or not np.array_equal(oof.prediction_model_fold, data.fold):
            raise ValueError("OOF identifiers or model folds are misaligned")
        save_csv(output / "oof_predictions.csv", oof)
        pooled = evaluate(oof, cfg)
        fold_metrics = pd.DataFrame(metric_rows)
        save_json(output / "metrics_summary.json", {
            "records": len(data), "patient_groups": data.patient_group_id.nunique(),
            "threshold": cfg["evaluation"]["classification_threshold"], "pooled_oof": pooled,
            "fold_mean": {k: float(fold_metrics[k].mean()) for k in pooled},
            "fold_sd_ddof1": {k: float(fold_metrics[k].std(ddof=1)) for k in pooled},
            "evaluation_unit": "record", "validation_type": "fixed_patient_grouped_fivefold_internal_validation",
        })
        LOG.info("OOF complete: %d records; pooled AUC=%.4f. Computing patient-group bootstrap intervals.", len(oof), pooled["roc_auc"])
        intervals = bootstrap_intervals(oof, cfg)
        save_csv(output / "oof_metrics_cluster_bootstrap_ci.csv", intervals)
        model, final_info = fit_model(data, cfg["model"])
        final_dir = output / "final_model"
        final_dir.mkdir()
        joblib.dump(model, final_dir / "model.joblib")
        save_json(final_dir / "coefficients_raw_units.json", final_info)
        coefficients.extend(coefficient_rows("full_data_final", final_info))
        save_csv(output / "coefficients_raw_units.csv", pd.DataFrame(coefficients))
        bins = make_figure(oof, pooled, output, cfg["evaluation"]["calibration_bins"])
        save_csv(output / "calibration_bins.csv", bins)
        write_report(output, cfg, data, pooled, fold_metrics, intervals, final_info)
        state.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                     final_model_fitted=True, bootstrap_replicates=cfg["evaluation"]["bootstrap_replicates"])
        save_json(output / "run_info.json", state)
        LOG.info("COMPLETED: pooled OOF AUC=%.4f accuracy=%.4f Brier=%.4f", pooled["roc_auc"], pooled["accuracy"], pooled["brier_score"])
        print(f"EXPERIMENT_DIR={output}")
    except Exception as error:
        state.update(status="failed", failed_at=datetime.now().astimezone().isoformat(), error=repr(error))
        save_json(output / "run_info.json", state)
        LOG.exception("Run failed after %d folds/%d OOF records; partial outputs retained at %s",
                      len(state["completed_folds"]), state["completed_oof_records"], output)
        raise
