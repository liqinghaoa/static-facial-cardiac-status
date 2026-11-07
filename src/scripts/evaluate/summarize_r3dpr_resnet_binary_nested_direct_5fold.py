"""Validate and summarize pooled outer-held-out predictions for R3DPR nested-direct experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.binary_classification_metrics import compute_binary_metrics, flatten_metrics
from utils.oof_validation import METRIC_LABELS, METRICS, expected_table, validate_oof
from utils.experiment_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def make_report(config: dict, fold_metrics: pd.DataFrame, metrics: dict, matrix: np.ndarray) -> str:
    data, train, nested = config["data"], config["train"], config["nested_direct"]
    backbone = str(config["model"]["backbone"]).replace("resnet", "ResNet")
    loss_name = str(train["loss"]).strip().lower()
    loss_text = (
        f"weighted label-smoothed cross entropy (alpha={float(train['label_smoothing_alpha']):g})"
        if loss_name == "weighted_ce_label_smoothing"
        else "weighted cross entropy"
    )
    metric_rows = "\n".join(f"| {METRIC_LABELS[name]} | {metrics[name]:.4f} |" for name in METRICS)
    fold_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.selected_epoch)} | {row.inner_best_macro_auc:.4f} | "
        f"{row.macro_auc:.4f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | "
        f"{row.sensitivity:.4f} | {row.specificity:.4f} |"
        for row in fold_metrics.itertuples()
    )
    return f"""# R3DPR {backbone} Nested-Direct Control vs Patient Binary Experiment

## Protocol

- Fixed outer five-fold split: 400-case development set and untouched 100-case outer held-out evaluation set per fold.
- Within each 400-case development set, an 80/20 split is generated using only `{data['label_column']}` x `{data['sex_column']}` stratification. `patient_group_id` is not used as a split constraint.
- The 80-case internal validation set selects `selected_epoch` by macro-AUC with patience={train['early_stopping_patience']}.
- The checkpoint with the best internal validation macro-AUC directly predicts its corresponding outer held-out 100 cases; no 400-case refitting stage is performed.
- Outer evaluation data are not used for model selection, early stopping, or epoch selection.
- Input resize: height x width = {data['image_height']} x {data['image_width']}; loss={loss_text}; AdamW lr={train['lr']}, weight decay={train['weight_decay']}.
- Hard labels are softmax argmax. No threshold search was performed.

## Pooled Outer-Held-Out OOF Metrics

| Metric | Value |
|---|---:|
{metric_rows}

## Pooled Outer-Held-Out Confusion Matrix

| True class | Predicted Control | Predicted Patient |
|---|---:|---:|
| Control | {int(matrix[0, 0])} | {int(matrix[0, 1])} |
| Patient | {int(matrix[1, 0])} | {int(matrix[1, 1])} |

## Fold Metrics

| Fold | Selected epoch | Inner best Macro-AUC | Outer Macro-AUC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_rows}

The primary result is the pooled outer-held-out OOF metric set. Per-fold metrics describe stability only.
"""


def summarize(experiment_dir: Path) -> None:
    config = load_yaml(experiment_dir / "config_snapshot.yaml")
    if config.get("nested_direct", {}).get("protocol") != "outer_5fold_inner_holdout_direct_test":
        raise ValueError("Experiment is not a nested-direct run")
    n_folds = int(config["data"]["n_folds"])
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for fold in range(n_folds):
        fold_dir = experiment_dir / f"fold_{fold}"
        prediction_path = fold_dir / "outer_test_predictions.csv"
        metric_path = fold_dir / "metrics.csv"
        if not prediction_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"Missing completed nested-direct artifacts for fold {fold}: {fold_dir}")
        predictions = pd.read_csv(prediction_path, dtype={"sample_id": "string", "patient_group_id": "string"})
        if set(predictions.get("protocol", [])) != {"outer_5fold_inner_holdout_direct_test"}:
            raise ValueError(f"Fold {fold} predictions do not declare the nested-direct protocol")
        prediction_frames.append(predictions)
        metric_frames.append(pd.read_csv(metric_path))
    oof = pd.concat(prediction_frames, ignore_index=True)
    validate_oof(oof, expected_table(config), n_folds)
    metrics = compute_binary_metrics(oof["binary_label"].to_numpy(int), oof[["prob_control", "prob_patient"]].to_numpy(float))
    fold_metrics = pd.concat(metric_frames, ignore_index=True).sort_values("fold").reset_index(drop=True)
    required_fold_columns = {"fold", "selected_epoch", "inner_best_macro_auc", *METRICS}
    missing = sorted(required_fold_columns.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"fold metrics lack required columns: {missing}")
    for fold, frame in enumerate(prediction_frames):
        derived = flatten_metrics(
            compute_binary_metrics(frame["binary_label"].to_numpy(int), frame[["prob_control", "prob_patient"]].to_numpy(float))
        )
        observed = fold_metrics.loc[fold_metrics["fold"] == fold].iloc[0]
        for metric in METRICS:
            if not np.isclose(float(observed[metric]), float(derived[metric]), atol=1e-10):
                raise ValueError(f"Fold {fold} stored {metric} does not match outer predictions")
    oof.to_csv(experiment_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(experiment_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(experiment_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]).to_csv(
        experiment_dir / "oof_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig"
    )
    (experiment_dir / "binary_experiment_results.md").write_text(
        make_report(config, fold_metrics, metrics, metrics["confusion_matrix"]), encoding="utf-8"
    )
    print(f"NESTED_DIRECT_OOF_SUMMARY_COMPLETED={experiment_dir}")


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else PROJECT_ROOT / args.experiment_dir
    summarize(experiment_dir)


if __name__ == "__main__":
    main()
