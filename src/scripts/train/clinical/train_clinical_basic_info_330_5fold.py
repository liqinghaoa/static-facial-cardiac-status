"""Train the SEX, age, and BMI model on the fixed 330-person cohort."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import clinical_logistic as trainer


def load_330_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg["data"]
    data = pd.read_excel(
        trainer.resolve_path(settings["workbook"]), sheet_name=settings["sheet"],
        dtype={"ID": str, "patient_group_id": str},
    )
    source = pd.read_excel(
        trainer.resolve_path(settings["validation_source_workbook"]),
        sheet_name=settings["validation_source_sheet"],
        dtype={"ID": str, "patient_group_id": str},
    )
    source_columns = [
        "ID", "patient_group_id", "SEX", "Age_y", "Height_cm", "Weight_kg",
        "NYHA", "binary_label",
    ]
    missing_source = sorted(set(source_columns) - set(source.columns))
    if missing_source:
        raise ValueError(f"Validation source missing columns: {missing_source}")
    source = source[source_columns]
    joined = data.merge(
        source, on="ID", how="left", suffixes=("", "_source"),
        indicator=True, validate="one_to_one", sort=False,
    )
    if len(joined) != len(data) or not joined["_merge"].eq("both").all():
        raise ValueError("Some 330 cohort IDs do not match the height/weight source")
    for column in ("patient_group_id", "SEX", "Age_y", "NYHA", "binary_label"):
        if not np.array_equal(joined[column], joined[f"{column}_source"]):
            raise ValueError(f"330 cohort/source mismatch in {column}")
    joined = joined.drop(columns=["_merge", "patient_group_id_source", "SEX_source",
                                 "Age_y_source", "NYHA_source", "binary_label_source"])

    # The source workbook contains a known height transcription error confirmed by the researcher.
    for sample_id, height_cm in settings.get("height_overrides_cm", {}).items():
        mask = joined.ID.astype(str).eq(str(sample_id))
        if int(mask.sum()) != 1:
            raise ValueError(f"Height override must match exactly one ID: {sample_id}")
        joined.loc[mask, "Height_cm"] = float(height_cm)

    required = ["ID", "patient_group_id", "fold", "binary_label", "SEX", "Age_y",
                "BMI_kg_m2", "Height_cm", "Weight_kg", "NYHA"]
    if joined[required].isna().any().any():
        raise ValueError("Required cohort fields contain missing values")
    for column in required[2:]:
        joined[column] = pd.to_numeric(joined[column], errors="raise")
    numeric = joined[required[2:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Required cohort fields contain non-finite values")
    if joined.ID.astype(str).str.strip().eq("").any() or joined.ID.duplicated().any():
        raise ValueError("Sample IDs must be non-empty and unique")
    if joined.patient_group_id.astype(str).str.strip().eq("").any():
        raise ValueError("Patient group IDs must be non-empty")
    for column, allowed in (("SEX", [0, 1]), ("binary_label", [0, 1]),
                            ("fold", list(range(5))), ("NYHA", list(range(5)))):
        if not joined[column].isin(allowed).all():
            raise ValueError(f"Invalid values in {column}")
        joined[column] = joined[column].astype(int)
    if (joined[["Age_y", "Height_cm", "Weight_kg", "BMI_kg_m2"]] <= 0).any().any():
        raise ValueError("Age, height, weight, and BMI must be positive")
    bmi = joined.Weight_kg / (joined.Height_cm / 100.0) ** 2
    if not np.allclose(joined.BMI_kg_m2, bmi, rtol=1e-10, atol=1e-10):
        raise ValueError("BMI does not match weight/(height/100)^2 after confirmed correction")
    if not np.array_equal((joined.NYHA > 0).astype(int), joined.binary_label):
        raise ValueError("NYHA and binary labels do not match")
    if (joined.groupby("patient_group_id").fold.nunique() > 1).any():
        raise ValueError("A patient group occurs in multiple folds")
    if (joined.groupby("patient_group_id").binary_label.nunique() > 1).any():
        raise ValueError("Outcome differs within a patient group")

    expected = settings
    if len(joined) != expected["expected_records"]:
        raise ValueError(f"Expected {expected['expected_records']} records, found {len(joined)}")
    if joined.patient_group_id.nunique() != expected["expected_groups"]:
        raise ValueError("Unexpected number of patient groups")
    if joined.binary_label.value_counts().reindex([0, 1], fill_value=0).tolist() != expected["expected_class_counts"]:
        raise ValueError("Unexpected overall class counts")
    for fold, (expected_n, expected_classes) in enumerate(zip(
        expected["expected_fold_sizes"], expected["expected_fold_class_counts"]
    )):
        test = joined.loc[joined.fold == fold]
        if len(test) != expected_n:
            raise ValueError(f"Unexpected sample count in fold {fold}")
        if test.binary_label.value_counts().reindex([0, 1], fill_value=0).tolist() != expected_classes:
            raise ValueError(f"Unexpected class counts in fold {fold}")
    return joined.drop(columns=["ID_source"], errors="ignore").reset_index(drop=True)


def write_330_report(output: Path, cfg: dict[str, Any], data: pd.DataFrame,
                     pooled: dict[str, float], folds: pd.DataFrame,
                     intervals: pd.DataFrame, final_info: dict[str, Any]) -> None:
    ci = intervals.set_index("metric")
    metrics = ["roc_auc", "brier_score", "log_loss", "accuracy", "sensitivity",
               "specificity", "balanced_accuracy", "ppv", "npv", "macro_f1",
               "calibration_in_the_large", "calibration_intercept", "calibration_slope"]
    metric_rows = ["|指标|汇总OOF|条件性95% CI|五折均值 ± SD|", "|---|---:|---|---:|"]
    for name in metrics:
        if name not in pooled:
            continue
        if name in ci.index:
            interval = ci.loc[name]
            ci_text = f"{interval.ci_lower:.4f}–{interval.ci_upper:.4f}" if np.isfinite(interval.ci_lower) and np.isfinite(interval.ci_upper) else "不可计算"
        else:
            ci_text = "未计算"
        metric_rows.append(f"|{name}|{pooled[name]:.4f}|{ci_text}|{folds[name].mean():.4f} ± {folds[name].std(ddof=1):.4f}|")

    fold_rows = ["|fold|训练/留出人数|对照/患者|AUC|Brier|敏感度|特异度|",
                 "|---|---:|---:|---:|---:|---:|---:|"]
    for row in folds.to_dict("records"):
        fold_rows.append(
            f"|{int(row['fold'])}|{int(row['train_records'])}/{int(row['test_records'])}|"
            f"{int(row['test_control'])}/{int(row['test_patient'])}|{row['roc_auc']:.4f}|"
            f"{row['brier_score']:.4f}|{row['sensitivity']:.4f}|{row['specificity']:.4f}|"
        )
    beta = final_info["coefficients"]
    formula = (f"L = {final_info['intercept']:.12g} {beta['SEX']:+.12g} × SEX "
               f"{beta['Age_y']:+.12g} × Age_y {beta['BMI_kg_m2']:+.12g} × BMI_kg_m2")
    text = f"""# 330人队列基础信息 Logistic 模型

## 队列与方法

- 输入为`{trainer.resolve_path(cfg['data']['workbook'])}`的`{cfg['data']['sheet']}`工作表，共{len(data)}条记录、{data.patient_group_id.nunique()}名患者；对照{int((data.binary_label == 0).sum())}人、患者{int((data.binary_label == 1).sum())}人。
- 按该330人队列预先固定的`fold`进行五折内部验证，每折66人；每轮264人训练、66人留出。未重新划分折号。
- 4名未成年人依研究者决定保留；年龄范围为{data.Age_y.min():g}–{data.Age_y.max():g}岁。性别为女性{int((data.SEX == 0).sum())}人、男性{int((data.SEX == 1).sum())}人。
- 模型输入为SEX（女性0、男性1）、Age_y及BMI_kg_m2。无惩罚Logistic回归，含截距；年龄和BMI的标准化仅在当轮训练折拟合，性别不标准化。无类别加权、变量筛选、非线性项、交互项、超参数搜索或插补。
- 分类阈值为0.5。汇总330条OOF线性预测值与概率计算性能；全330人拟合的最终模型只用于保存系数，不用于评价。
- BMI完整性以身高体重来源表核对。个体校正值保留在作者受保护的研究记录中，未修改源文件。

## 五折结果

{chr(10).join(fold_rows)}

## 汇总OOF结果

{chr(10).join(metric_rows)}

置信区间以患者组为单位对固定OOF预测进行{cfg['evaluation']['bootstrap_replicates']}次百分位bootstrap，种子{cfg['evaluation']['bootstrap_seed']}。区间条件于已拟合模型与固定分折，不包含重新训练或重新分折的不确定性，也不构成外部验证。

![330人基础信息模型OOF ROC及校准图](oof_roc_calibration.png)

## 全队列最终模型

五折评价后另用全部330人拟合的模型公式为：

```text
{formula}
p = 1 / (1 + exp(-L))
```

精确系数及可加载模型保存在`final_model/`。该模型未用于上述OOF性能评价。

## 复现文件

- `oof_predictions.csv`：每名患者一次留出预测，含logit、概率及折号。
- `metrics_summary.json`、`metrics_by_fold.csv`、`oof_metrics_cluster_bootstrap_ci.csv`：汇总、分折及条件性区间。
- `input_snapshot.csv`、`fold_summary.csv`、`cohort_summary.csv`：输入快照、折分及队列描述。
- `fold_*/`、`final_model/`：折模型、训练ID、留出预测及全队列模型。
- `config_used.yaml`、`run_info.json`、`training.log`：配置与运行记录。
"""
    (output / "clinical_experiment_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    trainer.run_fixed_fivefold(load_330_cohort, write_330_report)


if __name__ == "__main__":
    main()
