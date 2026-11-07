"""Evaluate fixed equal-weight fusion of 330-person basic and face OOF scores."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import fixed_fusion as evaluation

LOG = logging.getLogger("basic_info_face_330_fusion")


def write_report(output: Path, cfg: dict, scores: pd.DataFrame,
                 metrics: pd.DataFrame, intervals: pd.DataFrame) -> None:
    ci = intervals.set_index("comparison")
    labels = {"clinical": "基础信息", "face": "全脸ResNet18", "fusion": "基础信息＋面部"}
    rows = ["|模型|记录数|汇总OOF AUC|条件性95% CI|Brier|准确率|敏感度|特异度|",
            "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in metrics.to_dict("records"):
        interval = ci.loc[row["model"]]
        rows.append(
            f"|{labels[row['model']]}|{row['records']}|{row['roc_auc']:.4f}|"
            f"{interval.ci_lower:.4f}–{interval.ci_upper:.4f}|{row['brier_score']:.4f}|"
            f"{row['accuracy']:.2%}|{row['sensitivity']:.2%}|{row['specificity']:.2%}|"
        )
    delta = ci.loc["fusion_minus_clinical"]
    face_delta = ci.loc["fusion_minus_face"]
    different_folds = int((scores.clinical_fold != scores.face_fold).sum())
    text = f"""# 330人队列基础信息＋面部固定融合

## 结果

{chr(10).join(rows)}

- 融合减基础信息模型的AUC差值为**{delta.estimate:+.4f}**，条件性配对95% CI为**{delta.ci_lower:+.4f}至{delta.ci_upper:+.4f}**。
- 融合减面部模型的AUC差值为{face_delta.estimate:+.4f}，条件性配对95% CI为{face_delta.ci_lower:+.4f}至{face_delta.ci_upper:+.4f}。

## 固定公式与评价设计

```text
L_face = logit_patient − logit_control
L_fusion = 0.5 × L_basic + 0.5 × L_face
p_fusion = sigmoid(L_fusion)
```

权重0.5/0.5、截距0及阈值0.5在评价前固定，没有训练融合回归或根据本队列调整权重。基础信息分数来自本330人队列的性别、年龄、BMI五折Logistic回归OOF预测；面部分数按ID复用原500例嵌套五折ResNet18的OOF预测。模型均未用对应患者及其患者组训练或选择。

基础信息与面部来源折号有{different_folds}/330条不同。结果是在相同330名患者上对两套OOF分数作固定融合的内部评价，不表示两种来源模型使用同一外层折划分，也不是重新训练的联合模型。ID、患者组、结局、性别及原面部折号均逐条核对；五个面部外层留出清单通过检查。

## 条件性区间与解释范围

以`patient_group_id`为单位进行{cfg['evaluation']['bootstrap_replicates']}次配对百分位bootstrap，种子{cfg['evaluation']['bootstrap_seed']}；同一重抽样同时应用于各模型，因此AUC差值为配对差值。区间条件于固定OOF预测，不包含模型重训或分折的不确定性，也不构成外部验证。

330人队列包含4名未成年人，均按研究者决定纳入。面部数据已识别的设备与标签关联仍可能影响结果，融合增益不能单独归因于疾病特异性面部信息。

## 文件

- `paired_oof_predictions_330.csv`：共同330人的三种评分、概率及两套折号。
- `metrics_comparison.csv`：汇总OOF指标。
- `auc_paired_cluster_bootstrap_ci.csv`：模型AUC和配对AUC差值的条件性区间。
- `config_used.yaml`、`run_info.json`、`evaluation.log`：配置与运行记录。
"""
    (output / "fixed_fusion_experiment_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    cfg = evaluation.load_config(evaluation.resolve_path(args.config))
    scores = evaluation.load_inputs(cfg)
    LOG.info("Paired inputs valid: records=%d patient_groups=%d; face outer-holdout manifests verified",
             len(scores), scores.patient_group_id.nunique())
    if args.preflight:
        np.testing.assert_allclose(
            scores.L_fusion_OOF,
            0.5 * scores.L_clinical_OOF + 0.5 * scores.L_face_OOF,
            atol=0, rtol=0,
        )
        LOG.info("PREFLIGHT PASSED: IDs, labels, groups, folds, face manifests, and fixed formula verified; no output written")
        return

    output = evaluation.resolve_path(args.output_dir or cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "evaluation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(handler)
    state = {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "experiment_dir": str(output), "records": len(scores),
        "patient_groups": scores.patient_group_id.nunique(), "models_refitted": False,
        "fusion_parameters_fitted": False, "different_source_fold_records": int((scores.clinical_fold != scores.face_fold).sum()),
        "face_heldout_manifests_verified": True, "python_executable": sys.executable,
        "platform": platform.platform(), "sklearn_version": sklearn.__version__,
    }
    try:
        LOG.info("FULL EVALUATION STARTED: fixed 0.5/0.5 fusion; output=%s", output)
        (output / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        evaluation.save_json(output / "run_info.json", state)
        evaluation.save_csv(output / "paired_oof_predictions_330.csv", scores)
        metrics = evaluation.metrics_table(scores, cfg)
        evaluation.save_csv(output / "metrics_comparison.csv", metrics)
        intervals = evaluation.paired_bootstrap(scores, cfg["evaluation"])
        evaluation.save_csv(output / "auc_paired_cluster_bootstrap_ci.csv", intervals)
        write_report(output, cfg, scores, metrics, intervals)
        for row in metrics.to_dict("records"):
            LOG.info("model=%s AUC=%.6f Brier=%.6f", row["model"], row["roc_auc"], row["brier_score"])
        state.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                     bootstrap_replicates=cfg["evaluation"]["bootstrap_replicates"])
        evaluation.save_json(output / "run_info.json", state)
        delta = intervals.set_index("comparison").loc["fusion_minus_clinical"]
        LOG.info("COMPLETED: delta AUC=%+.6f, conditional 95%% CI [%+.6f, %+.6f]",
                 delta.estimate, delta.ci_lower, delta.ci_upper)
        print(f"EXPERIMENT_DIR={output}")
    except Exception as error:
        state.update(status="failed", error=repr(error))
        evaluation.save_json(output / "run_info.json", state)
        LOG.exception("Evaluation failed; partial outputs retained at %s", output)
        raise


if __name__ == "__main__":
    main()
