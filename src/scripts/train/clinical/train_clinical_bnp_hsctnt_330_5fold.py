"""Time-aligned 330-patient clinical fivefold CV and fixed face-logit fusion.

Reuse the original Logistic fitting, preprocessing, calibration and face-OOF
alignment functions. Only the corrected cohort and its independent folds change.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.clinical_models import load_inputs, fit_fold, fit_model, align_face, evaluate, coefficient_rows
from utils.clinical_logistic import resolve_path, save_csv, save_json, fold_summary, cohort_summary

LOG = logging.getLogger("clinical_bnp_hsctnt_330")
FEATURES = ["SEX", "Age_y", "BMI_kg_m2", "BNP_pg_mL", "hs_cTnT_source_unit"]
SCORES = {"clinical5": "L_clinical5_OOF", "face": "L_face_OOF", "clinical5_face": "L_clinical5_face_OOF"}
LABELS = {"clinical5": "临床模型", "face": "面部模型", "clinical5_face": "临床＋面部固定融合"}
CONTRASTS = {"clinical5_face_minus_clinical5": ("clinical5_face", "clinical5"),
             "clinical5_face_minus_face": ("clinical5_face", "face")}


def config_and_inputs(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    from utils.clinical_models import EXPECTED_CLINICAL_STRATEGY
    old = EXPECTED_CLINICAL_STRATEGY
    for section in ["model", "fusion", "evaluation", "final_model"]:
        if cfg[section] != old[section]:
            raise ValueError(f"Strategy differs from original experiment: {section}")
    if cfg["variants"] != {"clinical5": FEATURES}:
        raise ValueError("Expected only the requested five-variable clinical model")
    if cfg["data"]["expected_records"] != 330 or cfg["data"]["biomarker_window_days"] != 7:
        raise ValueError("Expected the corrected 330-patient, seven-day cohort")
    data, face = load_inputs(cfg)
    if not data.ID.is_unique or not data.patient_group_id.is_unique:
        raise ValueError("The corrected cohort must contain one record per patient")
    photo = pd.to_datetime(data.EXIF_DateTimeOriginal, errors="raise")
    for marker in ["BNP", "hs_cTnT"]:
        time = pd.to_datetime(data[f"{marker}_application_time"], errors="raise")
        gaps = (photo - time).dt.total_seconds() / 86400
        if not gaps.between(0, 7).all():
            raise ValueError(f"Invalid {marker} application/photo time pairing")
        # XLSX date serialization/readback may round a timestamp by one millisecond.
        np.testing.assert_allclose(gaps, data[f"{marker}_gap_days"], rtol=0, atol=2 / 86400000)
    return cfg, data, face


def paired_bootstrap(scores, settings):
    y = scores.binary_label.to_numpy(int)
    values = {name: scores[column].to_numpy(float) for name, column in SCORES.items()}
    points = {name: float(roc_auc_score(y, v)) for name, v in values.items()}
    for name, (left, right) in CONTRASTS.items():
        points[name] = points[left] - points[right]
    draws = {name: [] for name in points}
    codes, groups = pd.factorize(scores.patient_group_id, sort=True)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    for i in range(settings["bootstrap_replicates"]):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = counts[codes]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        current = {name: float(roc_auc_score(y, v, sample_weight=weights)) for name, v in values.items()}
        for name, (left, right) in CONTRASTS.items():
            current[name] = current[left] - current[right]
        for name, value in current.items():
            draws[name].append(value)
        if (i + 1) % 500 == 0:
            LOG.info("Paired patient bootstrap: %d/%d", i + 1, settings["bootstrap_replicates"])
    tail = (1 - settings["confidence_level"]) / 2
    rows = []
    for name, samples in draws.items():
        if len(samples) < .95 * settings["bootstrap_replicates"]:
            raise ValueError("Too few valid bootstrap samples")
        low, high = np.quantile(samples, [tail, 1 - tail])
        rows.append({"comparison": name, "estimate": points[name], "ci_lower": low, "ci_upper": high,
                     "confidence_level": settings["confidence_level"], "valid_replicates": len(samples),
                     "unit": "patient_group_id", "paired": True, "refit_models": False,
                     "method": "percentile_bootstrap_conditional_on_fixed_OOF_predictions"})
    return pd.DataFrame(rows)


def make_figure(scores, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"clinical5": ("Clinical", "#246FA4"), "face": ("Facial", "#3A9275"),
              "clinical5_face": ("Fixed fusion", "#CD591E")}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#738C9B", "axes.labelcolor": "#294654",
                         "xtick.color": "#668292", "ytick.color": "#668292"}):
        fig, ax = plt.subplots(figsize=(7.2, 7.0))
        fig.subplots_adjust(left=.14, right=.96, bottom=.13, top=.87)
        ax.set_axisbelow(True)
        ax.grid(color="#E4EDF3", linewidth=.8)
        ax.plot([0, 1], [0, 1], linestyle=(0, (5, 5)), color="#A7BDCD", linewidth=1)
        for name, column in SCORES.items():
            label, color = styles[name]
            fpr, tpr, _ = roc_curve(scores.binary_label, scores[column])
            auc = roc_auc_score(scores.binary_label, scores[column])
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  OOF AUC={auc:.4f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 − Specificity", ylabel="Sensitivity")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(length=0, pad=8)
        ax.xaxis.labelpad = 14
        ax.yaxis.labelpad = 14
        ax.set_title("Clinical, facial and fixed fusion models\nn=330", loc="left", fontsize=15,
                     fontweight="bold", color="#294654", pad=18, linespacing=1.5)
        ax.legend(loc="lower right", frameon=False, fontsize=10, labelspacing=.9, labelcolor="#294654")
        fig.savefig(output / "clinical_facial_fixed_fusion_330_oof_roc.png", dpi=300, facecolor="white")
        fig.savefig(output / "clinical_facial_fixed_fusion_330_oof_roc.pdf", facecolor="white")
        plt.close(fig)


def write_report(output, cfg, data, scores, metrics, ci):
    intervals = ci.set_index("comparison")
    table = ["|模型|OOF AUC|条件性95% CI|Brier|准确率|敏感度|特异度|",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in metrics.to_dict("records"):
        v = intervals.loc[row["model"]]
        table.append(f"|{LABELS[row['model']]}|{row['roc_auc']:.4f}|{v.ci_lower:.4f}–{v.ci_upper:.4f}|"
                     f"{row['brier_score']:.4f}|{row['accuracy']:.2%}|{row['sensitivity']:.2%}|{row['specificity']:.2%}|")
    fold_rows = ["|fold|训练／留出|对照／患者|女／男|NYHA 0／1／2／3／4|",
                 "|---|---:|---:|---:|---|"]
    for fold, frame in data.groupby("fold"):
        counts = [int((frame.NYHA == n).sum()) for n in range(5)]
        fold_rows.append(f"|{fold}|{len(data)-len(frame)}／{len(frame)}|"
                         f"{int((frame.binary_label==0).sum())}／{int((frame.binary_label==1).sum())}|"
                         f"{int((frame.SEX==0).sum())}／{int((frame.SEX==1).sum())}|{'／'.join(map(str, counts))}|")
    delta = intervals.loc["clinical5_face_minus_clinical5"]
    text = f"""# 330名患者临床模型及固定面部融合实验

## 数据修正与五折

从原340条完整配对记录中排除10条较晚照片：其检验值复用了同一患者最早照片的检验记录，超出各自照片前7天窗口。
本次330条记录对应330名患者，BNP与hs-cTnT申请时间均位于照片前0—7×24小时（含边界）。没有对较晚照片重新查找检验，没有插补。
对照{int((data.binary_label==0).sum())}人、患者{int((data.binary_label==1).sum())}人；女性{int((data.SEX==0).sum())}人、男性{int((data.SEX==1).sum())}人。

数据表：`{cfg['data']['workbook']}`，工作表`Data 330`。保留照片时间、检验申请时间、间隔及原340／460／500来源折号。
新fold按binary_label×SEX分层、patient_group_id分组；沿用StratifiedGroupKFold与患者组层面平衡策略，种子2026。
该划分未使用临床模型或面部模型的预测分数。

{chr(10).join(fold_rows)}

## 模型与评价策略

临床模型输入为性别、年龄、BMI、BNP、hs-cTnT。无惩罚Logistic回归，含截距，lbfgs，max_iter=5000，tol=1e-10；无类别加权。
性别0／1原样输入，其他四项仅用当前训练折拟合StandardScaler；检验值按原始连续值线性入模，不取对数、不筛变量、不调参。
BNP单位pg/mL，hs-cTnT沿用来源表μg/L。每轮训练264人，对留出66人预测；汇总330个OOF预测评价，无内部调参验证集。

面部模型复用原500条全脸ResNet18的对应ID折外预测，不重新训练。逐条检查标签、性别、患者组及来源折号一致，并检查原面部外层留出记录未进入训练或选epoch。

```text
L_face = logit_patient − logit_control
L_fusion = 0.5 × L_clinical + 0.5 × L_face
p_fusion = sigmoid(L_fusion)
```

临床与面部使用各自来源的OOF分数，折号不同的记录为{int((scores.fold!=scores.face_fold).sum())}条。
固定权重0.5／0.5、截距0，不拟合融合参数，不将其描述为共享同一训练集的联合建模。
分类阈值0.5。全330人临床模型另行拟合保存，其预测不参与性能评价。

## 结果

{chr(10).join(table)}

固定融合相对临床模型的OOF AUC差值为{delta.estimate:+.4f}，条件性95% CI为{delta.ci_lower:+.4f}至{delta.ci_upper:+.4f}。
区间沿用原实验策略：2000次患者分组配对百分位bootstrap，种子2026；条件于已固定的OOF预测，不包含重训与重新分折的不确定性。

![330名患者三个模型的OOF ROC](clinical_facial_fixed_fusion_330_oof_roc.png)

图中仅显示OOF AUC，不显示95% CI；AUC不是五折AUC的算术平均。
原340条结果保留供追溯，其包含10条时间匹配不合格记录，不能作为满足本次时间窗口的最终结果。
原面部数据已识别的设备与标签关联仍可能影响融合评分，当前结果属于内部评价。
"""
    (output / "clinical_bnp_hsctnt_330_experiment_results.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    cfg, data, face = config_and_inputs(resolve_path(args.config))
    LOG.info("Inputs passed: 330 unique patients; both biomarkers within seven days; five folds of 66; face OOF aligned")
    if args.preflight:
        _, info, prediction = fit_fold(data, 0, "clinical5", cfg)
        evaluate(prediction.binary_label.to_numpy(int), prediction.L_clinical5_OOF.to_numpy(float), cfg)
        LOG.info("PREFLIGHT PASSED: fold0 train=%d test=%d iterations=%d; raw logits/probabilities verified; no experiment files written",
                 info["n_train_records"], len(prediction), info["solver_iterations"])
        return
    output = resolve_path(cfg["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(handler)
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
             "python_executable": sys.executable, "sklearn_version": sklearn.__version__,
             "records": 330, "patient_groups": 330, "completed_folds": [], "oof_records": 0,
             "face_model_refitted": False, "fusion_parameters_fitted": False}
    try:
        LOG.info("FULL RUN STARTED: clinical fivefold and fixed face fusion; output=%s", output)
        save_json(output / "run_info.json", state)
        (output / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        save_csv(output / "input_snapshot.csv", data)
        save_csv(output / "fold_summary.csv", fold_summary(data))
        save_csv(output / "cohort_summary.csv", cohort_summary(data))
        pieces, coefficients = [], []
        for fold in range(5):
            model, info, prediction = fit_fold(data, fold, "clinical5", cfg)
            directory = output / f"fold_{fold}"
            directory.mkdir()
            joblib.dump(model, directory / "model.joblib")
            save_json(directory / "coefficients_raw_units.json", info)
            save_csv(directory / "train_ids.csv", data.loc[data.fold != fold, ["ID", "patient_group_id", "fold", "binary_label"]])
            save_csv(directory / "test_predictions.csv", prediction)
            coefficients.extend(coefficient_rows(f"fold_{fold}", info))
            pieces.append(prediction)
            state["completed_folds"].append(fold)
            state["oof_records"] += len(prediction)
            save_json(output / "run_info.json", state)
            LOG.info("fold=%d train=264 test=66 iterations=%d clinical_OOF_AUC=%.4f", fold,
                     info["solver_iterations"], roc_auc_score(prediction.binary_label, prediction.L_clinical5_OOF))
        oof = pd.concat(pieces).sort_index()
        if not oof.index.equals(data.index) or not np.array_equal(oof.ID, data.ID):
            raise ValueError("Incomplete or misordered OOF predictions")
        scores = align_face(oof, face)
        scores["L_clinical5_face_OOF"] = .5 * scores.L_clinical5_OOF + .5 * scores.L_face_OOF
        scores["p_clinical5_face_OOF"] = expit(scores.L_clinical5_face_OOF)
        for name, column in SCORES.items():
            scores[f"predicted_{name}"] = (scores[column] >= 0).astype(int)
        save_csv(output / "oof_predictions_330.csv", scores)
        metric_rows, fold_rows = [], []
        for name, column in SCORES.items():
            values = evaluate(scores.binary_label.to_numpy(int), scores[column].to_numpy(float), cfg)
            metric_rows.append({"model": name, "records": 330, "patient_groups": 330, **values})
            LOG.info("Pooled model=%s OOF_AUC=%.6f Brier=%.6f", name, values["roc_auc"], values["brier_score"])
            for fold, subset in scores.groupby("fold"):
                fold_rows.append({"model": name, "fold": fold, "records": len(subset),
                    **evaluate(subset.binary_label.to_numpy(int), subset[column].to_numpy(float), cfg)})
        metrics = pd.DataFrame(metric_rows)
        save_csv(output / "metrics_comparison.csv", metrics)
        save_csv(output / "metrics_by_clinical_fold.csv", pd.DataFrame(fold_rows))
        ci = paired_bootstrap(scores, cfg["evaluation"])
        save_csv(output / "auc_paired_cluster_bootstrap_ci.csv", ci)
        model, info = fit_model(data, FEATURES, cfg["model"])
        directory = output / "final_model"
        directory.mkdir()
        joblib.dump(model, directory / "model.joblib")
        save_json(directory / "coefficients_raw_units.json", info)
        coefficients.extend(coefficient_rows("full_data_final", info))
        save_csv(output / "coefficients_raw_units.csv", pd.DataFrame(coefficients))
        make_figure(scores, output)
        write_report(output, cfg, data, scores, metrics, ci)
        state.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                     final_model_fitted=True, bootstrap_replicates=cfg["evaluation"]["bootstrap_replicates"],
                     matched_face_records=len(scores), different_source_fold_records=int((scores.fold != scores.face_fold).sum()))
        save_json(output / "run_info.json", state)
        LOG.info("COMPLETED: 330 clinical and fixed-fusion OOF predictions")
        print(f"EXPERIMENT_DIR={output}")
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=datetime.now().astimezone().isoformat())
        save_json(output / "run_info.json", state)
        LOG.exception("Run failed after %d folds; partial outputs retained at %s", len(state["completed_folds"]), output)
        raise


if __name__ == "__main__":
    main()
