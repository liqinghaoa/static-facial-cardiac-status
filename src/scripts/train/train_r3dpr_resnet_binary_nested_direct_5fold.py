"""Train an outer-five-fold R3DPR binary experiment with direct inner-checkpoint testing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import sklearn  # noqa: F401  # Initialize OpenMP before torch on Windows.
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.binary_face_dataset import build_r3dpr_transforms
from losses.classification_losses import build_criterion
from metrics.binary_classification_metrics import flatten_metrics
from models.resnet_binary__model import SUPPORTED_R3DPR_RESNET_BACKBONES, build_r3dpr_resnet_binary, count_parameters
from utils.face_training import apply_training_mode, build_dataset, build_lr_scheduler, build_optimizer, classifier_linear, compute_class_weights, configure_trainability, evaluate, make_loader, save_environment, transition_to_layer4
from utils.experiment_utils import configure_logging, load_yaml, save_yaml, set_random_seed


LOGGER = logging.getLogger("r3dpr_binary_nested_direct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=None, help="Only for isolated smoke runs")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_config(config: dict) -> None:
    if config["task"]["type"] != "binary_control_vs_patient" or int(config["task"]["num_classes"]) != 2:
        raise ValueError("R3DPR nested-direct requires the fixed Control-versus-Patient binary task")
    backbone = str(config["model"]["backbone"]).strip().lower()
    if backbone not in SUPPORTED_R3DPR_RESNET_BACKBONES or int(config["model"]["num_classes"]) != 2:
        raise ValueError("R3DPR nested-direct requires resnet18, resnet34, or resnet50 with two output logits")
    strategy = str(config["model"].get("trainability_strategy", "")).strip().lower()
    if strategy not in {"full_finetune", "head_only", "staged_layer4"}:
        raise ValueError("Unsupported trainability strategy")
    if strategy == "staged_layer4":
        if int(config["train"].get("head_warmup_epochs", 0)) < 1:
            raise ValueError("staged_layer4 requires train.head_warmup_epochs >= 1")
        for key in ("head_warmup_lr", "fc_lr", "layer4_lr"):
            if float(config["train"].get(key, 0.0)) <= 0.0:
                raise ValueError(f"staged_layer4 requires a positive train.{key}")
    if config["train"]["optimizer"].lower() != "adamw" or config["train"]["monitor_metric"] != "macro_auc":
        raise ValueError("R3DPR nested-direct requires AdamW and inner-validation macro-AUC selection")
    if str(config["train"].get("lr_scheduler", "none")).strip().lower() not in {"none", "cosine"}:
        raise ValueError("train.lr_scheduler must be none or cosine")
    loss_name = str(config["train"].get("loss", "weighted_cross_entropy")).strip().lower()
    alpha = float(config["train"].get("label_smoothing_alpha", 0.0))
    if loss_name == "weighted_cross_entropy" and alpha != 0.0:
        raise ValueError("weighted_cross_entropy requires label_smoothing_alpha=0")
    if loss_name == "weighted_ce_label_smoothing" and not (0.0 < alpha < 1.0):
        raise ValueError("weighted_ce_label_smoothing requires 0 < label_smoothing_alpha < 1")
    if loss_name not in {"weighted_cross_entropy", "weighted_ce_label_smoothing"}:
        raise ValueError("Unsupported loss")
    nested = config.get("nested_direct", {})
    if nested.get("protocol") != "outer_5fold_inner_holdout_direct_test":
        raise ValueError("nested_direct.protocol must be outer_5fold_inner_holdout_direct_test")
    if bool(nested.get("use_group_constraint", True)):
        raise ValueError("This nested-direct protocol must set nested_direct.use_group_constraint=false")
    if list(nested.get("stratify_columns", [])) != [config["data"]["label_column"], config["data"]["sex_column"]]:
        raise ValueError("nested_direct.stratify_columns must be [label_column, sex_column]")
    fraction = float(nested.get("inner_validation_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("nested_direct.inner_validation_fraction must be in (0, 1)")


def make_strata(table: pd.DataFrame, label_column: str, sex_column: str) -> pd.Series:
    return "label_" + table[label_column].astype(str) + "__sex_" + table[sex_column].astype(str)


def audit_nested_direct_data(config: dict, output_dir: Path) -> pd.DataFrame:
    """Audit ID-level outer folds without applying a patient-group split constraint."""
    data = config["data"]
    table_path = project_path(data["table_csv"])
    image_root = project_path(data["image_root"])
    id_column = data["id_column"]
    group_id_column = data["group_id_column"]
    fold_column = data["fold_column"]
    label_column = data["label_column"]
    sex_column = data["sex_column"]
    table = pd.read_csv(table_path, dtype={id_column: "string", group_id_column: "string"})
    required = {id_column, group_id_column, fold_column, label_column, sex_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Nested-direct CSV lacks required columns: {missing}")
    table[id_column] = table[id_column].astype("string").str.strip()
    table[group_id_column] = table[group_id_column].astype("string").str.strip()
    try:
        table[fold_column] = pd.to_numeric(table[fold_column], errors="raise").astype(int)
        table[label_column] = pd.to_numeric(table[label_column], errors="raise").astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("Nested-direct fold and label columns must be integer-like") from error
    errors: list[str] = []
    if table.empty:
        errors.append("CSV contains no samples")
    if table[id_column].isna().any() or (table[id_column] == "").any():
        errors.append("ID contains empty values")
    if table[id_column].duplicated().any():
        errors.append(f"ID is not unique: {int(table[id_column].duplicated().sum())} duplicate rows")
    n_folds = int(data["n_folds"])
    if set(table[fold_column].unique()) != set(range(n_folds)):
        errors.append(f"fold values must be exactly 0..{n_folds - 1}")
    if not table[label_column].isin([0, 1]).all():
        errors.append("binary label column must contain only 0 and 1")
    image_paths = table[id_column].map(lambda sample_id: image_root / str(data["image_filename_template"]).format(ID=sample_id))
    missing_images = table.loc[~image_paths.map(Path.is_file), [id_column, fold_column, label_column]].copy()
    if not missing_images.empty:
        errors.append(f"{len(missing_images)} samples have no image")
    mismatches: list[dict[str, object]] = []
    for sample_id, image_path in zip(table[id_column], image_paths):
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            if image.size != (int(data["source_image_width"]), int(data["source_image_height"])):
                mismatches.append({"sample_id": sample_id, "path": str(image_path), "width": image.width, "height": image.height})
    if mismatches:
        errors.append(f"{len(mismatches)} images have an unexpected source size")
    fold_rows = []
    for fold in range(n_folds):
        development = table.loc[table[fold_column] != fold]
        outer_test = table.loc[table[fold_column] == fold]
        if set(development[label_column].unique()) != {0, 1} or set(outer_test[label_column].unique()) != {0, 1}:
            errors.append(f"fold {fold} development or outer test lacks a class")
        fold_rows.append(
            {
                "fold": fold,
                "development_n": len(development),
                "development_control": int((development[label_column] == 0).sum()),
                "development_patient": int((development[label_column] == 1).sum()),
                "outer_test_n": len(outer_test),
                "outer_test_control": int((outer_test[label_column] == 0).sum()),
                "outer_test_patient": int((outer_test[label_column] == 1).sum()),
                "outer_test_sex_distribution": outer_test[sex_column].value_counts().sort_index().to_dict(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_data_distribution.csv", index=False, encoding="utf-8-sig")
    if not missing_images.empty:
        missing_images.to_csv(output_dir / "missing_images.csv", index=False, encoding="utf-8-sig")
    if mismatches:
        pd.DataFrame(mismatches).to_csv(output_dir / "image_size_mismatches.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Nested-Direct ID-Level Data Audit",
        "",
        f"- CSV: `{table_path}`",
        f"- Image root: `{image_root}`",
        f"- Samples: {len(table)}",
        f"- Unique IDs: {table[id_column].nunique()}",
        f"- Label column: `{label_column}`",
        "- Split constraint: ID-level only; patient_group_id is intentionally not used.",
        f"- Missing images: {len(missing_images)}",
        f"- Image size mismatches: {len(mismatches)}",
        "",
        "## Status",
        "",
        "PASSED" if not errors else "FAILED",
        *([f"- {error}" for error in errors] if errors else ["- All IDs, labels, outer folds, and image dimensions passed."]),
    ]
    (output_dir / "data_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("Nested-direct data audit failed: " + "; ".join(errors))
    return table


def make_inner_split(config: dict, development: pd.DataFrame, outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = config["data"]
    nested = config["nested_direct"]
    target_size = int(round(len(development) * float(nested["inner_validation_fraction"])))
    strata = make_strata(development, data["label_column"], data["sex_column"])
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=target_size,
        random_state=int(nested["inner_split_seed"]) + int(outer_fold),
    )
    train_indices, validation_indices = next(splitter.split(development, strata))
    inner_train = development.iloc[train_indices].copy().reset_index(drop=True)
    inner_validation = development.iloc[validation_indices].copy().reset_index(drop=True)
    if len(inner_validation) != target_size or len(inner_train) + len(inner_validation) != len(development):
        raise RuntimeError("Inner split sizes are invalid")
    train_ids = set(inner_train[data["id_column"]].astype(str))
    validation_ids = set(inner_validation[data["id_column"]].astype(str))
    if train_ids.intersection(validation_ids):
        raise RuntimeError("Internal train and validation IDs overlap")
    return inner_train, inner_validation


def make_transforms(config: dict):
    data = config["data"]
    train_transform = build_r3dpr_transforms(
        "train",
        int(data["image_height"]),
        int(data["image_width"]),
        config["normalize"]["mean"],
        config["normalize"]["std"],
        bool(config["augmentation"]["horizontal_flip"]),
    )
    evaluation_transform = build_r3dpr_transforms(
        "val",
        int(data["image_height"]),
        int(data["image_width"]),
        config["normalize"]["mean"],
        config["normalize"]["std"],
        False,
    )
    return train_transform, evaluation_transform


def build_training_state(
    config: dict,
    train_table: pd.DataFrame,
    device: torch.device,
    initialization_seed: int,
) -> tuple[torch.nn.Module, torch.optim.AdamW, object, torch.nn.Module, dict]:
    set_random_seed(initialization_seed)
    train_cfg = config["train"]
    strategy = str(config["model"]["trainability_strategy"]).strip().lower()
    backbone = str(config["model"]["backbone"]).strip().lower()
    model = build_r3dpr_resnet_binary(
        backbone,
        config["model"]["pretrained"],
        dropout=config["model"].get("dropout"),
    ).to(device)
    configure_trainability(model, strategy)
    final_linear = classifier_linear(model)
    if final_linear.out_features != 2:
        raise RuntimeError("R3DPR nested-direct must use a two-logit classifier head")
    weights = compute_class_weights(train_table[config["data"]["label_column"]].astype(int).tolist()).to(device)
    criterion = build_criterion(
        str(train_cfg["loss"]).strip().lower(),
        class_weights=weights,
        device=device,
        smoothing=float(train_cfg.get("label_smoothing_alpha", 0.0)),
        num_classes=int(config["task"]["num_classes"]),
    )
    optimizer = build_optimizer(model, strategy, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg)
    counts = np.bincount(train_table[config["data"]["label_column"]].astype(int).to_numpy(), minlength=2)
    info = {
        "backbone": backbone,
        "classifier_in_features": int(final_linear.in_features),
        "trainability_strategy": strategy,
        "batchnorm_mode": str(train_cfg.get("batchnorm_mode", "train")).strip().lower(),
        "loss_name": str(train_cfg["loss"]).strip().lower(),
        "label_smoothing_alpha": float(train_cfg.get("label_smoothing_alpha", 0.0)),
        "initialization_seed": int(initialization_seed),
        "train_control_count": int(counts[0]),
        "train_patient_count": int(counts[1]),
        "weight_control": float(weights[0]),
        "weight_patient": float(weights[1]),
        **count_parameters(model),
    }
    return model, optimizer, scheduler, criterion, info


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.AdamW,
    criterion: torch.nn.Module,
    strategy: str,
    batchnorm_mode: str,
) -> tuple[float, int]:
    apply_training_mode(model, strategy, batchnorm_mode)
    total_loss, seen = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(label)
        seen += len(label)
    if seen == 0:
        raise RuntimeError("Training loader produced zero samples")
    return total_loss / seen, seen


def maybe_transition_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    train_cfg: dict,
    strategy: str,
    epoch: int,
) -> str:
    if strategy != "staged_layer4":
        return "full_finetune" if strategy == "full_finetune" else "head_only"
    if epoch == int(train_cfg["head_warmup_epochs"]) + 1:
        transition_to_layer4(model, optimizer, train_cfg)
    return "layer4_plus_fc" if epoch > int(train_cfg["head_warmup_epochs"]) else "head_only"


def save_split_manifest(
    config: dict,
    fold_dir: Path,
    outer_fold: int,
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    outer_test: pd.DataFrame,
) -> None:
    data = config["data"]
    columns = [data["id_column"], data["group_id_column"], data["label_column"], data["sex_column"], data["fold_column"]]
    frames = []
    for role, table in (
        ("inner_train", inner_train),
        ("inner_validation", inner_validation),
        ("outer_test", outer_test),
    ):
        frame = table.loc[:, columns].copy()
        frame.insert(0, "role", role)
        frame.insert(0, "outer_fold", outer_fold)
        frame["stratum"] = make_strata(frame, data["label_column"], data["sex_column"])
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    manifest.to_csv(fold_dir / "inner_split.csv", index=False, encoding="utf-8-sig")


def run_fold(config: dict, table: pd.DataFrame, output_dir: Path, fold: int, device: torch.device) -> None:
    data, train_cfg, nested = config["data"], config["train"], config["nested_direct"]
    fold_column = data["fold_column"]
    label_column = data["label_column"]
    development = table.loc[table[fold_column] != fold].copy().reset_index(drop=True)
    outer_test = table.loc[table[fold_column] == fold].copy().reset_index(drop=True)
    inner_train, inner_validation = make_inner_split(config, development, fold)
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(config, fold_dir, fold, inner_train, inner_validation, outer_test)
    train_transform, evaluation_transform = make_transforms(config)
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["num_workers"])
    pin_memory = bool(train_cfg["pin_memory"])
    base_seed = int(train_cfg["random_seed"])
    initialization_seed = base_seed + 10000 + int(fold)
    strategy = str(config["model"]["trainability_strategy"]).strip().lower()
    batchnorm_mode = str(train_cfg.get("batchnorm_mode", "train")).strip().lower()
    started = time.perf_counter()

    selection_train_set = build_dataset(inner_train, config, "train", train_transform)
    selection_train_eval_set = build_dataset(inner_train, config, "train", evaluation_transform)
    selection_val_set = build_dataset(inner_validation, config, "val", evaluation_transform)
    selection_train_loader = make_loader(selection_train_set, batch_size, True, workers, base_seed + fold, pin_memory)
    selection_train_eval_loader = make_loader(selection_train_eval_set, batch_size, False, workers, base_seed + 100 + fold, pin_memory)
    selection_val_loader = make_loader(selection_val_set, batch_size, False, workers, base_seed + 1000 + fold, pin_memory)
    model, optimizer, scheduler, criterion, selection_info = build_training_state(
        config, inner_train, device, initialization_seed
    )
    LOGGER.info(
        "fold=%d inner split train=%d validation=%d outer_test=%d counts=%s",
        fold, len(inner_train), len(inner_validation), len(outer_test), selection_info["train_control_count"],
    )
    best_auc, selected_epoch, patience = float("-inf"), 0, 0
    selection_history: list[dict] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        phase = maybe_transition_stage(model, optimizer, train_cfg, strategy, epoch)
        train_loss, _ = train_epoch(model, selection_train_loader, optimizer, criterion, strategy, batchnorm_mode)
        train_metrics, _, _ = evaluate(model, selection_train_eval_loader, device, criterion, False)
        validation_metrics, validation_loss, _ = evaluate(model, selection_val_loader, device, criterion, False)
        validation_auc = float(validation_metrics["macro_auc"])
        row = {
            "epoch": epoch,
            "phase": phase,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            **{f"train_{key}": value for key, value in flatten_metrics(train_metrics).items()},
            "inner_val_loss": validation_loss,
            **{f"inner_val_{key}": value for key, value in flatten_metrics(validation_metrics).items()},
        }
        selection_history.append(row)
        if validation_auc > best_auc:
            best_auc, selected_epoch, patience = validation_auc, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "inner_best_macro_auc": validation_auc,
                    "config": config,
                    "selection_info": selection_info,
                },
                checkpoint_dir / "inner_best_macro_auc.pth",
            )
        else:
            patience += 1
        LOGGER.info(
            "fold=%d selection epoch=%d train_loss=%.5f inner_val_loss=%.5f inner_val_macro_auc=%.4f best=%.4f patience=%d/%d",
            fold, epoch, train_loss, validation_loss, validation_auc, best_auc, patience, int(train_cfg["early_stopping_patience"]),
        )
        if scheduler is not None:
            scheduler.step()
        if patience >= int(train_cfg["early_stopping_patience"]):
            break
    pd.DataFrame(selection_history).to_csv(fold_dir / "inner_selection_history.csv", index=False, encoding="utf-8-sig")
    selection_seconds = time.perf_counter() - started
    selected_payload = {
        "outer_fold": fold,
        "selected_epoch": selected_epoch,
        "inner_best_macro_auc": best_auc,
        "selection_epochs_run": len(selection_history),
        "inner_train_size": len(inner_train),
        "inner_validation_size": len(inner_validation),
        "selection_checkpoint": str(checkpoint_dir / "inner_best_macro_auc.pth"),
    }
    (fold_dir / "selected_epoch.json").write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Restore the selected internal-validation checkpoint. It is directly evaluated on the untouched outer fold.
    selection_checkpoint = checkpoint_dir / "inner_best_macro_auc.pth"
    checkpoint = torch.load(selection_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The outer held-out fold is instantiated only after internal model selection has finished.
    outer_test_set = build_dataset(outer_test, config, "outer_test", evaluation_transform)
    outer_test_loader = make_loader(outer_test_set, batch_size, False, workers, base_seed + 3000 + fold, pin_memory)
    outer_metrics, _, predictions = evaluate(
        model, outer_test_loader, device, criterion, True, selected_epoch, selection_checkpoint
    )
    assert predictions is not None
    predictions["selection_best_macro_auc"] = best_auc
    predictions["selection_epochs_run"] = len(selection_history)
    predictions["protocol"] = nested["protocol"]
    predictions.to_csv(fold_dir / "outer_test_predictions.csv", index=False, encoding="utf-8-sig")
    fold_info = {
        "fold": fold,
        "protocol": nested["protocol"],
        "outer_development_size": len(development),
        "outer_test_size": len(outer_test),
        "inner_train_size": len(inner_train),
        "inner_validation_size": len(inner_validation),
        "selected_epoch": selected_epoch,
        "inner_best_macro_auc": best_auc,
        "selection_epochs_run": len(selection_history),
        "selection_training_seconds": selection_seconds,
        "outer_test_checkpoint": str(selection_checkpoint),
        "selection": selection_info,
        **flatten_metrics(outer_metrics),
    }
    pd.DataFrame([fold_info]).to_json(fold_dir / "fold_protocol.json", orient="records", indent=2, force_ascii=False)
    metric_row = {key: value for key, value in fold_info.items() if key != "selection"}
    pd.DataFrame([metric_row]).to_csv(fold_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        outer_metrics["confusion_matrix"], index=["control", "patient"], columns=["control", "patient"]
    ).to_csv(fold_dir / "outer_test_confusion_matrix.csv", index_label="true\\pred", encoding="utf-8-sig")
    LOGGER.info(
        "fold=%d complete selected_epoch=%d inner_best_auc=%.4f outer_test_macro_auc=%.4f",
        fold, selected_epoch, best_auc, float(outer_metrics["macro_auc"]),
    )
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()


def main() -> Path:
    args = parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R3DPR nested-direct baseline; CPU fallback is disabled")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "experiment.log")
    save_yaml(config, output_dir / "config_snapshot.yaml")
    save_environment(output_dir)
    table = audit_nested_direct_data(config, output_dir)
    folds = args.folds if args.folds is not None else list(range(int(config["data"]["n_folds"])))
    if set(folds).difference(range(int(config["data"]["n_folds"]))):
        raise ValueError(f"Invalid requested folds: {folds}")
    device = torch.device("cuda")
    for fold in folds:
        run_fold(config, table, output_dir, fold, device)
    (output_dir / "run_finished_at.txt").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    print(f"EXPERIMENT_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
