"""Shared training primitives used by the current outer-fivefold experiment. No historical training entry point."""

from __future__ import annotations

import json

import platform

import subprocess

import sys

from datetime import datetime

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader

from datasets.binary_face_dataset import R3DPRBinaryFaceDataset

from metrics.binary_classification_metrics import compute_binary_metrics

from utils.experiment_utils import seed_worker

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def make_loader(dataset: R3DPRBinaryFaceDataset, batch_size: int, shuffle: bool, workers: int, seed: int, pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).to(torch.float32)
    if (counts == 0).any():
        raise ValueError(f"Cannot calculate weighted CE with missing class: counts={counts.tolist()}")
    return counts.sum() / (2.0 * counts)

def configure_trainability(model: torch.nn.Module, strategy: str, stage: str = "head_only") -> None:
    normalized = str(strategy).strip().lower()
    if normalized == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if normalized == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return
    if normalized == "staged_layer4":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if stage == "layer4_plus_fc":
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        elif stage != "head_only":
            raise ValueError(f"Unsupported staged_layer4 stage: {stage!r}")
        return
    raise ValueError(f"Unsupported R3DPR trainability_strategy: {strategy!r}")

def apply_training_mode(model: torch.nn.Module, strategy: str, batchnorm_mode: str = "train") -> None:
    model.train()
    freeze_batchnorm = (
        str(strategy).strip().lower() in {"head_only", "staged_layer4"}
        or str(batchnorm_mode).strip().lower() == "eval"
    )
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                module.eval()

def classifier_linear(model: torch.nn.Module) -> torch.nn.Linear:
    if isinstance(model.fc, torch.nn.Linear):
        return model.fc
    for module in reversed(list(model.fc.modules())):
        if isinstance(module, torch.nn.Linear):
            return module
    raise RuntimeError("R3DPR classifier head has no Linear layer")

def build_optimizer(model: torch.nn.Module, strategy: str, train_cfg: dict) -> torch.optim.AdamW:
    normalized = str(strategy).strip().lower()
    weight_decay = float(train_cfg["weight_decay"])
    if normalized == "staged_layer4":
        return torch.optim.AdamW(
            [
                {"params": list(model.fc.parameters()), "lr": float(train_cfg["head_warmup_lr"]), "group_name": "fc"},
                {"params": list(model.layer4.parameters()), "lr": float(train_cfg["layer4_lr"]), "group_name": "layer4"},
            ],
            weight_decay=weight_decay,
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("R3DPR training strategy left no trainable parameters")
    return torch.optim.AdamW(trainable_parameters, lr=float(train_cfg["lr"]), weight_decay=weight_decay)

def build_lr_scheduler(optimizer: torch.optim.AdamW, train_cfg: dict) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(train_cfg.get("lr_scheduler", "none")).strip().lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(train_cfg.get("lr_scheduler_t_max", train_cfg["epochs"]))
        eta_min = float(train_cfg.get("lr_scheduler_eta_min", 0.0))
        if t_max < 1:
            raise ValueError("lr_scheduler_t_max must be at least 1")
        if eta_min < 0.0:
            raise ValueError("lr_scheduler_eta_min must be non-negative")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    raise ValueError(f"Unsupported lr_scheduler: {name!r}")

def transition_to_layer4(model: torch.nn.Module, optimizer: torch.optim.AdamW, train_cfg: dict) -> None:
    configure_trainability(model, "staged_layer4", stage="layer4_plus_fc")
    for group in optimizer.param_groups:
        if group.get("group_name") == "fc":
            group["lr"] = float(train_cfg["fc_lr"])

def save_environment(output_dir: Path) -> None:
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": git.stdout.strip() or None,
    }
    (output_dir / "environment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    with_predictions: bool,
    selected_epoch: int | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[dict, float, pd.DataFrame | None]:
    model.eval()
    labels_all: list[int] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict] = []
    total_loss = 0.0
    sample_count = 0
    for batch in data_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = label.cpu().numpy().astype(int)
        logits_np = logits.cpu().numpy()
        labels_all.extend(labels.tolist())
        probabilities.append(probs)
        total_loss += float(loss) * len(labels)
        sample_count += len(labels)
        if with_predictions:
            prediction = probs.argmax(axis=1)
            for index, target in enumerate(labels):
                rows.append({
                    "sample_id": str(batch["sample_id"][index]),
                    "patient_group_id": str(batch["patient_group_id"][index]),
                    "sex": str(batch["sex"][index]),
                    "fold": int(batch["fold"][index]),
                    "binary_label": int(target),
                    "logit_control": float(logits_np[index, 0]),
                    "logit_patient": float(logits_np[index, 1]),
                    "prob_control": float(probs[index, 0]),
                    "prob_patient": float(probs[index, 1]),
                    "pred_class": int(prediction[index]),
                    "is_correct": int(prediction[index] == target),
                    "image_path": str(batch["image_path"][index]),
                    "selected_epoch": selected_epoch,
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                })
    return compute_binary_metrics(np.asarray(labels_all), np.concatenate(probabilities)), total_loss / sample_count, pd.DataFrame(rows) if with_predictions else None

def build_dataset(table: pd.DataFrame, config: dict, split: str, transform) -> R3DPRBinaryFaceDataset:
    data = config["data"]
    return R3DPRBinaryFaceDataset(
        table=table,
        image_root=project_path(data["image_root"]),
        image_filename_template=str(data["image_filename_template"]),
        transform=transform,
        id_column=data["id_column"],
        group_id_column=data["group_id_column"],
        fold_column=data["fold_column"],
        label_column=data["label_column"],
        sex_column=data["sex_column"],
    )
