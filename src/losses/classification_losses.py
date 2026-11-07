"""Classification loss builders."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights(
    labels: Iterable[int], num_classes: int = 3
) -> torch.Tensor:
    """Compute fold-specific balanced weights: N / (C * class_count)."""
    label_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    if label_tensor.numel() == 0:
        raise ValueError("Cannot compute class weights from an empty label sequence")
    counts = torch.bincount(label_tensor, minlength=num_classes).to(torch.float32)
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            f"Training labels are missing classes {missing}; weighted CE is undefined"
        )
    return label_tensor.numel() / (num_classes * counts)


class WeightedSoftCrossEntropyWithLabelSmoothing(nn.Module):
    """Weighted soft-label cross entropy with exclude-true-class smoothing."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        smoothing: float = 0.1,
        num_classes: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.num_classes = int(num_classes)
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        if not (0.0 <= self.smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {self.smoothing}")
        if self.num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {self.num_classes}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss from raw logits [B, C] and integer targets [B]."""
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B, C], got {logits.shape}")
        if logits.size(1) != self.num_classes:
            raise ValueError(
                f"logits class dimension {logits.size(1)} != num_classes "
                f"{self.num_classes}"
            )

        target = target.long()
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            smooth_value = self.smoothing / (self.num_classes - 1)
            true_value = 1.0 - self.smoothing
            soft_target = torch.full_like(log_probs, smooth_value)
            soft_target.scatter_(1, target.unsqueeze(1), true_value)

        loss = -soft_target * log_probs
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device).unsqueeze(0)
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


def build_criterion(
    loss_name: str = "weighted_cross_entropy",
    class_weights: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    smoothing: float = 0.1,
    num_classes: int = 3,
    reduction: str = "mean",
) -> nn.Module:
    """Build a cross-entropy criterion that consumes raw logits."""
    normalized_name = loss_name.lower()
    if normalized_name not in {
        "weighted_cross_entropy",
        "cross_entropy",
        "weighted_soft_cross_entropy",
        "weighted_ce_label_smoothing",
    }:
        raise ValueError(f"Unsupported classification loss: {loss_name!r}")
    weighted_losses = {
        "weighted_cross_entropy",
        "weighted_soft_cross_entropy",
        "weighted_ce_label_smoothing",
    }
    if normalized_name in weighted_losses and class_weights is None:
        raise ValueError(f"class_weights are required for {normalized_name}")
    weights = class_weights
    if weights is not None and device is not None:
        weights = weights.to(device)
    if normalized_name in {
        "weighted_soft_cross_entropy",
        "weighted_ce_label_smoothing",
    }:
        return WeightedSoftCrossEntropyWithLabelSmoothing(
            class_weights=weights,
            smoothing=smoothing,
            num_classes=num_classes,
            reduction=reduction,
        )
    return nn.CrossEntropyLoss(
        weight=weights if normalized_name == "weighted_cross_entropy" else None
    )
