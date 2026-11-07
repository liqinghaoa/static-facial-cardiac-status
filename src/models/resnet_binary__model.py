"""ImageNet-pretrained ResNet binary classifiers for R3DPR."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


SUPPORTED_R3DPR_RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")


def _pretrained_enabled(pretrained: bool | str) -> bool:
    enabled = str(pretrained).strip().lower() in {"imagenet", "default", "true", "1", "yes"}
    disabled = str(pretrained).strip().lower() in {"none", "false", "0", "no", "random"}
    if not enabled and not disabled:
        raise ValueError(f"Unsupported pretrained setting: {pretrained!r}")
    return enabled


def build_r3dpr_resnet_binary(
    backbone: str,
    pretrained: bool | str = "imagenet",
    dropout: float | None = None,
) -> nn.Module:
    """Build an R3DPR ResNet with logits ordered as [Control, Patient]."""
    normalized = str(backbone).strip().lower()
    if normalized not in SUPPORTED_R3DPR_RESNET_BACKBONES:
        raise ValueError(
            f"Unsupported R3DPR backbone {backbone!r}; choose from "
            f"{list(SUPPORTED_R3DPR_RESNET_BACKBONES)}"
        )

    constructors = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
    }
    constructor, weights_name = constructors[normalized]
    weights_enum = getattr(models, weights_name, None)
    if weights_enum is None:  # pragma: no cover - compatibility with old torchvision.
        model = constructor(pretrained=_pretrained_enabled(pretrained))
    else:
        weights = weights_enum.IMAGENET1K_V1 if _pretrained_enabled(pretrained) else None
        model = constructor(weights=weights)

    linear = nn.Linear(model.fc.in_features, 2)
    if dropout is None:
        model.fc = linear
    else:
        probability = float(dropout)
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        model.fc = nn.Sequential(nn.Dropout(p=probability), linear)
    return model


def build_r3dpr_resnet18_binary(
    pretrained: bool | str = "imagenet",
    dropout: float | None = None,
) -> nn.Module:
    """Build the backward-compatible ResNet18 R3DPR classifier."""
    return build_r3dpr_resnet_binary("resnet18", pretrained=pretrained, dropout=dropout)


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_params": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
