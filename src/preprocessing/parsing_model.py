"""BiSeNet architecture, parsing inference and face-mask operations used by current preprocessing."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any, Iterable

import cv2

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from preprocessing import face_detection as alignment

PARSING_INPUT_SIZE = 512

NUM_PARSING_CLASSES = 19

CELEBAMASK_HQ_CLASSES = {
    0: "background",
    1: "skin",
    2: "left_brow",
    3: "right_brow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglass",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

CLASS_BACKGROUND = 0

CLASS_SKIN = 1

CLASS_LEFT_BROW = 2

CLASS_RIGHT_BROW = 3

CLASS_LEFT_EYE = 4

CLASS_RIGHT_EYE = 5

CLASS_LEFT_EAR = 7

CLASS_RIGHT_EAR = 8

CLASS_NOSE = 10

CLASS_MOUTH = 11

CLASS_UPPER_LIP = 12

CLASS_LOWER_LIP = 13

CLASS_NECK = 14

CLASS_CLOTH = 16

CLASS_HAIR = 17

CLASS_HAT = 18

SELECTED_SEMANTIC_CLASSES = (
    CLASS_SKIN,
    CLASS_LEFT_BROW,
    CLASS_RIGHT_BROW,
    CLASS_LEFT_EYE,
    CLASS_RIGHT_EYE,
    CLASS_NOSE,
    CLASS_MOUTH,
    CLASS_UPPER_LIP,
    CLASS_LOWER_LIP,
)

PARSING_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 170, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
    ],
    dtype=np.uint8,
)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return self.relu(shortcut + residual)

def create_layer_basic(
    in_chan: int,
    out_chan: int,
    block_count: int,
    stride: int = 1,
) -> nn.Sequential:
    layers: list[nn.Module] = [BasicBlock(in_chan, out_chan, stride=stride)]
    layers.extend(
        BasicBlock(out_chan, out_chan, stride=1)
        for _ in range(block_count - 1)
    )
    return nn.Sequential(*layers)

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, 2, stride=1)
        self.layer2 = create_layer_basic(64, 128, 2, stride=2)
        self.layer3 = create_layer_basic(128, 256, 2, stride=2)
        self.layer4 = create_layer_basic(256, 512, 2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return feat * atten

class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, padding=0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        avg = self.conv_avg(F.adaptive_avg_pool2d(feat32, 1))
        avg_up = F.interpolate(avg, feat32.shape[2:], mode="nearest")
        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(
            F.interpolate(feat32_sum, feat16.shape[2:], mode="nearest")
        )
        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(
            F.interpolate(feat16_sum, feat8.shape[2:], mode="nearest")
        )
        return feat8, feat16_up, feat32_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.convblk = ConvBNReLU(
            in_chan, out_chan, kernel_size=1, padding=0
        )
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatial: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        feat = self.convblk(torch.cat([spatial, context], dim=1))
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv2(self.relu(self.conv1(atten)))
        return feat * self.sigmoid(atten) + feat

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, class_count: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(
            mid_chan, class_count, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))

class BiSeNet(nn.Module):
    def __init__(self, n_classes: int = NUM_PARSING_CLASSES):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        output = F.interpolate(
            self.conv_out(feat_fuse),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        output16 = F.interpolate(
            self.conv_out16(feat_cp8),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        output32 = F.interpolate(
            self.conv_out32(feat_cp16),
            (height, width),
            mode="bilinear",
            align_corners=True,
        )
        return output, output16, output32

def resolve_parsing_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--parsing-device cuda requested but CUDA is unavailable")
    return torch.device(requested)

def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint does not contain a model state_dict")
    state_dict = {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("Checkpoint state_dict contains no tensors")
    for prefix in ("module.", "model.", "net."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items()
            }
    return state_dict

def load_face_parsing_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a strict 19-class BiSeNet checkpoint without network downloads."""
    if model_name.lower() != "bisenet":
        raise ValueError(f"Unsupported parsing model: {model_name}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Face parsing checkpoint does not exist: "
            f"{checkpoint_path}\nProvide --parsing-checkpoint with a compatible "
            "19-class CelebAMask-HQ BiSeNet .pth file. "
            "Third-party weights are not bundled; see README.md "
            "in the package root for acquisition and placement."
        )
    model = BiSeNet(n_classes=NUM_PARSING_CLASSES)
    try:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to load compatible BiSeNet checkpoint {checkpoint_path}: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model

def run_face_parsing(
    aligned_rgb: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Return a 224-space CelebAMask-HQ label map."""
    resized = cv2.resize(
        aligned_rgb,
        (PARSING_INPUT_SIZE, PARSING_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.ndim != 4 or logits.shape[1] != NUM_PARSING_CLASSES:
            raise RuntimeError(f"Unexpected parsing logits shape: {tuple(logits.shape)}")
        label_map = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    target_height, target_width = aligned_rgb.shape[:2]
    if label_map.shape != (target_height, target_width):
        label_map = cv2.resize(
            label_map,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    if int(label_map.max(initial=0)) >= NUM_PARSING_CLASSES:
        raise RuntimeError("Parsing label map contains an out-of-range class")
    return label_map

def colorize_parsing_label_map(label_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(label_map.astype(np.int32), 0, len(PARSING_COLORS) - 1)
    return PARSING_COLORS[clipped]

def build_selected_semantic_mask(label_map: np.ndarray) -> np.ndarray:
    """Build the semantic face evidence mask; this is not the final mask."""
    selected = np.isin(label_map, SELECTED_SEMANTIC_CLASSES).astype(np.uint8)
    minimum_pixels = max(64, int(round(label_map.size * 0.0025)))
    if int(selected.sum()) < minimum_pixels:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask",
            f"selected_semantic_mask_too_small:{int(selected.sum())}",
        )

    # Closing is used only to identify the principal face region. Returned
    # pixels remain genuine selected semantic classes, not the closed mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(selected * 255, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (closed > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "no_connected_selected_face_region"
        )
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary_region = (labels == largest_label).astype(np.uint8) * 255
    support = cv2.dilate(
        primary_region,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    primary_selected = ((selected > 0) & (support > 0)).astype(np.uint8)
    if int(primary_selected.sum()) < minimum_pixels:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "primary_selected_face_region_too_small"
        )
    return primary_selected

def _binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "cannot_compute_empty_mask_bbox"
        )
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def build_regularized_face_envelope_mask(
    selected_mask: np.ndarray,
    label_map: np.ndarray,
    forehead_expand_ratio: float,
    side_expand_ratio: float,
    chin_expand_ratio: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a continuous ellipse from the principal semantic face bbox.

    Ear, neck, cloth, hair, and background classes do not define the bbox.
    Hair pixels that happen to lie inside the resulting face envelope remain
    intact. Neck/cloth labels may conservatively cap the lower envelope without
    subtracting pixels or fragmenting the mask.
    """
    height, width = selected_mask.shape
    xmin, ymin, xmax, ymax = _binary_bbox(selected_mask)
    selected_width = max(1.0, float(xmax - xmin))
    selected_height = max(1.0, float(ymax - ymin))
    env_xmin = max(0.0, xmin - side_expand_ratio * selected_width)
    env_xmax = min(float(width - 1), xmax + side_expand_ratio * selected_width)
    env_ymin = max(0.0, ymin - forehead_expand_ratio * selected_height)
    env_ymax = min(float(height - 1), ymax + chin_expand_ratio * selected_height)

    # Use parsed ears only as side-boundary constraints. We do not subtract ear
    # pixels from the mask because that would create side holes; instead the
    # single ellipse is narrowed while never cutting inside the selected face
    # bbox itself.
    _, ear_x = np.where(
        np.isin(label_map, (CLASS_LEFT_EAR, CLASS_RIGHT_EAR))
    )
    selected_center_x = (xmin + xmax) / 2.0
    image_left_ear_x = ear_x[ear_x < selected_center_x]
    image_right_ear_x = ear_x[ear_x > selected_center_x]
    if len(image_left_ear_x):
        left_cap = float(int(image_left_ear_x.max()) + 1)
        env_xmin = min(float(xmin), max(env_xmin, left_cap))
    if len(image_right_ear_x):
        right_cap = float(int(image_right_ear_x.min()) - 1)
        env_xmax = max(float(xmax), min(env_xmax, right_cap))

    # Keep the envelope continuous while preventing its lower edge from
    # extending deeply into parsed neck or clothing.
    lower_classes = np.isin(label_map, (CLASS_NECK, CLASS_CLOTH))
    lower_y, _ = np.where(
        lower_classes
        & (np.indices(label_map.shape)[1] >= int(math.floor(env_xmin)))
        & (np.indices(label_map.shape)[1] <= int(math.ceil(env_xmax)))
        & (np.indices(label_map.shape)[0] >= ymin)
    )
    if len(lower_y):
        semantic_cap = float(int(lower_y.min()) - 1)
        env_ymax = max(float(ymax), min(env_ymax, semantic_cap))

    center_x = (env_xmin + env_xmax) / 2.0
    center_y = (env_ymin + env_ymax) / 2.0
    radius_x = max(1.0, (env_xmax - env_xmin) / 2.0)
    radius_y = max(1.0, (env_ymax - env_ymin) / 2.0)
    grid_y, grid_x = np.ogrid[:height, :width]
    ellipse = (
        ((grid_x - center_x) / radius_x) ** 2
        + ((grid_y - center_y) / radius_y) ** 2
        <= 1.0
    )
    envelope = ellipse.astype(np.uint8) * 255
    if int((envelope > 0).sum()) == 0:
        raise alignment.SampleFailure(
            "failed_empty_selected_mask", "regularized_envelope_is_empty"
        )
    bbox = {
        "envelope_bbox_xmin": int(math.floor(env_xmin)),
        "envelope_bbox_ymin": int(math.floor(env_ymin)),
        "envelope_bbox_xmax": int(math.ceil(env_xmax)),
        "envelope_bbox_ymax": int(math.ceil(env_ymax)),
    }
    return envelope, bbox

def compute_semantic_area_ratios(label_map: np.ndarray) -> dict[str, float]:
    denominator = float(label_map.size)

    def ratio(classes: Iterable[int]) -> float:
        return float(np.isin(label_map, tuple(classes)).sum()) / denominator

    return {
        "skin_area_ratio": ratio((CLASS_SKIN,)),
        "brow_area_ratio": ratio((CLASS_LEFT_BROW, CLASS_RIGHT_BROW)),
        "eye_area_ratio": ratio((CLASS_LEFT_EYE, CLASS_RIGHT_EYE)),
        "nose_area_ratio": ratio((CLASS_NOSE,)),
        "mouth_area_ratio": ratio((CLASS_MOUTH,)),
        "lip_area_ratio": ratio((CLASS_UPPER_LIP, CLASS_LOWER_LIP)),
        "hair_area_ratio": ratio((CLASS_HAIR,)),
        "ear_area_ratio": ratio((CLASS_LEFT_EAR, CLASS_RIGHT_EAR)),
        "neck_area_ratio": ratio((CLASS_NECK,)),
        "cloth_area_ratio": ratio((CLASS_CLOTH,)),
        "background_area_ratio": ratio((CLASS_BACKGROUND,)),
    }
