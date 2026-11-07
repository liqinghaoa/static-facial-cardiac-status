from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import parsing_model as legacy_parser


CLASS_NAME_TO_ID = {name: idx for idx, name in legacy_parser.CELEBAMASK_HQ_CLASSES.items()}
CLASS_ID_TO_NAME = dict(legacy_parser.CELEBAMASK_HQ_CLASSES)
PARSING_INPUT_SIZE = int(legacy_parser.PARSING_INPUT_SIZE)
NUM_PARSING_CLASSES = int(legacy_parser.NUM_PARSING_CLASSES)


def class_ids(names: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    aliases = {"eye_glasses": "eyeglass", "glasses": "eyeglass"}
    ids: list[int] = []
    for name in names:
        key = aliases.get(name, name)
        if key not in CLASS_NAME_TO_ID:
            raise KeyError(f"unknown parser class name: {name}")
        ids.append(int(CLASS_NAME_TO_ID[key]))
    return tuple(ids)


def resolve_device(requested: str) -> Any:
    requested = requested.lower()
    if requested.startswith("cuda:"):
        if not legacy_parser.torch.cuda.is_available():
            raise RuntimeError(f"--device {requested} requested but CUDA is unavailable")
        return legacy_parser.torch.device(requested)
    return legacy_parser.resolve_parsing_device(requested)


def load_model(model_name: str, checkpoint_path: Path, device: Any) -> Any:
    return legacy_parser.load_face_parsing_model(model_name, checkpoint_path, device)


def run(aligned_srgb_uint8: np.ndarray, model: Any, device: Any) -> np.ndarray:
    return legacy_parser.run_face_parsing(aligned_srgb_uint8, model, device)


def colorize(label_map: np.ndarray) -> np.ndarray:
    return legacy_parser.colorize_parsing_label_map(label_map)


def semantic_area_ratios(label_map: np.ndarray) -> dict[str, float]:
    return legacy_parser.compute_semantic_area_ratios(label_map)
