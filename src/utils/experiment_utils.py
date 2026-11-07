"""Configuration, reproducibility and filesystem helpers."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(
    value: str | Path | None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path | None:
    """Resolve paths relative to the repository root unless already absolute."""
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).expanduser().resolve() / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_experiment_dir(
    output_root: str | Path, experiment_name: str
) -> Path:
    root = resolve_project_path(output_root)
    if root is None:
        raise ValueError("output_root must not be empty")
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / experiment_name
    if candidate.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / f"{experiment_name}_{timestamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def configure_logging(log_path: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
