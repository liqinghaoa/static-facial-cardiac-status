"""Dataset and transforms for the R3DPR Control-versus-Patient task."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_r3dpr_transforms(
    split: str,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
    horizontal_flip: bool,
) -> transforms.Compose:
    """Build deterministic validation and lightly augmented training transforms."""
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    steps: list[object] = [transforms.Resize((int(image_height), int(image_width)))]
    if split == "train" and horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


class R3DPRBinaryFaceDataset(Dataset):
    """Read direct binary labels from the R3DPR single-table five-fold CSV."""

    def __init__(
        self,
        table: pd.DataFrame,
        image_root: str | Path,
        image_filename_template: str,
        transform: transforms.Compose,
        id_column: str,
        group_id_column: str,
        fold_column: str,
        label_column: str,
        sex_column: str,
    ) -> None:
        required = {id_column, group_id_column, fold_column, label_column, sex_column}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"R3DPR table lacks required columns: {sorted(missing)}")
        self.table = table.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.image_filename_template = image_filename_template
        self.transform = transform
        self.id_column = id_column
        self.group_id_column = group_id_column
        self.fold_column = fold_column
        self.label_column = label_column
        self.sex_column = sex_column

        labels = pd.to_numeric(self.table[label_column], errors="raise").astype(int)
        if not labels.isin([0, 1]).all():
            raise ValueError("R3DPR binary labels must be directly encoded as 0 or 1")
        self.table[label_column] = labels
        self.labels = labels.tolist()

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        sample_id = str(row[self.id_column])
        image_path = self.image_root / self.image_filename_template.format(ID=sample_id)
        if not image_path.is_file():
            raise FileNotFoundError(f"R3DPR image does not exist: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "sample_id": sample_id,
            "patient_group_id": str(row[self.group_id_column]),
            "sex": str(row[self.sex_column]),
            "fold": torch.tensor(int(row[self.fold_column]), dtype=torch.long),
            "image_path": str(image_path),
        }
