"""Shared image loading and patient-group validation split for Task 2."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ("sample_id", "path", "label", "patient_group")
GEOMETRY_POLICIES = {
    "flip_rotate": "horizontal flip p=0.5; rotation uniform [-7,7] deg expand=True before letterbox",
    "flip_only": "horizontal flip p=0.5 before letterbox",
    "rotate_only": "rotation uniform [-7,7] deg expand=True before letterbox",
}


def load_image_tensor(
    image_path: Path, image_size: int = 224, augment: bool = False,
    geometry_mode: str = "flip_rotate",
) -> torch.Tensor:
    """Apply identical preprocessing during training and label-free inference."""
    flip = augment and torch.rand(()).item() < 0.5
    angle = torch.empty(()).uniform_(-7.0, 7.0).item() if augment else 0.0
    if os.environ.get("TASK2_PIL_PYTHON"):
        from local_cpu import pixels as local_pixels
        values = local_pixels(image_path, image_size, augment, flip and geometry_mode != "rotate_only",
                              0.0 if geometry_mode == "flip_only" else angle)
        return torch.from_numpy((values - 0.5) / 0.5).permute(2, 0, 1).contiguous()
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    from PIL import Image, ImageOps
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if augment:
            # Resize first for inexpensive, single-image training augmentation.
            # expand=True preserves all image corners before the final letterbox.
            image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
            # Match the legacy RNG sequence across all three policies.
            if flip and geometry_mode != "rotate_only":
                image = ImageOps.mirror(image)
            if geometry_mode != "flip_only":
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR,
                                  expand=True, fill=0)
        image = ImageOps.pad(
            image, (image_size, image_size),
            method=Image.Resampling.BILINEAR, color=(0, 0, 0),
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1).contiguous()


def load_records(data_root: Path) -> pd.DataFrame:
    """Read labels.csv and validate IDs, binary labels, and image locations."""
    data_root = Path(data_root).resolve()
    records = pd.read_csv(data_root / "labels.csv", dtype=str)
    missing = set(REQUIRED_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("labels.csv contains no samples")
    for column in REQUIRED_COLUMNS:
        if records[column].isna().any() or records[column].str.strip().eq("").any():
            raise ValueError(f"Column {column!r} contains missing or empty values")
    for column in ("sample_id", "path"):
        if records[column].duplicated().any():
            raise ValueError(f"Column {column!r} must contain unique values")
    if not set(records["label"]).issubset({"0", "1"}):
        raise ValueError("Labels must be exactly 0 or 1")
    records["label"] = records["label"].astype(int)
    if set(records["label"]) != {0, 1}:
        raise ValueError("Both labels 0 and 1 must be present")

    resolved_paths = []
    for relative_path in records["path"]:
        image_path = (data_root / relative_path).resolve()
        if Path(relative_path).is_absolute() or not image_path.is_relative_to(data_root):
            raise ValueError(f"Image path must stay inside data_root: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        resolved_paths.append(image_path)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Multiple records refer to the same image file")
    return records.sort_values("sample_id", kind="stable").reset_index(drop=True)


def make_splits(
    records: pd.DataFrame, n_splits: int = 5, val_fold: int = 0
) -> pd.DataFrame:
    """Keep one group-stratified fold for validation, all others for training.

    The private TA test set is external and is never simulated by holding out
    another part of the supplied training data. Sorting and shuffle=False make
    the split deterministic without choosing a seed to optimize validation.
    """
    if n_splits < 2 or not 0 <= val_fold < n_splits:
        raise ValueError("Require n_splits >= 2 and 0 <= val_fold < n_splits")
    records = records.sort_values("sample_id", kind="stable").reset_index(drop=True).copy()
    if records["patient_group"].nunique() < n_splits:
        raise ValueError("Not enough patient groups for the requested number of folds")
    if set(records["label"]) != {0, 1}:
        raise ValueError("Both labels 0 and 1 must be present")
    if records["label"].value_counts().min() < n_splits:
        raise ValueError("Each class needs at least n_splits samples")

    folds = np.full(len(records), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    for fold, (_, valid_indices) in enumerate(
        splitter.split(records, records["label"], groups=records["patient_group"])
    ):
        folds[valid_indices] = fold
    if (folds < 0).any():
        raise RuntimeError("Some samples were not assigned a fold")
    records["fold"] = folds
    records["split"] = np.where(folds == val_fold, "valid", "train")

    train_groups = set(records.loc[records["split"] == "train", "patient_group"])
    valid_groups = set(records.loc[records["split"] == "valid", "patient_group"])
    if train_groups & valid_groups:
        raise RuntimeError("Patient groups overlap between training and validation")
    for split in ("train", "valid"):
        if set(records.loc[records["split"] == split, "label"]) != {0, 1}:
            raise ValueError(f"Split {split!r} must contain both classes")
    return records


def split_summary(records_with_splits: pd.DataFrame) -> dict:
    """Return JSON-serializable class counts and patient groups per partition."""
    summary = {}
    for split in ("train", "valid"):
        records = records_with_splits.loc[records_with_splits["split"] == split]
        counts = records["label"].value_counts()
        summary[split] = {
            "samples": int(len(records)),
            "groups": int(records["patient_group"].nunique()),
            "patient_groups": sorted(records["patient_group"].unique().tolist()),
            "label_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1)},
        }
    return summary


class ImageDataset(Dataset):
    """Load RGB images with a deterministic, aspect-preserving square letterbox."""

    def __init__(
        self, records: pd.DataFrame, data_root: Path, image_size: int = 224,
        augment: bool = False, geometry_mode: str = "flip_rotate",
    ):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.augment = bool(augment)
        self.geometry_mode = geometry_mode
        self.records = records.reset_index(drop=True).copy()
        self.data_root = Path(data_root).resolve()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        tensor = load_image_tensor(
            self.data_root / row["path"], self.image_size,
            augment=self.augment, geometry_mode=self.geometry_mode,
        )
        return tensor, torch.tensor(int(row["label"]), dtype=torch.long)
