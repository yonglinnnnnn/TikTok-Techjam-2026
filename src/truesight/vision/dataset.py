from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import build_clean_transform, build_consistency_transform, build_train_transform
from .config import AugmentationConfig


class AIGCManifestDataset(Dataset):
    """Dataset driven by a CSV manifest.

    Required columns: image_path, label.
    Optional columns: source, split, sample_id.

    This keeps dataset preparation separate from Member 1's model code and lets
    Member 5 add/remove datasets without changing this package.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 224,
        train: bool = True,
        consistency_view: bool = False,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.frame = pd.read_csv(self.manifest_path)
        required = {"image_path", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        self.frame["label"] = self.frame["label"].astype(float)
        if not self.frame["label"].isin([0.0, 1.0]).all():
            raise ValueError("Labels must be binary: 0=real, 1=AIGC")

        self.train = train
        self.consistency_view = consistency_view and train
        self.image_size = image_size
        self.aug_cfg = augmentation_config or AugmentationConfig()

        if train:
            self.transform = build_train_transform(image_size, self.aug_cfg)
            self.consistency_transform = (
                build_consistency_transform(image_size, self.aug_cfg)
                if self.consistency_view
                else None
            )
        else:
            self.transform = build_clean_transform(image_size)
            self.consistency_transform = None

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: str) -> Any:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = str(row["image_path"])
        image = self._load_image(image_path)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        output = {
            "image": self.transform(image=image)["image"],
            "label": label,
            "image_path": image_path,
        }

        if "source" in row.index:
            output["source"] = str(row["source"])

        if self.consistency_transform is not None:
            output["image_consistency"] = self.consistency_transform(image=image)["image"]

        return output
