from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "convnext_tiny"
    pretrained: bool = True
    num_classes: int = 1
    dropout: float = 0.15
    unfreeze_stages: int = 2
    target_image_size: int = 224


@dataclass
class TrainingConfig:
    epochs: int = 12
    warmup_epochs: int = 2
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    backbone_lr: float = 1e-5
    head_lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 4
    consistency_weight: float = 0.05
    use_consistency_view: bool = True


@dataclass
class AugmentationConfig:
    # Whether robustness augmentations are enabled.
    enabled: bool = False

    jpeg_quality: tuple[int, int] = (30, 90)
    blur_sigma: tuple[float, float] = (0.5, 2.0)
    downscale: tuple[float, float] = (0.25, 0.5)
    noise_std: tuple[float, float] = (0.02, 0.10)
    color_jitter: tuple[float, float] = (0.8, 1.2)
    crop_scale: tuple[float, float] = (0.8, 1.0)

    p_jpeg: float = 0.45
    p_blur: float = 0.25
    p_downscale: float = 0.30
    p_noise: float = 0.20
    p_color: float = 0.35
    p_crop: float = 0.35
    p_horizontal_flip: float = 0.5


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    min_lr_ratio: float = 0.05


@dataclass
class OutputConfig:
    save_last: bool = True
    save_best: bool = True


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _convert_tuple_fields(data: dict[str, Any]) -> dict[str, Any]:
    for section, keys in {
        "augmentation": [
            "jpeg_quality",
            "blur_sigma",
            "downscale",
            "noise_std",
            "color_jitter",
            "crop_scale",
        ]
    }.items():
        if section in data:
            for key in keys:
                if key in data[section]:
                    data[section][key] = tuple(data[section][key])
    return data


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _convert_tuple_fields(raw)
    return Config(
        seed=raw.get("seed", 42),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
