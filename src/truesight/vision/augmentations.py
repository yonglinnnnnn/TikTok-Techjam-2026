from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import AugmentationConfig

# ImageNet statistics match the pretrained torchvision ConvNeXt weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(size: int, cfg: AugmentationConfig) -> A.Compose:
    """Build robustness-oriented training augmentation.

    The pipeline intentionally covers the transformation families listed by the
    hackathon brief. Parameters are exposed through YAML so experiments can be
    logged without editing source code.
    """
    return A.Compose([
        A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, p=1.0),
        A.RandomCrop(height=size, width=size, p=1.0),
        A.HorizontalFlip(p=cfg.p_horizontal_flip),
        # A corruption slot prevents every heavy degradation from being applied
        # simultaneously, while still sampling across all challenge-specified
        # failure modes. Color jitter is kept separate because it is usually
        # compatible with compression/downsampling in real reposting pipelines.
        A.OneOf([
            A.ImageCompression(
                quality_range=cfg.jpeg_quality,
                compression_type="jpeg",
                p=1.0,
            ),
            A.GaussianBlur(
                sigma_range=cfg.blur_sigma,
                blur_range=(0, 0),
                p=1.0,
            ),
            A.Downscale(
                scale_range=cfg.downscale,
                interpolation_pair={
                    "downscale": cv2.INTER_AREA,
                    "upscale": cv2.INTER_LINEAR,
                },
                p=1.0,
            ),
            A.GaussNoise(
                std_range=cfg.noise_std,
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=1.0,
            ),
        ], p=0.65),
        A.ColorJitter(
            brightness_range=cfg.color_jitter,
            contrast_range=cfg.color_jitter,
            saturation_range=cfg.color_jitter,
            hue_range=(-0.05, 0.05),
            p=cfg.p_color,
        ),
        # Explicitly simulate the 80% center-crop scenario from the brief.
        A.OneOf([
            A.CenterCrop(height=int(size * 0.8), width=int(size * 0.8), p=1.0),
            A.NoOp(p=1.0),
        ], p=cfg.p_crop),
        A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_clean_transform(size: int) -> A.Compose:
    """Deterministic-ish evaluation preprocessing without robustness corruption."""
    return A.Compose([
        A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, p=1.0),
        A.CenterCrop(height=size, width=size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_consistency_transform(size: int, cfg: AugmentationConfig) -> A.Compose:
    """Stronger transformation view used for optional consistency regularisation."""
    return A.Compose([
        A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, p=1.0),
        A.RandomResizedCrop(
            size=(size, size),
            scale=cfg.crop_scale,
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_AREA,
            area_for_downscale="image",
            p=1.0,
        ),
        A.OneOf([
            A.ImageCompression(quality_range=cfg.jpeg_quality, compression_type="jpeg", p=1.0),
            A.GaussianBlur(sigma_range=cfg.blur_sigma, blur_range=(0, 0), p=1.0),
            A.Downscale(
                scale_range=cfg.downscale,
                interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                p=1.0,
            ),
            A.GaussNoise(
                std_range=cfg.noise_std,
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=1.0,
            ),
        ], p=0.85),
        A.ColorJitter(
            brightness_range=cfg.color_jitter,
            contrast_range=cfg.color_jitter,
            saturation_range=cfg.color_jitter,
            hue_range=(-0.05, 0.05),
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
