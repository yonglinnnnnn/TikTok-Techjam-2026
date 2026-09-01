from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.inference import load_detector


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def final_preprocessing(size: int) -> list:
    """Resize and normalize an image for ConvNeXt."""
    return [
        A.Resize(
            height=size,
            width=size,
            interpolation=cv2.INTER_AREA,
            p=1.0,
        ),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]


def build_transforms(size: int) -> dict[str, Callable]:
    """Build deterministic robustness transformations.

    Each transformed image receives the same final resize and normalization.
    """

    clean = A.Compose(
        [
            A.SmallestMaxSize(
                max_size=size,
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            A.CenterCrop(
                height=size,
                width=size,
                p=1.0,
            ),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    jpeg = A.Compose(
        [
            A.ImageCompression(
                quality_range=(30, 30),
                compression_type="jpeg",
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    blur = A.Compose(
        [
            A.GaussianBlur(
                blur_limit=(3, 3),
                sigma_limit=(2.0, 2.0),
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    resize = A.Compose(
        [
            A.Downscale(
                scale_range=(0.25, 0.25),
                interpolation_pair={
                    "downscale": cv2.INTER_AREA,
                    "upscale": cv2.INTER_LINEAR,
                },
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    noise = A.Compose(
        [
            A.GaussNoise(
                std_range=(0.10, 0.10),
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    colour = A.Compose(
        [
            A.ColorJitter(
                brightness=(1.2, 1.2),
                contrast=(1.2, 1.2),
                saturation=(1.2, 1.2),
                hue=(0.0, 0.0),
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    crop = A.Compose(
        [
            A.SmallestMaxSize(
                max_size=size,
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            A.CenterCrop(
                height=int(size * 0.8),
                width=int(size * 0.8),
                p=1.0,
            ),
            *final_preprocessing(size),
        ]
    )

    return {
        "clean": clean,
        "jpeg_q30": jpeg,
        "blur_sigma2": blur,
        "resize_0.25": resize,
        "noise_0.10": noise,
        "colour_plus20": colour,
        "crop_80": crop,
    }


def load_image(path: str) -> np.ndarray:
    """Read an image as RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict:
    """Calculate classification and confusion-matrix metrics."""

    predictions = [
        1 if probability >= threshold else 0
        for probability in probabilities
    ]

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_real = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_ai = int(matrix[1, 1])

    result = {
        "total_images": len(labels),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": {
            "true_real": true_real,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_ai": true_ai,
        },
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["real", "ai"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return result


def evaluate_condition(
    model,
    dataframe: pd.DataFrame,
    transform,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate one transformation condition."""

    labels: list[int] = []
    probabilities: list[float] = []

    batch_images: list[torch.Tensor] = []
    batch_labels: list[int] = []

    def process_batch() -> None:
        if not batch_images:
            return

        inputs = torch.stack(batch_images).to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.sigmoid(logits)

        probabilities.extend(
            float(value)
            for value in probs.detach().cpu().flatten().tolist()
        )
        labels.extend(batch_labels)

        batch_images.clear()
        batch_labels.clear()

    for row in dataframe.itertuples(index=False):
        image_path = Path(row.image_path)
        label = int(row.label)

        image = load_image(str(image_path))
        transformed = transform(image=image)["image"]

        if transformed.ndim != 3:
            raise ValueError(
                f"Unexpected tensor shape for {image_path}: "
                f"{tuple(transformed.shape)}"
            )

        batch_images.append(transformed.float())
        batch_labels.append(label)

        if len(batch_images) >= batch_size:
            process_batch()

    process_batch()

    return calculate_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=0.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ConvNeXt robustness on CIFAKE."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the ConvNeXt checkpoint.",
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Validation CSV manifest.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON report path.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="AI classification threshold.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device, for example cpu or cuda.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    manifest_path = Path(args.manifest)
    output_path = Path(args.output_json)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    dataframe = pd.read_csv(manifest_path)

    required_columns = {"image_path", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("The manifest contains no images.")

    missing_images = [
        str(path)
        for path in dataframe["image_path"]
        if not Path(path).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images listed in the manifest do not exist. "
            f"First missing image: {missing_images[0]}"
        )

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Images: {len(dataframe)}")

    model = load_detector(
        args.checkpoint,
        device=str(device),
    )
    model.eval()

    transforms = build_transforms(args.image_size)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(manifest_path.resolve()),
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "total_images": len(dataframe),
        "conditions": {},
    }

    for condition_name, transform in transforms.items():
        print(f"Evaluating condition: {condition_name}")

        condition_report = evaluate_condition(
            model=model,
            dataframe=dataframe,
            transform=transform,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
        )

        condition_report["threshold"] = args.threshold
        report["conditions"][condition_name] = condition_report

        print(
            f"{condition_name}: "
            f"accuracy={condition_report['accuracy']:.4f}, "
            f"F1={condition_report['f1']:.4f}, "
            f"ROC-AUC={condition_report['roc_auc']:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nWrote robustness report to: {output_path}")


if __name__ == "__main__":
    main()