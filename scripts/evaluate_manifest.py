#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from truesight.vision.config import (
    AugmentationConfig,
)
from truesight.vision.dataset import (
    AIGCManifestDataset,
)
from truesight.vision.inference import (
    load_detector,
)
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    model = load_detector(
        args.checkpoint,
        device="cpu",
    )

    model = model.to(device)
    model.eval()

    dataset = AIGCManifestDataset(
        args.manifest,
        image_size=224,
        train=False,
        consistency_view=False,
        augmentation_config=AugmentationConfig(
            enabled=False,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_labels: list[int] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_labels.extend(
                labels.cpu().numpy().astype(int).tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    labels_array = np.asarray(all_labels)
    probabilities_array = np.asarray(
        all_probabilities
    )

    predictions_array = (
        probabilities_array >= args.threshold
    ).astype(int)

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=[0, 1],
    )

    report = classification_report(
        labels_array,
        predictions_array,
        labels=[0, 1],
        target_names=["real", "ai"],
        output_dict=True,
        zero_division=0,
    )

    output = {
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "manifest": str(
            Path(args.manifest).resolve()
        ),
        "device": str(device),
        "threshold": args.threshold,
        "total_images": len(labels_array),
        "confusion_matrix": {
            "true_real": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_ai": int(matrix[1, 1]),
        },
        "classification_report": report,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

    