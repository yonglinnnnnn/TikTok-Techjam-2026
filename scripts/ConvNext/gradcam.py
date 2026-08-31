#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# scripts/convnext/gradcam.py
# parents[0] = scripts/convnext
# parents[1] = scripts
# parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src"

if not SRC_DIR.exists():
    raise FileNotFoundError(
        f"Source directory not found: {SRC_DIR}"
    )

sys.path.insert(0, str(SRC_DIR))


from truesight.vision.gradcam import (
    IMAGE_EXTENSIONS,
    get_device,
    preprocess_image,
    save_heatmap_overlay,
    GradCAM,
)
from truesight.vision.inference import load_detector


def find_images(
    input_dir: Path,
) -> list[Path]:
    """Find images recursively inside a folder."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def write_json(
    data: dict,
    output_path: Path,
) -> None:
    """Write the prediction and heatmap report."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def predict_probability(
    model,
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> float:
    """Predict the AI-generated probability for one image."""
    model_input, _ = preprocess_image(
        image_path,
        image_size=image_size,
    )

    model_input = model_input.to(device)

    with torch.no_grad():
        logits = model(model_input)
        probability = torch.sigmoid(
            logits.reshape(-1)[0]
        ).item()

    return float(probability)


def process_folder(
    checkpoint: Path,
    input_dir: Path,
    output_dir: Path,
    output_json: Path,
    threshold: float,
    image_size: int,
    requested_device: str | None,
) -> None:
    """Predict a folder and generate heatmaps for fake predictions."""
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device(
        requested_device
    )

    print(
        f"[INFO] Device: {device}",
        flush=True,
    )

    if device.type == "cuda":
        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

    print(
        f"[INFO] Loading checkpoint: "
        f"{checkpoint}",
        flush=True,
    )

    model = load_detector(
        str(checkpoint),
        device=str(device),
    )

    model = model.to(device)
    model.eval()

    image_paths = find_images(
        input_dir
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {input_dir}"
        )

    print(
        f"[INFO] Found {len(image_paths):,} images",
        flush=True,
    )

    gradcam = GradCAM(model)

    predictions: list[dict] = []
    fake_count = 0

    try:
        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            try:
                probability = predict_probability(
                    model=model,
                    image_path=image_path,
                    image_size=image_size,
                    device=device,
                )

                is_fake = (
                    probability >= threshold
                )

                heatmap_path: str | None = None

                if is_fake:
                    fake_count += 1

                    model_input, display_image = (
                        preprocess_image(
                            image_path,
                            image_size=image_size,
                        )
                    )

                    model_input = model_input.to(
                        device
                    )

                    # Generate Grad-CAM only for images
                    # predicted as AI-generated.
                    with torch.enable_grad():
                        cam_result = gradcam.generate(
                            model_input
                        )

                    heatmap_file = (
                        output_dir
                        / f"{index:06d}_"
                        f"{image_path.stem}_"
                        "gradcam.png"
                    )

                    save_heatmap_overlay(
                        display_image=display_image,
                        heatmap=cam_result.heatmap,
                        output_path=heatmap_file,
                    )

                    heatmap_path = str(
                        heatmap_file.resolve()
                    )

                predictions.append(
                    {
                        "image_path": str(
                            image_path.resolve()
                        ),
                        "pred": probability,
                        "is_ai_generated": is_fake,
                        "threshold": threshold,
                        "heatmap_path": heatmap_path,
                    }
                )

                if index % 10 == 0:
                    print(
                        f"[INFO] Processed "
                        f"{index:,}/{len(image_paths):,}",
                        flush=True,
                    )

            except Exception as error:
                predictions.append(
                    {
                        "image_path": str(
                            image_path.resolve()
                        ),
                        "pred": None,
                        "is_ai_generated": None,
                        "threshold": threshold,
                        "heatmap_path": None,
                        "error": str(error),
                    }
                )

                print(
                    f"[WARNING] Failed: "
                    f"{image_path} ({error})",
                    flush=True,
                )

    finally:
        gradcam.remove_hooks()

    report = {
        "checkpoint": str(
            checkpoint.resolve()
        ),
        "input_dir": str(
            input_dir.resolve()
        ),
        "heatmap_dir": str(
            output_dir.resolve()
        ),
        "device": str(device),
        "threshold": threshold,
        "total_images": len(image_paths),
        "predicted_ai_images": fake_count,
        "predictions": predictions,
    }

    write_json(
        report,
        output_json,
    )

    print(
        f"[COMPLETE] Processed "
        f"{len(image_paths):,} images",
        flush=True,
    )

    print(
        f"[COMPLETE] Generated "
        f"{fake_count:,} heatmaps",
        flush=True,
    )

    print(
        f"[COMPLETE] Report saved to "
        f"{output_json}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict a folder and generate "
            "Grad-CAM heatmaps for images "
            "predicted as AI-generated."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.pt",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing input images",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder for Grad-CAM heatmaps",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="JSON prediction report path",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for AI prediction. "
            "Default: 0.5"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input size. Default: 224",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or automatic selection",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )

    process_folder(
        checkpoint=Path(
            args.checkpoint
        ).resolve(),
        input_dir=Path(
            args.input_dir
        ).resolve(),
        output_dir=Path(
            args.output_dir
        ).resolve(),
        output_json=Path(
            args.output_json
        ).resolve(),
        threshold=args.threshold,
        image_size=args.image_size,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
    