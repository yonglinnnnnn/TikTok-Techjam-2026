#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.augmentations import build_clean_transform
from truesight.vision.gradcam import ConvNeXtGradCAM, overlay_heatmap
from truesight.vision.inference import load_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_detector(args.checkpoint, device=args.device)
    device = next(model.parameters()).device

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tensor = build_clean_transform(args.image_size)(image=image_rgb)["image"].unsqueeze(0).to(device)
    cam = ConvNeXtGradCAM(model)
    try:
        probability, heat = cam(tensor)
    finally:
        cam.remove_hooks()

    overlay_heatmap(image_rgb, heat, args.output)
    print(f"AIGC probability: {probability:.4f}")
    print(f"Heatmap: {args.output}")


if __name__ == "__main__":
    main()
