#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.inference import load_detector, predict_directory, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Member 1 ConvNeXt inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = load_detector(args.checkpoint, device=args.device)
    results = predict_directory(model, args.input_dir, image_size=args.image_size)
    save_predictions(results, args.output_json)
    print(f"Wrote {len(results)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
