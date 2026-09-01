#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from truesight.pipeline import PipelineComponents, run_pipeline
from truesight.utils import ALLOWED_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete TrueSight three-tier pipeline on every image in "
            "a directory. pred is the fused confidence from 0.0 (real) to "
            "1.0 (AIGC)."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained .pt checkpoint")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of images")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("predictions.json"),
        help="Destination JSON file (default: predictions.json)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Inference device (default: CUDA when available, otherwise CPU)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be greater than zero")

    image_paths = sorted(
        (
            path
            for path in args.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if not image_paths:
        raise SystemExit(f"No supported images found in: {args.input_dir}")

    # PipelineComponents.real connects Tier 1 provenance/forensics, Tier 2 VLM,
    # and Tier 3 ConvNeXt. The orchestrator applies routing and score fusion.
    components = PipelineComponents.real(
        checkpoint=args.checkpoint,
        run_vlm=True,
        run_forensics=True,
        generate_heatmap=False,
        image_size=args.image_size,
        device=args.device,
    )

    results: list[dict[str, str | float]] = []
    for image_path in image_paths:
        try:
            pipeline_result = run_pipeline(str(image_path), components=components)
        except Exception as exc:
            raise SystemExit(f"AIGC prediction failed for {image_path}: {exc}") from exc

        if pipeline_result.confidence is None:
            raise SystemExit(
                f"AIGC prediction failed for {image_path}: "
                "the three-tier pipeline produced no final confidence"
            )
        results.append(
            {
                "image_path": image_path.as_posix(),
                "pred": pipeline_result.confidence,
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        f"{json.dumps(results, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
