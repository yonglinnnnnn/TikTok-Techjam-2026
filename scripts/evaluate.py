#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from truesight.pipeline import TrueSightResult, run_pipeline
from truesight.utils import ALLOWED_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TrueSight pipeline on a directory of images.",
    )
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        required=True,
        type=Path,
        dest="input_dir",
        help="Directory containing images to evaluate recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.json"),
        help="Output JSON path (default: predictions.json).",
    )
    return parser.parse_args()


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def build_evaluation_record(result: TrueSightResult) -> dict[str, Any]:
    if result.confidence is None:
        raise ValueError(
            f"Pipeline returned no final confidence for {result.image_path}"
        )

    return {
        "image_path": Path(result.image_path).as_posix(),
        "pred": result.confidence,
        "metadata": {
            "tier1_watermark": result.tier1.watermark_detected,
            "vlm_source_estimate": (
                result.tier2.source if result.tier2 is not None else None
            ),
            "heatmap_generated": result.heatmap_path is not None,
        },
    }


def evaluate_directory(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    predictions: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []

    for image_path in discover_images(input_dir):
        try:
            result = run_pipeline(str(image_path))
            predictions.append(build_evaluation_record(result))
        except Exception as exc:
            failures.append((image_path, str(exc)))

    return predictions, failures


def main() -> None:
    args = parse_args()

    try:
        predictions, failures = evaluate_directory(args.input_dir)
    except NotADirectoryError as exc:
        raise SystemExit(f"TrueSight evaluation failed: {exc}") from exc

    processed_count = len(predictions) + len(failures)
    if processed_count == 0:
        raise SystemExit(
            f"TrueSight evaluation failed: no supported images in "
            f"{args.input_dir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(predictions, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    for image_path, error in failures:
        print(f"Failed: {image_path}: {error}", file=sys.stderr)

    print(
        f"Processed {processed_count} images: "
        f"{len(predictions)} succeeded, {len(failures)} failed.",
        file=sys.stderr,
    )
    print(f"Saved predictions to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
