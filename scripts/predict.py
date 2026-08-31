#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from truesight.pipeline import PipelineComponents, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TrueSight pipeline on one image.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        dest="input_path",
        help="Path to the image to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to save the unified JSON result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = PipelineComponents.real()

    try:
        result = run_pipeline(str(args.input_path), components=components)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"TrueSight prediction failed: {exc}") from exc

    json_result = json.dumps(
        result.to_dict(),
        indent=2,
        ensure_ascii=False,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{json_result}\n", encoding="utf-8")
        print(f"Saved prediction to: {args.output}", file=sys.stderr)

    print(json_result)


if __name__ == "__main__":
    main()
