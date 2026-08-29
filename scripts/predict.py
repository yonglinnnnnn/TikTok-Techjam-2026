import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import ALLOWED_EXTENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TrueSight on a directory of images."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing images"
    )

    parser.add_argument(
        "--output",
        default="predictions.json",
        help="Path to output JSON file"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # Check input folder exists
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    # Make output folder if needed
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = []

    for image_path in sorted(input_dir.iterdir()):

        # Skip non-image files
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in ALLOWED_EXTENSIONS
        ):
            continue

        result = run_pipeline(
            str(image_path)
        )

        predictions.append({
            "image_path": str(image_path),
            "pred": result.final_confidence
        })

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Processed {len(predictions)} images.")

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
