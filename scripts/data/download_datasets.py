"""
Download CIFAKE and SID-Set for the ConvNeXt training pipeline.

Run this file from the repository root:

    python -u scripts\\data\\download_datasets.py --skip-cifake --sid-max-images 100
    python -u scripts\\data\\download_datasets.py --skip-sid
    python -u scripts\\data\\download_datasets.py --sid-max-images 100

The script creates:

    data/raw/cifake/
    data/raw/sid_set/
    data/processed/cifake_manifest.csv
    data/processed/sid_set_manifest.csv
    data/processed/train_manifest.csv

Binary labels used by ConvNeXt:

    0 = real
    1 = AI-generated or tampered
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CIFAKE_DIR = RAW_DIR / "cifake"
SID_DIR = RAW_DIR / "sid_set"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def log(message: str) -> None:
    """
    Print immediately.

    The flush=True option ensures progress appears immediately
    in Windows CMD or PowerShell.
    """
    print(message, flush=True)


def write_manifest(rows: list[dict], output_path: Path) -> None:
    """
    Write image records into a CSV manifest.

    Each row contains:

        image_path
        label
        source
        split
        original_label
        image_id
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "image_path",
        "label",
        "source",
        "split",
        "original_label",
        "image_id",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    log(
        f"[OK] Wrote {len(rows):,} records: "
        f"{output_path}"
    )


def find_directory(
    root: Path,
    name: str,
) -> Path | None:
    """
    Find a directory without depending on capitalization.

    For example, this can find both:

        train
        TRAIN
        Train
    """
    target_name = name.lower()

    for path in root.rglob("*"):
        if (
            path.is_dir()
            and path.name.lower() == target_name
        ):
            return path

    return None


def find_images(root: Path) -> list[Path]:
    """
    Find supported image files below a directory.
    """
    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def download_cifake() -> Path:
    """
    Download CIFAKE from Kaggle using KaggleHub.

    The dataset is placed inside:

        data/raw/cifake/
    """
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is missing. Install it with:\n"
            "python -m pip install kagglehub"
        ) from error

    CIFAKE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("[INFO] Downloading CIFAKE from Kaggle...")

    downloaded_path = kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images",
        output_dir=str(CIFAKE_DIR),
    )

    dataset_path = Path(downloaded_path)

    log(
        f"[OK] CIFAKE downloaded to: "
        f"{dataset_path}"
    )

    return dataset_path


def build_cifake_manifest(
    root: Path,
) -> list[dict]:
    """
    Scan CIFAKE and create a manifest.

    Expected CIFAKE structure:

        train/
        ├── FAKE/
        └── REAL/

        test/
        ├── FAKE/
        └── REAL/

    Binary label mapping:

        REAL -> 0
        FAKE -> 1
    """
    rows: list[dict] = []

    for split in ("train", "test"):
        split_dir = find_directory(
            root,
            split,
        )

        if split_dir is None:
            log(
                f"[WARNING] CIFAKE split not found: "
                f"{split}"
            )
            continue

        for class_name, label in (
            ("REAL", 0),
            ("FAKE", 1),
        ):
            class_dir = find_directory(
                split_dir,
                class_name,
            )

            if class_dir is None:
                log(
                    f"[WARNING] CIFAKE folder not found: "
                    f"{split}/{class_name}"
                )
                continue

            image_files = find_images(class_dir)

            log(
                f"[INFO] CIFAKE "
                f"{split}/{class_name}: "
                f"{len(image_files):,} images"
            )

            for image_path in image_files:
                rows.append(
                    {
                        "image_path": str(
                            image_path.resolve()
                        ),
                        "label": label,
                        "source": "CIFAKE",
                        "split": split,
                        "original_label": class_name,
                        "image_id": image_path.stem,
                    }
                )

    return rows


def export_sid_set(
    split: str,
    max_images: int | None,
) -> list[dict]:
    """
    Load SID-Set from Hugging Face and export images locally.

    SID-Set original labels:

        0 = real
        1 = full synthetic
        2 = tampered

    Binary labels used by ConvNeXt:

        0 -> real
        1 -> AI
        2 -> AI

    Images are exported to:

        data/raw/sid_set/<split>/real/
        data/raw/sid_set/<split>/ai/

    Streaming is used so that:

        --sid-max-images 100

    does not first download the entire SID-Set split.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "datasets is missing. Install it with:\n"
            "python -m pip install datasets"
        ) from error

    log(
        f"[INFO] Loading SID-Set split "
        f"'{split}' from Hugging Face..."
    )

    log(
        "[INFO] Streaming mode is enabled. "
        "Only the required rows will be processed."
    )

    dataset = load_dataset(
        "saberzl/SID_Set",
        split=split,
        streaming=True,
    )

    if max_images is None:
        row_iterator = enumerate(dataset)
        progress_text = (
            "all available SID-Set rows"
        )
    else:
        row_iterator = enumerate(
            itertools.islice(
                dataset,
                max_images,
            )
        )
        progress_text = (
            f"up to {max_images:,} SID-Set rows"
        )

    log(
        f"[INFO] Exporting {progress_text}..."
    )

    real_dir = (
        SID_DIR
        / split
        / "real"
    )

    ai_dir = (
        SID_DIR
        / split
        / "ai"
    )

    real_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ai_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    processed = 0

    for index, record in row_iterator:
        processed += 1

        original_label = int(
            record["label"]
        )

        if original_label == 0:
            binary_label = 0
            class_name = "real"

        elif original_label in (1, 2):
            binary_label = 1
            class_name = "ai"

        else:
            log(
                f"[WARNING] Unknown SID-Set label "
                f"{original_label}; skipping row {index}"
            )
            continue

        image = record["image"]

        if image is None:
            log(
                f"[WARNING] Missing image at "
                f"SID-Set row {index}; skipping"
            )
            continue

        image = image.convert("RGB")

        raw_id = (
            record.get("img_id")
            or f"{split}_{index:08d}"
        )

        image_id = (
            str(raw_id)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        output_path = (
            SID_DIR
            / split
            / class_name
            / f"{image_id}.jpg"
        )

        # The script is resumable.
        # Existing images are not exported again.
        if not output_path.exists():
            image.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

        rows.append(
            {
                "image_path": str(
                    output_path.resolve()
                ),
                "label": binary_label,
                "source": "SID_Set",
                "split": split,
                "original_label": original_label,
                "image_id": image_id,
            }
        )

        if processed % 25 == 0:
            log(
                f"[INFO] SID-Set progress: "
                f"{processed:,} rows exported"
            )

    log(
        f"[OK] Finished exporting "
        f"{processed:,} SID-Set rows"
    )

    return rows


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--skip-cifake",
        action="store_true",
        help="Skip CIFAKE download and processing.",
    )

    parser.add_argument(
        "--skip-sid",
        action="store_true",
        help="Skip SID-Set download and processing.",
    )

    parser.add_argument(
        "--sid-split",
        choices=("train", "val", "test"),
        default="train",
        help=(
            "SID-Set split to process. "
            "Default: train."
        ),
    )

    parser.add_argument(
        "--sid-max-images",
        type=int,
        default=None,
        help=(
            "Maximum SID-Set images to export. "
            "Use 100 for a small test."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Main dataset download and manifest-generation workflow.
    """
    args = parse_args()

    if (
        args.skip_cifake
        and args.skip_sid
    ):
        log(
            "[ERROR] You cannot skip both "
            "CIFAKE and SID-Set."
        )
        return 1

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cifake_rows: list[dict] = []
    sid_rows: list[dict] = []

    if not args.skip_cifake:
        cifake_root = download_cifake()

        cifake_rows = build_cifake_manifest(
            cifake_root
        )

        write_manifest(
            cifake_rows,
            PROCESSED_DIR
            / "cifake_manifest.csv",
        )

    if not args.skip_sid:
        sid_rows = export_sid_set(
            split=args.sid_split,
            max_images=args.sid_max_images,
        )

        write_manifest(
            sid_rows,
            PROCESSED_DIR
            / "sid_set_manifest.csv",
        )

    # Only training rows are added to the
    # combined training manifest.
    combined_rows = [
        row
        for row in (
            *cifake_rows,
            *sid_rows,
        )
        if row["split"] == "train"
    ]

    write_manifest(
        combined_rows,
        PROCESSED_DIR
        / "train_manifest.csv",
    )

    log(
        "[COMPLETE] Dataset preparation finished."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

