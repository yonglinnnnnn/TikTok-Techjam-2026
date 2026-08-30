import random
from pathlib import Path

import kagglehub
import pandas as pd


def download_data():
    path = kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images"
    )
    print("Dataset location:", path)
    return Path(path)


def find_split_directory(dataset_path: Path, split: str) -> Path:
    candidates = [
        dataset_path / "transformed_data" / "transformed_data" / split,
        dataset_path / split,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find the '{split}' directory")


def create_manifest(
    dataset_path: Path,
    split: str,
    output_csv: str,
    limit: int = 1000,
    seed: int = 42,
):
    split_directory = find_split_directory(dataset_path, split)
    rng = random.Random(seed)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Half real and half fake.
    per_class = limit // 2
    records = []

    for class_name, label in {"real": 0, "fake": 1}.items():
        class_directory = split_directory / class_name

        images = [
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ]

        rng.shuffle(images)
        selected_images = images[:per_class]

        if len(selected_images) < per_class:
            raise ValueError(
                f"Requested {per_class} {class_name} images, "
                f"but only found {len(selected_images)}"
            )

        for image_path in selected_images:
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "source": "CIFAKE",
                    "split": split,
                }
            )

    rng.shuffle(records)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    dataset_path = download_data()

    create_manifest(
        dataset_path=dataset_path,
        split="train",
        output_csv="data/processed/cifake_train_1000.csv",
        limit=1000,
    )

    create_manifest(
        dataset_path=dataset_path,
        split="test",
        output_csv="data/processed/cifake_val_1000.csv",
        limit=1000,
    )