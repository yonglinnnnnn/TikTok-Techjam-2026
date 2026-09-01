from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_balanced(
    dataframe: pd.DataFrame,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Sample an equal number of real and AI images."""

    if "label" not in dataframe.columns:
        raise ValueError("Manifest must contain a 'label' column.")

    groups = []

    for label in [0, 1]:
        group = dataframe[dataframe["label"] == label]

        if len(group) == 0:
            raise ValueError(f"No images found for binary label {label}.")

        count = min(len(group), samples_per_class)

        groups.append(
            group.sample(
                n=count,
                random_state=seed,
            )
        )

    result = pd.concat(groups, ignore_index=True)

    return result.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def prepare_sid_manifest(input_path: Path) -> pd.DataFrame:
    """Load SID-Set and convert its labels to binary classification.

    SID-Set labels:
        0 = real
        1 = full synthetic
        2 = tampered

    ConvNeXt labels:
        0 = real
        1 = AI-generated or manipulated
    """

    dataframe = pd.read_csv(input_path)

    required_columns = {"image_path", "label", "split"}
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"SID manifest is missing columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()

    # Preserve the original SID label for analysis.
    dataframe["original_sid_label"] = dataframe["label"]

    # Convert SID labels into the ConvNeXt binary task.
    dataframe["label"] = dataframe["original_sid_label"].apply(
        lambda value: 0 if int(value) == 0 else 1
    )

    # Make sure all records identify themselves as SID-Set.
    dataframe["source"] = "SID_Set"

    # Confirm that the referenced files exist.
    missing_images = [
        path for path in dataframe["image_path"]
        if not Path(path).exists()
    ]

    if missing_images:
        print(
            f"Warning: {len(missing_images)} image paths do not exist."
        )
        dataframe = dataframe[
            dataframe["image_path"].map(
                lambda path: Path(path).exists()
            )
        ].copy()

    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create balanced SID-Set ConvNeXt manifests."
    )

    parser.add_argument(
        "--input",
        default="data/processed/sid_set_manifest.csv",
        help="Input SID-Set manifest.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for generated manifests.",
    )

    parser.add_argument(
        "--train-per-class",
        type=int,
        default=10000,
        help="Number of real and AI images for training.",
    )

    parser.add_argument(
        "--val-per-class",
        type=int,
        default=2000,
        help="Number of real and AI images for validation.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of a train-only manifest to reserve for validation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random sampling seed.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"SID manifest not found: {input_path}"
        )

    dataframe = prepare_sid_manifest(input_path)

    print("SID-Set records after cleaning:", len(dataframe))
    print("\nOriginal SID labels:")
    print(dataframe["original_sid_label"].value_counts().sort_index())

    train_df = dataframe[
        dataframe["split"].astype(str).str.lower() == "train"
    ].copy()

    validation_splits = {"val", "valid", "validation"}

    val_df = dataframe[
        dataframe["split"]
        .astype(str)
        .str.lower()
        .isin(validation_splits)
    ].copy()

    if train_df.empty:
        raise ValueError("No SID training records were found.")

    if val_df.empty:
        if not 0 < args.validation_fraction < 1:
            raise ValueError(
                "--validation-fraction must be between 0 and 1."
            )

        # The downloader exports one SID split at a time. Stratify a
        # train-only manifest so the manifest builder remains usable without
        # requiring a second, potentially very large, dataset download.
        validation_parts = []
        for label in [0, 1]:
            label_rows = train_df[train_df["label"] == label]
            validation_count = max(
                1,
                round(len(label_rows) * args.validation_fraction),
            )
            validation_parts.append(
                label_rows.sample(
                    n=validation_count,
                    random_state=args.seed,
                )
            )

        val_df = pd.concat(validation_parts, ignore_index=False)
        train_df = train_df.drop(index=val_df.index)
        print(
            "No validation split found; reserved "
            f"{len(val_df)} stratified rows from train."
        )

    train_small = sample_balanced(
        train_df,
        samples_per_class=args.train_per_class,
        seed=args.seed,
    )

    val_small = sample_balanced(
        val_df,
        samples_per_class=args.val_per_class,
        seed=args.seed,
    )

    train_output = output_dir / "train_manifest_sid.csv"
    val_output = output_dir / "val_manifest_sid.csv"

    train_small.to_csv(train_output, index=False)
    val_small.to_csv(val_output, index=False)

    print("\nCreated SID-only manifests:")
    print(f"Training:   {train_output}")
    print(f"Validation: {val_output}")

    print("\nTraining rows:", len(train_small))
    print(train_small["label"].value_counts().sort_index())

    print("\nValidation rows:", len(val_small))
    print(val_small["label"].value_counts().sort_index())

    print("\nOriginal SID training labels:")
    print(train_small["original_sid_label"].value_counts().sort_index())

    print("\nOriginal SID validation labels:")
    print(val_small["original_sid_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()

    