import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data/processed")
    
    # Load existing full manifests
    train_df = pd.read_csv(data_dir / "train_manifest_sid.csv")
    val_df = pd.read_csv(data_dir / "val_manifest_sid.csv")

    # Sample exactly 2500 per class for training (Total: 5000)
    # random_state ensures reproducibility
    small_train = train_df.groupby("label").sample(n=5000, random_state=42)
    
    # Sample exactly 500 per class for validation (Total: 1000)
    small_val = val_df.groupby("label").sample(n=500, random_state=42)

    # Save to new smaller manifest files
    train_out = data_dir / "train_manifest_sid_small.csv"
    val_out = data_dir / "val_manifest_sid_small.csv"
    
    small_train.to_csv(train_out, index=False)
    small_val.to_csv(val_out, index=False)

    print("Created downsampled manifests!")
    print(f"Training Saved to: {train_out}")
    print(small_train["label"].value_counts())
    print(f"\nValidation Saved to: {val_out}")
    print(small_val["label"].value_counts())

if __name__ == "__main__":
    main()

