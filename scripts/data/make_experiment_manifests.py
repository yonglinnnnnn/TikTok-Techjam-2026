import pandas as pd
from pathlib import Path

input_file = Path("data/processed/cifake_manifest.csv")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# Balanced sampling:
# 10,000 real + 10,000 fake for training
train_small = (
    train.groupby("label", group_keys=False)
    .apply(lambda group: group.sample(n=min(len(group), 10000), random_state=42))
    .reset_index(drop=True)
)

# 2,000 real + 2,000 fake for validation
val_small = (
    test.groupby("label", group_keys=False)
    .apply(lambda group: group.sample(n=min(len(group), 2000), random_state=42))
    .reset_index(drop=True)
)

train_small.to_csv(output_dir / "train_manifest_small.csv", index=False)
val_small.to_csv(output_dir / "val_manifest_small.csv", index=False)

print("Training rows:", len(train_small))
print("Validation rows:", len(val_small))
print(train_small["label"].value_counts())
print(val_small["label"].value_counts())
