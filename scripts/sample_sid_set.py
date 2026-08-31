"""
Pulls a small, balanced sample from SID_Set (real / full-synthetic / tampered)
for local testing of the Tier 2 VLM pipeline — NOT for training the ConvNeXt
model (that's Member 1/5's job with the full 210k-row train split).

Why this exists: your VLM providers call external APIs, so there's no
"training set" for you — but you still need real example images to
(a) sanity-check your prompts actually produce sensible JSON,
(b) tune the confidence-weighted merge in aggregator.py, and
(c) generate the DALL-E-vs-real / tampered-vs-full-synthetic case study
    examples the "Multi-Agent Reasoning" Devpost section asks for.

Usage:
    python scripts/sample_sid_set.py --n-per-class 15 --split validation

Requires: pip install datasets  (already in requirements.txt)
Downloads only the requested rows (streaming), not the full 140GB dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# label -> class name, matching SID_Set's Dataset Card exactly
LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}

OUT_DIR = Path("tests/fixtures/sample_images")


def main(n_per_class: int, split: str) -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    counts: dict[int, int] = defaultdict(int)

    print(f"Streaming saberzl/SID_Set [{split}] — stopping once we have "
          f"{n_per_class} images per class...")

    # streaming=True avoids pulling the full parquet shards to disk —
    # important since the dataset is 140GB total.
    ds = load_dataset("saberzl/SID_Set", split=split, streaming=True)

    for row in ds:
        label = row["label"]
        if counts[label] >= n_per_class:
            if all(counts[l] >= n_per_class for l in LABEL_NAMES):
                break
            continue

        class_name = LABEL_NAMES[label]
        img_id = row["img_id"]
        out_path = OUT_DIR / f"{class_name}_{img_id}.jpg"
        row["image"].convert("RGB").save(out_path, "JPEG", quality=95)

        manifest.append({
            "image_path": str(out_path),
            "img_id": img_id,
            "label": label,
            "label_name": class_name,
            "has_mask": row.get("mask") is not None,
        })
        counts[label] += 1
        print(f"  saved {out_path.name}  ({dict(counts)})")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} images saved to {OUT_DIR}/")
    print(f"Manifest (ground truth labels): {manifest_path}")
    print("\nNote: label 1 (full_synthetic) is the closest match to what "
          "your VLM prompt calls 'AI-generated'; label 2 (tampered) is a "
          "partially-edited real photo — a good stress test for whether "
          "your prompt's ai_coverage score responds sensibly to partial "
          "edits rather than just doing a binary real/fake call.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=15,
                         help="Images to sample per label (default: 15)")
    parser.add_argument("--split", default="validation",
                         choices=["train", "validation"],
                         help="SID_Set split to sample from (default: validation, "
                              "to avoid touching Member 1's training data)")
    args = parser.parse_args()
    main(args.n_per_class, args.split)