#!/usr/bin/env python3
"""Small, efficient tuning runner.

This is intentionally a lightweight launcher rather than a full HPO framework.
Run a handful of short experiments, inspect validation metrics, then promote the
best configuration to a full run.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.config import load_config
from truesight.vision.train import train


def set_nested(config: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Run a small parameter sweep")
    parser.add_argument("--base-config", default=str(ROOT / "configs" / "convnext_tiny.yaml"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "member1" / "tuning"))
    parser.add_argument("--epochs", type=int, default=3, help="Use short runs for screening")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    candidates = list(itertools.product([1, 2, 3], [1e-5, 3e-5], [224]))

    for idx, (unfreeze_stages, backbone_lr, image_size) in enumerate(candidates, start=1):
        raw = copy.deepcopy(base)
        set_nested(raw, "model.unfreeze_stages", unfreeze_stages)
        set_nested(raw, "training.backbone_lr", backbone_lr)
        set_nested(raw, "training.epochs", args.epochs)
        set_nested(raw, "model.target_image_size", image_size)
        temp_config = Path(args.output_root) / f"trial_{idx}" / "config.yaml"
        temp_config.parent.mkdir(parents=True, exist_ok=True)
        temp_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        config = load_config(temp_config)
        print(f"\n=== Trial {idx}: stages={unfreeze_stages}, backbone_lr={backbone_lr}, size={image_size} ===")
        train(config, args.manifest, args.val_manifest, temp_config.parent)


if __name__ == "__main__":
    main()
