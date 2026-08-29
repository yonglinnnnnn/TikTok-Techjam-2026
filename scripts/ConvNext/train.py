#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.config import load_config
from truesight.vision.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TrueSight ConvNeXt AIGC detector")
    parser.add_argument("--manifest", required=True, help="Training CSV manifest")
    parser.add_argument("--val-manifest", required=True, help="Validation CSV manifest")
    parser.add_argument("--config", default=str(ROOT / "configs" / "convnext_tiny.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "member1" / "convnext_tiny"))
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.manifest, args.val_manifest, args.output_dir)


if __name__ == "__main__":
    main()
