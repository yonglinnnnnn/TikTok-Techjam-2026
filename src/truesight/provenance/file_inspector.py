"""Stable file identity and container facts (not evidence of AI generation)."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from PIL import Image

def inspect_file(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
            "format": image.format,
            "mime_type": image.get_format_mimetype(),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
