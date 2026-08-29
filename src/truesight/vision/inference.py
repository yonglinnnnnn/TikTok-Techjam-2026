from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

from .augmentations import build_clean_transform
from .model import ConvNeXtAIGCDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_detector(checkpoint: str | Path, device: str | None = None) -> ConvNeXtAIGCDetector:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    model = ConvNeXtAIGCDetector.from_checkpoint(payload, map_location=device_obj)
    model.eval()
    return model


def predict_image(model: ConvNeXtAIGCDetector, image_path: str | Path, image_size: int = 224) -> float:
    image_path = str(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_clean_transform(image_size)
    tensor = transform(image=image)["image"].unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        return float(model.predict_proba(tensor)[0].item())


def predict_directory(
    model: ConvNeXtAIGCDetector,
    input_dir: str | Path,
    image_size: int = 224,
) -> list[dict[str, float | str]]:
    input_dir = Path(input_dir)
    paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    results = []
    for path in paths:
        results.append({"image_path": str(path), "pred": predict_image(model, path, image_size)})
    return results


def save_predictions(results: list[dict], output_json: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
