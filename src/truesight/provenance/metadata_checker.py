"""Non-cryptographic EXIF/XMP/software-marker inspection."""
from pathlib import Path
from typing import Any
from PIL import Image

MARKERS = {"openai": "OpenAI", "dall-e": "DALL-E", "dalle": "DALL-E",
           "gpt-image": "OpenAI", "midjourney": "Midjourney",
           "stable diffusion": "Stable Diffusion", "firefly": "Adobe Firefly",
           "trainedalgorithmicmedia": "Generative AI"}

def check_metadata(image_path: str | Path) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image:
            raw = {str(k): str(v) for k, v in image.getexif().items()}
            raw.update({str(k): str(v) for k, v in image.info.items()
                        if k in {"xmp", "XML:com.adobe.xmp", "parameters", "Software"}})
        text = " ".join(f"{k}={v}" for k, v in raw.items()).lower()
        hits = sorted({label for marker, label in MARKERS.items() if marker in text})
        return {"checked": True, "ai_markers": hits, "metadata": raw, "error": None}
    except Exception as exc:
        return {"checked": False, "ai_markers": [], "metadata": {}, "error": str(exc)}
