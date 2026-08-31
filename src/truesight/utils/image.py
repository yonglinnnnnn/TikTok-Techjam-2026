from pathlib import Path

from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Validate that the image exists, is of a supported format, and is not corrupted
def validate_image(image_path: str) -> None:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image: {image_path}") from exc
