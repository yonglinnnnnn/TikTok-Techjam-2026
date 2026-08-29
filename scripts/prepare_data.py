
"""
TrueSight - Single-image robustness transformation generator.

Purpose
-------
Take **ONE** real or AI-generated image and create label-preserving variants that
simulate common image-processing operations encountered during redistribution
on social-media platforms or messaging apps.

The transformations are intentionally motivated by the team's hackathon brief
and CS4243 Computer Vision lecture material:

Lecture-derived operations:
    - Brightness / intensity scaling
    - Gamma mapping
    - Histogram stretching
    - Gaussian smoothing / blur
    - Median filtering for impulse / salt-and-pepper noise
    - Sharpening via unsharp masking (original + alpha * details)
    - JPEG compression and recompression artifacts
    - Spatial downsampling / upsampling
    - RGB / HSV colour perturbation

Additional robustness operations:
    - Gaussian sensor-like noise
    - Salt-and-pepper noise
    - Mild rotation / translation / scale changes
    - Random crop followed by resize

Why this matters for TrueSight
------------------------------
The hackathon asks the detector to remain useful after JPEG compression,
blur, resize, noise, colour adjustment and cropping. Training only on clean
images risks learning shortcuts that disappear after redistribution.

This script creates *training variants* while preserving the original class
label. It does NOT decide whether an image is AI-generated. The caller must
provide the label in the command line or use the generated directory only as
an image source for a later manifest-building step.

Recommended usage
-----------------
Create the full deterministic transformation set for one image:

    python scripts/prepare_data.py \
        --input data/raw/example.jpg \
        --label ai \
        --output-dir data/processed/example_ai

Create only selected transformations:

    python scripts/prepare_data.py \
        --input data/raw/example.jpg \
        --label real \
        --output-dir data/processed/example_real \
        --transforms jpeg blur resize gaussian_noise gamma median sharpen

Create random robustness compositions (useful for generating more training
samples without saving every possible combination):

    python scripts/prepare_data.py \
        --input data/raw/example.jpg \
        --label ai \
        --output-dir data/processed/example_ai \
        --random-variants 10 \
        --seed 42

Important
---------
Do not run this script on the WildFake validation subset. The hackathon brief
explicitly reserves the specified WildFake subset for validation/reference
and says not to use it during training.

The script writes:
    - transformed JPEG images
    - manifest.json describing the transformation and label

The manifest is useful for auditing experiments and for converting the output
into the CSV manifest expected by Member 1's training pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structure used in the generated manifest.
# ---------------------------------------------------------------------------


@dataclass
class OutputRecord:
    """Metadata for one generated training image."""

    image_path: str
    label: int
    label_name: str
    source_image: str
    transformation: str
    parameters: dict
    seed: int | None


# ---------------------------------------------------------------------------
# Basic validation / I/O helpers
# ---------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray:
    """Read an image as an RGB uint8 NumPy array.

    OpenCV reads colour images in BGR order, while the rest of this script uses
    RGB. Keeping one explicit convention avoids the common RGB/BGR bug.
    """

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_image(path: Path, image_rgb: np.ndarray, jpeg_quality: int = 95) -> None:
    """Write an RGB uint8 image, creating parent directories if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(
            str(path),
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
    else:
        ok = cv2.imwrite(str(path), image_bgr)

    if not ok:
        raise IOError(f"Could not write image: {path}")


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Clip an image into the standard 8-bit [0, 255] representation."""

    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lecture-derived point processing operations
# ---------------------------------------------------------------------------


def adjust_brightness(image: np.ndarray, offset: int) -> np.ndarray:
    """Apply x_ij = p_ij + b, followed by clipping.

    This follows the lecture's brightness transformation. Positive b makes an
    image brighter; negative b makes it darker.
    """

    out = image.astype(np.int16) + int(offset)
    return ensure_uint8(out)


def adjust_contrast(image: np.ndarray, scale: float) -> np.ndarray:
    """Apply simple intensity scaling around mid-gray.

    This is the practical form of intensity scaling discussed in the lecture.
    Values are clipped to the 8-bit display range.
    """

    out = (image.astype(np.float32) - 127.5) * float(scale) + 127.5
    return ensure_uint8(out)


def gamma_map(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma mapping: 255 * (p / 255)^gamma.

    gamma < 1 brightens mid-level intensities; gamma > 1 darkens them.
    """

    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    normalized = image.astype(np.float32) / 255.0
    out = 255.0 * np.power(normalized, gamma)
    return ensure_uint8(out)


def histogram_stretch(image: np.ndarray) -> np.ndarray:
    """Linearly stretch each RGB channel to approximately [0, 255].

    The lecture describes histogram stretching as a linear mapping based on
    the observed minimum and maximum intensities. We use per-channel min/max
    here because the operation is intended as a robustness perturbation, not
    as a colour-preserving enhancement step.

    For a training dataset, keep this transformation probability modest: an
    overly strong use of it can make the detector learn the preprocessing
    itself instead of the underlying image evidence.
    """

    out = np.empty_like(image)
    for channel in range(3):
        x = image[:, :, channel].astype(np.float32)
        p_min = float(x.min())
        p_max = float(x.max())
        if math.isclose(p_min, p_max):
            out[:, :, channel] = image[:, :, channel]
        else:
            out[:, :, channel] = ensure_uint8(
                (x - p_min) * 255.0 / (p_max - p_min)
            )
    return out


# ---------------------------------------------------------------------------
# Lecture-derived local filtering / noise operations
# ---------------------------------------------------------------------------


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing with a kernel sized from sigma.

    The lecture emphasizes that sigma controls the amount/scale of smoothing
    and that the discrete kernel should be large enough to contain the useful
    Gaussian support. We choose an odd kernel width near 6*sigma.
    """

    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    kernel_size = max(3, int(math.ceil(6.0 * sigma)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a median filter to suppress impulse / salt-and-pepper noise."""

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("median kernel_size must be an odd integer >= 3")
    return cv2.medianBlur(image, kernel_size)


def unsharp_mask(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Sharpen using the lecture's detail-extraction idea.

    detail = original - blurred
    sharpened = original + alpha * detail

    The lecture notes explicitly warn that sharpening can also emphasize
    noise, so this is kept as a *mild* optional augmentation.
    """

    blurred = gaussian_blur(image, sigma)
    original = image.astype(np.float32)
    details = original - blurred.astype(np.float32)
    out = original + float(amount) * details
    return ensure_uint8(out)


def gaussian_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise to simulate sensor/noise variation."""

    noise = rng.normal(0.0, std * 255.0, size=image.shape)
    return ensure_uint8(image.astype(np.float32) + noise)


def salt_pepper_noise(
    image: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add sparse impulse noise, matching the lecture's salt-and-pepper case."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")

    out = image.copy()
    mask = rng.random(image.shape[:2])
    out[mask < probability / 2] = 0
    out[mask > 1 - probability / 2] = 255
    return out


# ---------------------------------------------------------------------------
# JPEG / spatial-resolution operations
# ---------------------------------------------------------------------------


def jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode/decode JPEG to reproduce lossy compression effects.

    The lecture explains that JPEG uses 8x8 blocks and quantized DCT
    coefficients, producing irreversible detail loss and artifacts such as
    blocking and ringing at stronger compression.
    """

    quality = int(np.clip(quality, 1, 100))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise IOError("JPEG encoding failed")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise IOError("JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def downscale_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Reduce spatial resolution and then restore the original dimensions."""

    if not 0 < scale < 1:
        raise ValueError("scale must be in (0, 1)")

    height, width = image.shape[:2]
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))

    # INTER_AREA is a sensible downsampling choice for reducing resolution.
    # The original lecture highlights spatial resolution and resampling as
    # important sources of information loss.
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return restored


# ---------------------------------------------------------------------------
# Colour-space operation inspired by the lecture's RGB/HSV discussion
# ---------------------------------------------------------------------------


def hsv_perturb(
    image: np.ndarray,
    hue_shift: float,
    saturation_scale: float,
    value_scale: float,
) -> np.ndarray:
    """Perturb hue, saturation and value in HSV space.

    HSV is useful because it separates colour-related quantities from value.
    The lecture notes also caution that HSV does not create new information
    and is not automatically robust to lighting, so this is deliberately mild.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 90.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Additional geometric operations
# ---------------------------------------------------------------------------


def center_crop_resize(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Crop the central fraction of an image and resize back to original size."""

    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in (0, 1]")

    height, width = image.shape[:2]
    crop_h = max(1, int(round(height * crop_fraction)))
    crop_w = max(1, int(round(width * crop_fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def mild_affine(image: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Apply a small rotation/scale change without changing the canvas size."""

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ---------------------------------------------------------------------------
# Transformation registry
# ---------------------------------------------------------------------------


def make_transformations(
    rng: np.random.Generator,
) -> dict[str, tuple[Callable[[np.ndarray], np.ndarray], dict]]:
    """Build the deterministic transformation catalogue for one image."""

    # Randomly sample parameters once per input image. This makes the output
    # reproducible with --seed while still giving varied training examples.
    jpeg_quality = int(rng.integers(30, 91))
    blur_sigma = float(rng.uniform(0.5, 2.0))
    downscale = float(rng.choice([0.25, 0.5]))
    noise_std = float(rng.uniform(0.02, 0.10))
    gamma = float(rng.uniform(0.65, 1.45))
    brightness = int(rng.integers(-32, 33))
    contrast = float(rng.uniform(0.8, 1.2))
    hue_shift = float(rng.uniform(-0.10, 0.10))
    saturation = float(rng.uniform(0.8, 1.2))
    value = float(rng.uniform(0.8, 1.2))
    affine_angle = float(rng.uniform(-5.0, 5.0))
    affine_scale = float(rng.uniform(0.95, 1.05))

    return {
        "jpeg": (
            lambda x: jpeg_recompress(x, jpeg_quality),
            {"quality": jpeg_quality},
        ),
        "blur": (
            lambda x: gaussian_blur(x, blur_sigma),
            {"sigma": round(blur_sigma, 4)},
        ),
        "resize": (
            lambda x: downscale_upscale(x, downscale),
            {"scale": downscale},
        ),
        "gaussian_noise": (
            lambda x: gaussian_noise(x, noise_std, rng),
            {"std": round(noise_std, 4)},
        ),
        "salt_pepper": (
            lambda x: salt_pepper_noise(x, 0.005, rng),
            {"probability": 0.005},
        ),
        "median": (
            lambda x: median_filter(x, 3),
            {"kernel_size": 3},
        ),
        "sharpen": (
            lambda x: unsharp_mask(x, sigma=1.0, amount=0.5),
            {"sigma": 1.0, "amount": 0.5},
        ),
        "brightness": (
            lambda x: adjust_brightness(x, brightness),
            {"offset": brightness},
        ),
        "contrast": (
            lambda x: adjust_contrast(x, contrast),
            {"scale": round(contrast, 4)},
        ),
        "gamma": (
            lambda x: gamma_map(x, gamma),
            {"gamma": round(gamma, 4)},
        ),
        "histogram_stretch": (
            histogram_stretch,
            {},
        ),
        "hsv": (
            lambda x: hsv_perturb(x, hue_shift, saturation, value),
            {
                "hue_shift": round(hue_shift, 4),
                "saturation_scale": round(saturation, 4),
                "value_scale": round(value, 4),
            },
        ),
        "center_crop": (
            lambda x: center_crop_resize(x, 0.80),
            {"crop_fraction": 0.80},
        ),
        "affine": (
            lambda x: mild_affine(x, affine_angle, affine_scale),
            {"angle_deg": round(affine_angle, 3), "scale": round(affine_scale, 4)},
        ),
    }


# ---------------------------------------------------------------------------
# Random composition strategy
# ---------------------------------------------------------------------------


def random_composite(
    image: np.ndarray,
    catalogue: dict[str, tuple[Callable[[np.ndarray], np.ndarray], dict]],
    rng: np.random.Generator,
    count: int,
) -> list[tuple[np.ndarray, str, dict]]:
    """Generate random but controlled compositions of 2-4 operations.

    We avoid composing every operation. Extremely destructive combinations can
    teach the classifier unrealistic shortcuts rather than robustness.
    """

    names = list(catalogue)
    results: list[tuple[np.ndarray, str, dict]] = []

    # Groups make the compositions more realistic and reduce destructive
    # combinations such as blur + median + resize + heavy JPEG every time.
    spatial = ["resize", "center_crop", "affine"]
    appearance = ["brightness", "contrast", "gamma", "hsv", "histogram_stretch"]
    degradation = ["jpeg", "blur", "gaussian_noise", "salt_pepper"]

    for _ in range(count):
        selected: list[str] = []
        selected.append(str(rng.choice(spatial)))
        selected.append(str(rng.choice(appearance)))
        selected.append(str(rng.choice(degradation)))

        # 25% chance of a fourth mild operation.
        if rng.random() < 0.25:
            extra = str(rng.choice(["sharpen", "median", "brightness", "contrast"]))
            selected.append(extra)

        # Randomize application order so the model does not see one fixed
        # preprocessing sequence.
        rng.shuffle(selected)

        out = image.copy()
        parameter_log: dict[str, dict] = {}
        for name in selected:
            fn, params = catalogue[name]
            out = fn(out)
            parameter_log[name] = params

        description = "_".join(selected)
        results.append((out, description, parameter_log))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate robustness-augmented variants from one real or AI image "
            "for TrueSight Member 1 training."
        )
    )
    parser.add_argument("--input", required=True, help="Path to one input image")
    parser.add_argument(
        "--label",
        required=True,
        choices=["real", "ai"],
        help="Ground-truth class of the input image: real or ai",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which transformed images and manifest.json are saved",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["all"],
        help=(
            "Transform names to generate. Use 'all' for the complete catalogue. "
            "Examples: jpeg blur resize gaussian_noise gamma median sharpen"
        ),
    )
    parser.add_argument(
        "--random-variants",
        type=int,
        default=0,
        help="Number of random 3-4-operation compositions to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible parameters",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Also copy the original image into the output set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(input_path)
    rng = np.random.default_rng(args.seed)
    catalogue = make_transformations(rng)

    requested = set(args.transforms)
    if "all" in requested:
        selected_names = list(catalogue.keys())
    else:
        unknown = requested.difference(catalogue.keys())
        if unknown:
            available = ", ".join(sorted(catalogue))
            raise ValueError(
                f"Unknown transformations: {sorted(unknown)}. "
                f"Available: {available}"
            )
        selected_names = [name for name in catalogue if name in requested]

    records: list[OutputRecord] = []
    label_id = 1 if args.label == "ai" else 0
    source_name = input_path.name

    if args.include_original:
        original_path = output_dir / f"original_{args.label}.jpg"
        write_image(original_path, image, jpeg_quality=95)
        records.append(
            OutputRecord(
                image_path=str(original_path),
                label=label_id,
                label_name=args.label,
                source_image=str(input_path),
                transformation="original",
                parameters={},
                seed=args.seed,
            )
        )

    # Generate each selected transformation independently. Independent variants
    # are easier to diagnose in robustness experiments than only using chained
    # transformations.
    for name in selected_names:
        fn, params = catalogue[name]
        transformed = fn(image.copy())
        output_path = output_dir / f"{Path(source_name).stem}__{name}.jpg"
        write_image(output_path, transformed, jpeg_quality=95)
        records.append(
            OutputRecord(
                image_path=str(output_path),
                label=label_id,
                label_name=args.label,
                source_image=str(input_path),
                transformation=name,
                parameters=params,
                seed=args.seed,
            )
        )

    # Optional random compositions provide diversity for training without
    # requiring every possible transformation combination to be saved.
    if args.random_variants > 0:
        compositions = random_composite(image, catalogue, rng, args.random_variants)
        for index, (transformed, description, params) in enumerate(compositions, start=1):
            output_path = output_dir / (
                f"{Path(source_name).stem}__random_{index:03d}__{description}.jpg"
            )
            write_image(output_path, transformed, jpeg_quality=95)
            records.append(
                OutputRecord(
                    image_path=str(output_path),
                    label=label_id,
                    label_name=args.label,
                    source_image=str(input_path),
                    transformation=f"random_composite:{description}",
                    parameters=params,
                    seed=args.seed,
                )
            )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "source_image": str(input_path),
        "label": label_id,
        "label_name": args.label,
        "seed": args.seed,
        "num_outputs": len(records),
        "outputs": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Input:       {input_path}")
    print(f"Label:       {args.label} ({label_id})")
    print(f"Output dir:  {output_dir}")
    print(f"Generated:   {len(records)} images")
    print(f"Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
