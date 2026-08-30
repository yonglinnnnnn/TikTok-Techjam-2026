"""Deterministic forensic features for downstream VLM/model analysis.

This module needs neither an original image nor a learned model. Its outputs are
supporting integrity signals, not a standalone AI-generation verdict.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or corrupt image: {path}")
    if min(image.shape[:2]) < 32:
        raise ValueError("Image must be at least 32x32 pixels")
    return image


def _robust_unit(values: np.ndarray, low: float = 50, high: float = 99) -> np.ndarray:
    array = np.nan_to_num(values.astype(np.float32))
    lo, hi = np.percentile(array, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0, 1)


def _ela_map(image: np.ndarray, quality: int = 90) -> tuple[np.ndarray, dict[str, Any]]:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG recompression failed")
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    difference = cv2.absdiff(image, recompressed).astype(np.float32)
    raw = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    heatmap = _robust_unit(cv2.GaussianBlur(raw, (0, 0), 1.0), 60, 99.5)
    return heatmap, {
        "kind": "recompression_error",
        "method": "jpeg_ela",
        "recompression_quality": quality,
        "raw_mean": round(float(raw.mean()), 4),
        "raw_p99": round(float(np.percentile(raw, 99)), 4),
        "interpretation": "supporting evidence only; ordinary JPEG history can cause the same response",
    }


def _noise_map(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.1)
    residual = gray - smooth
    local_energy = cv2.boxFilter(residual * residual, -1, (31, 31), normalize=True)
    log_energy = np.log(local_energy + 1e-7)
    median = float(np.median(log_energy))
    mad = float(np.median(np.abs(log_energy - median))) + 1e-6
    anomaly = np.abs(log_energy - median) / (1.4826 * mad)

    # Reduce the tendency to mark every strong semantic edge as a noise anomaly.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = _robust_unit(cv2.magnitude(gx, gy), 70, 99)
    anomaly *= 1.0 - 0.35 * edge
    heatmap = _robust_unit(cv2.GaussianBlur(anomaly, (0, 0), 2.0), 65, 99)
    return heatmap, {
        "kind": "noise_inconsistency",
        "method": "local_high_pass_energy",
        "robust_peak_z": round(float(np.percentile(anomaly, 99)), 4),
        "interpretation": "different local processing/noise may indicate splicing or inpainting",
    }


def _jpeg_grid_map(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    vertical = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    height, width = gray.shape
    grid = np.zeros_like(gray, dtype=np.float32)
    grid[:, 8::8] = horizontal[:, 8::8]
    grid[8::8, :] += vertical[8::8, :]
    grid_density = cv2.boxFilter(grid, -1, (33, 33), normalize=True)
    texture = cv2.boxFilter(horizontal + vertical, -1, (33, 33), normalize=True)
    ratio = grid_density / (texture + 1e-3)
    median = float(np.median(ratio))
    anomaly = np.abs(ratio - median)
    heatmap = _robust_unit(cv2.GaussianBlur(anomaly, (0, 0), 2.0), 65, 99)
    return heatmap, {
        "kind": "jpeg_grid_inconsistency",
        "method": "local_8x8_boundary_ratio",
        "grid_ratio_median": round(median, 6),
        "grid_ratio_p99": round(float(np.percentile(ratio, 99)), 6),
        "interpretation": "local JPEG-grid changes may indicate pasted or recompressed regions",
    }


def _copy_move_map(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    heatmap = np.zeros(gray.shape, dtype=np.float32)
    accepted = 0
    if descriptors is not None and len(keypoints) >= 4:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, descriptors, k=3)
        min_separation = max(24.0, min(gray.shape) * 0.06)
        for index, candidates in enumerate(matches):
            chosen = None
            for candidate in candidates:
                if candidate.trainIdx == index:
                    continue
                p1 = np.asarray(keypoints[index].pt)
                p2 = np.asarray(keypoints[candidate.trainIdx].pt)
                if np.linalg.norm(p1 - p2) >= min_separation:
                    chosen = candidate
                    break
            if chosen is None or chosen.distance > 28:
                continue
            p1 = tuple(np.rint(keypoints[index].pt).astype(int))
            p2 = tuple(np.rint(keypoints[chosen.trainIdx].pt).astype(int))
            cv2.circle(heatmap, p1, 12, 1.0, -1)
            cv2.circle(heatmap, p2, 12, 1.0, -1)
            accepted += 1
    if accepted:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), 7.0)
        heatmap /= max(float(heatmap.max()), 1e-6)
    return heatmap, {
        "kind": "copy_move_similarity",
        "method": "spatially_separated_orb_matches",
        "keypoints": len(keypoints),
        "accepted_matches": accepted,
        "interpretation": "repeated natural patterns can create false matches",
    }


def _candidate_mask(maps: list[np.ndarray], image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = np.asarray([0.34, 0.40, 0.20, 0.06], dtype=np.float32)
    stack = np.stack(maps)
    combined = np.average(stack, axis=0, weights=weights)
    votes = np.sum(stack[:3] >= 0.72, axis=0)
    # Always return a small set of review candidates. This is localization aid,
    # not a positive verdict: authentic textured regions can also rank highly.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clipped = ((gray <= 5) | (gray >= 250)).astype(np.float32)
    clipped_neighborhood = cv2.boxFilter(clipped, -1, (41, 41), normalize=True)
    valid = clipped_neighborhood < 0.12
    margin_y = max(2, int(combined.shape[0] * .015))
    margin_x = max(2, int(combined.shape[1] * .015))
    valid[:margin_y, :] = valid[-margin_y:, :] = False
    valid[:, :margin_x] = valid[:, -margin_x:] = False
    valid_values = combined[valid]
    threshold = max(0.38, float(np.percentile(valid_values, 98.5)))
    candidate = np.where((combined >= threshold) & valid, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    minimum_area = max(24, int(candidate.size * 0.0001))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    cleaned = np.zeros_like(candidate)
    regions: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label].tolist()
        if area < minimum_area:
            continue
        selected = labels == label
        cleaned[selected] = 255
        regions.append({"x": x, "y": y, "width": width, "height": height,
                        "area_pixels": area,
                        "anomaly_mean": round(float(combined[selected].mean()), 4),
                        "anomaly_peak": round(float(combined[selected].max()), 4),
                        "signal_support": round(float(votes[selected].mean()), 4),
                        "normalized_box": {
                            "x": round(x / combined.shape[1], 6),
                            "y": round(y / combined.shape[0], 6),
                            "width": round(width / combined.shape[1], 6),
                            "height": round(height / combined.shape[0], 6),
                        }})
    regions.sort(key=lambda item: item["area_pixels"], reverse=True)
    image_height, image_width = combined.shape
    for index, region in enumerate(regions, 1):
        left, top = region["x"], region["y"]
        right = left + region["width"] - 1
        bottom = top + region["height"] - 1
        center_x = left + region["width"] / 2
        center_y = top + region["height"] / 2
        horizontal = "left" if center_x < image_width / 3 else (
            "right" if center_x > image_width * 2 / 3 else "center")
        vertical = "upper" if center_y < image_height / 3 else (
            "lower" if center_y > image_height * 2 / 3 else "middle")
        box_area = region["width"] * region["height"]
        region.update({
            "id": f"R{index}",
            "location": f"{vertical}-{horizontal}",
            "box_pixels": {"left": left, "top": top, "right": right,
                           "bottom": bottom},
            "center_pixels": {"x": round(center_x, 1), "y": round(center_y, 1)},
            "box_percent": {
                "left": round(left / image_width * 100, 2),
                "top": round(top / image_height * 100, 2),
                "right": round((right + 1) / image_width * 100, 2),
                "bottom": round((bottom + 1) / image_height * 100, 2),
            },
            "box_area_pixels": box_area,
            "mask_fill_ratio": round(region["area_pixels"] / box_area, 4),
        })
    coverage = float(np.count_nonzero(cleaned) / cleaned.size)
    support = float(np.mean(votes[cleaned > 0])) if np.any(cleaned) else 0.0
    heuristic_score = float(np.clip(0.45 * np.percentile(combined, 99)
                                    + 0.35 * min(support / 3.0, 1.0)
                                    + 0.20 * min(coverage / 0.05, 1.0), 0, 1))
    return combined, cleaned, {
        "regions": regions[:20],
        "coverage": coverage,
        "candidate_threshold": threshold,
        "consensus_support": support,
        "heuristic_score": heuristic_score,
    }


def _save_gray(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.clip(values * 255, 0, 255).astype(np.uint8))[1].tofile(path)


def _labeled(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(canvas, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .72, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _review_artifacts(image: np.ndarray, mask: np.ndarray,
                      combined: np.ndarray,
                      regions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    annotated = image.copy()
    for index, region in enumerate(regions, 1):
        x, y = region["x"], region["y"]
        right, bottom = x + region["width"], y + region["height"]
        cv2.rectangle(annotated, (x, y), (right, bottom), (0, 0, 255), 2)
        text = f"{region['id']}  peak={region['anomaly_peak']:.2f}"
        text_y = max(18, y - 7)
        cv2.putText(annotated, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 255), 2, cv2.LINE_AA)

    mask_overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    selected = mask > 0
    mask_overlay[selected] = cv2.addWeighted(image, .35, red, .65, 0)[selected]
    for index, region in enumerate(regions, 1):
        x, y = region["x"], region["y"]
        cv2.putText(mask_overlay, region["id"], (x, max(18, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2, cv2.LINE_AA)

    heat = cv2.applyColorMap(np.clip(combined * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    heat_overlay = cv2.addWeighted(image, .62, heat, .38, 0)
    panels = [_labeled(annotated, "1. Numbered candidate boxes"),
              _labeled(mask_overlay, "2. Binary candidate-mask overlay"),
              _labeled(heat_overlay, "3. Continuous anomaly heatmap")]
    panel_width = min(560, image.shape[1])
    resized = [cv2.resize(panel, (panel_width,
               max(1, round(panel.shape[0] * panel_width / panel.shape[1]))))
               for panel in panels]
    review_panel = np.hstack(resized)

    crop_width, crop_height = 240, 190
    crop_tiles: list[np.ndarray] = []
    for index, region in enumerate(regions[:12], 1):
        padding = max(20, round(max(region["width"], region["height"]) * .35))
        x1, y1 = max(0, region["x"] - padding), max(0, region["y"] - padding)
        x2 = min(image.shape[1], region["x"] + region["width"] + padding)
        y2 = min(image.shape[0], region["y"] + region["height"] + padding)
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (0, 0), (crop_width - 1, crop_height - 1), (0, 0, 255), 2)
        cv2.putText(crop, f"{region['id']}  mean={region['anomaly_mean']:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2, cv2.LINE_AA)
        crop_tiles.append(crop)
    if crop_tiles:
        columns = min(4, len(crop_tiles))
        rows = (len(crop_tiles) + columns - 1) // columns
        blank = np.full((crop_height, crop_width, 3), 32, dtype=np.uint8)
        crop_tiles += [blank] * (rows * columns - len(crop_tiles))
        crop_sheet = np.vstack([np.hstack(crop_tiles[row * columns:(row + 1) * columns])
                                for row in range(rows)])
    else:
        crop_sheet = np.full((190, 480, 3), 32, dtype=np.uint8)
        cv2.putText(crop_sheet, "No candidate regions", (110, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    return {"annotated_regions": annotated, "mask_overlay": mask_overlay,
            "review_panel": review_panel, "region_crops": crop_sheet}


def _save_outputs(output_dir: Path, stem: str, image: np.ndarray,
                  maps: dict[str, np.ndarray], mask: np.ndarray,
                  regions: list[dict[str, Any]],
                  debug_artifacts: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    debug_paths: dict[str, str] = {}
    mask_path = output_dir / f"{stem}_candidate_mask.png"
    cv2.imencode(".png", mask)[1].tofile(mask_path)
    paths["candidate_mask"] = _portable_path(mask_path)

    color = cv2.applyColorMap(np.clip(maps["combined"] * 255, 0, 255).astype(np.uint8),
                              cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.62, color, 0.38, 0)
    overlay_path = output_dir / f"{stem}_forensics_overlay.png"
    cv2.imencode(".png", overlay)[1].tofile(overlay_path)
    paths["vlm_overlay"] = _portable_path(overlay_path)
    review = _review_artifacts(image, mask, maps["combined"], regions)
    review_path = output_dir / f"{stem}_review_panel.png"
    cv2.imencode(".png", review["review_panel"])[1].tofile(review_path)
    paths["human_review"] = _portable_path(review_path)
    if debug_artifacts:
        for name, values in maps.items():
            map_path = output_dir / f"{stem}_{name}.png"
            _save_gray(map_path, values)
            debug_paths[name] = _portable_path(map_path)
        for name, artifact in review.items():
            if name == "review_panel":
                continue
            artifact_path = output_dir / f"{stem}_{name}.png"
            cv2.imencode(".png", artifact)[1].tofile(artifact_path)
            debug_paths[name] = _portable_path(artifact_path)
    return paths, debug_paths


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    project_root = Path(__file__).resolve().parents[3]
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _public_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": region["id"],
        "location": region["location"],
        "box_pixels": region["box_pixels"],
        "box_normalized": {
            "left": region["normalized_box"]["x"],
            "top": region["normalized_box"]["y"],
            "right": round(region["normalized_box"]["x"]
                           + region["normalized_box"]["width"], 6),
            "bottom": round(region["normalized_box"]["y"]
                            + region["normalized_box"]["height"], 6),
        },
        "anomaly_mean": region["anomaly_mean"],
        "signal_support": region["signal_support"],
    } for region in regions]


def _image_statistics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return {"clipped_pixel_ratio": round(clipped, 6),
            "laplacian_variance": round(laplacian_variance, 4)}


def analyze_blind_forensics(image_path: str | Path,
                            output_dir: str | Path | None = None,
                            debug_artifacts: bool = False) -> dict[str, Any]:
    """Analyze one image without an original/reference image."""
    started = time.perf_counter()
    path = Path(image_path)
    image = _read_image(path)
    ela, ela_signal = _ela_map(image)
    noise, noise_signal = _noise_map(image)
    grid, grid_signal = _jpeg_grid_map(image)
    copy_move, copy_signal = _copy_move_map(image)
    combined, candidate, summary = _candidate_mask([ela, noise, grid, copy_move], image)

    maps = {"ela": ela, "noise": noise, "jpeg_grid": grid,
            "copy_move": copy_move, "combined": combined}
    resolved_output = ((Path(output_dir).resolve()) if output_dir is not None else
                       Path(__file__).resolve().parents[3] / "outputs" / "forensics")
    paths, debug_paths = _save_outputs(resolved_output, path.stem, image, maps,
                                       candidate, summary["regions"], debug_artifacts)
    statistics = _image_statistics(image)
    feature_vector = {
        "ela_p99": ela_signal["raw_p99"],
        "noise_peak_robust_z": noise_signal["robust_peak_z"],
        "jpeg_grid_ratio_p99": grid_signal["grid_ratio_p99"],
        "copy_move_matches": copy_signal["accepted_matches"],
        "consensus_support": round(summary["consensus_support"], 4),
        "clipped_pixel_ratio": statistics["clipped_pixel_ratio"],
        "sharpness_laplacian_variance": statistics["laplacian_variance"],
    }
    # Cap deterministic forensics at weak evidence. It can prioritize escalation
    # but cannot determine AI origin or tampering without downstream analysis.
    integrity_weight = round(min(0.25, summary["heuristic_score"] * 0.25), 4)

    result = {
        "integrity_risk_weight": integrity_weight,
        "candidate_coverage": round(summary["coverage"], 6),
        "candidate_regions": _public_regions(summary["regions"]),
        "feature_vector": feature_vector,
        "artifacts": paths,
    }
    if debug_artifacts:
        result["debug"] = {
            "heuristic_score": round(summary["heuristic_score"], 4),
            "candidate_threshold": round(summary["candidate_threshold"], 4),
            "signals": [ela_signal, noise_signal, grid_signal, copy_signal],
            "artifacts": debug_paths,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Blind image-forensics analysis")
    parser.add_argument("image")
    parser.add_argument("--output-dir")
    parser.add_argument("--debug-artifacts", action="store_true")
    args = parser.parse_args()
    result = analyze_blind_forensics(args.image, args.output_dir, args.debug_artifacts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
