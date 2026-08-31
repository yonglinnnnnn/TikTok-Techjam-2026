"""Tier 1 aggregator returning the pipeline-wide result contract."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any
from .forensics import analyze_blind_forensics
from .c2pa_checker import check_c2pa
from .file_inspector import inspect_file
from .metadata_checker import check_metadata
from .scoring import score_signals
from .signals import (aggregate_signals, best_ai_source, has_verified_ai_signal,
                      has_verified_capture_signal, watermark_detection)

def _public_c2pa(c2pa: dict[str, Any]) -> dict[str, Any]:
    """Keep the verified history, but omit potentially large SDK diagnostics."""
    keys = ("checked", "status", "present", "verified", "active_manifest",
            "validation_state", "history", "error")
    return {key: c2pa.get(key) for key in keys}

def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "checked": metadata["checked"],
        "present": bool(metadata["metadata"]),
        "ai_markers": metadata["ai_markers"],
        "error": metadata["error"],
    }

def _evidence(signals: list[dict[str, Any]], verified_capture: bool) -> list[str]:
    evidence: list[str] = []
    for signal in signals:
        if signal["verified"] and signal["ai_claim"]:
            evidence.append(
                f"{signal['provider']} verified an AI-attributed {signal['kind']}")
        elif signal["present"] and signal["ai_claim"]:
            evidence.append(
                f"{signal['provider']} reported an unverified AI-attributed signal")
        elif signal["kind"] == "content_credential" and signal["status"] == "invalid":
            evidence.append(f"{signal['provider']} content credential failed validation")
    if verified_capture:
        evidence.append("trusted C2PA history declares digital capture with no declared edits")
    return evidence

def analyze_tier1(image_path: str | Path,
                  output_dir: str | Path | None = None,
                  run_forensics: bool = True,
                  debug_forensics: bool = False,
                  debug_provenance: bool = False,
                  calibration_path: str | Path | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    file_info = inspect_file(path)
    c2pa, metadata = check_c2pa(path), check_metadata(path)
    forensics = (analyze_blind_forensics(path, output_dir, debug_forensics)
                 if run_forensics else None)
    signals, severity = score_signals(
        aggregate_signals(c2pa, metadata), calibration_path)
    verified_ai = has_verified_ai_signal(signals)
    verified_capture = has_verified_capture_signal(signals, c2pa)
    provenance_verified = any(signal["verified"] for signal in signals)
    source = best_ai_source(signals) or (
        (c2pa.get("history") or {}).get("origin_type") if verified_capture else None)
    evidence = _evidence(signals, verified_capture)
    provenance_detected = any(signal["present"] is True for signal in signals)
    status = "verified_ai" if verified_ai else "verified_capture" if verified_capture else (
        "ai_indicated" if severity["score"] > 0 else
        "invalid" if any(signal["status"] == "invalid" for signal in signals) else
        "detected" if provenance_detected else
        "no_provenance_found")
    if forensics and forensics["candidate_regions"]:
        evidence.append("deterministic forensics produced candidate regions for downstream review")
    result = {"schema_version": "1.0",
        "image_path": path.as_posix(),
        "is_ai_generated": True if verified_ai else None,
        "confidence": severity["score"] if verified_ai else None,
        "source": source, "ai_coverage": None,
        "heatmap_path": None,
        "provenance": {"status": status,
            "conclusion": ("verified AI provenance" if verified_ai else
                "trusted digital-capture claim found; downstream analysis still required" if verified_capture else
                "provenance signals are suggestive but unverified" if severity["score"] > 0 else
                "a content credential is present but invalid" if status == "invalid" else
                "a verified content credential was found, but it has no supported AI or capture claim" if status == "detected" and provenance_verified else
                "a provenance signal was detected but is not verified" if status == "detected" else
                "no supported provenance signal found; this does not establish authenticity"),
            "signals": signals, "file": file_info,
            "c2pa": _public_c2pa(c2pa), "metadata": _public_metadata(metadata)},
        "forensics": forensics,
        "evidence": evidence, "latency_ms": round((time.perf_counter()-start)*1000),
        "tier1": {"watermark_detected": watermark_detection(signals),
                  "provenance_detected": provenance_detected,
                  "provenance_verified": provenance_verified,
                  "verified_ai_signal": verified_ai,
                  "verified_capture_signal": verified_capture,
                  "severity_weight": severity["score"],
                  "severity_calibration": severity,
                  "forensic_integrity_weight": (forensics["integrity_risk_weight"]
                                                  if forensics else 0.0),
                  # Only a trusted AI-attributed C2PA credential may use the fast
                  # path. Verified capture remains supporting evidence.
                  "requires_tier2": not verified_ai},
        "tier2": None, "tier3": None, "fusion": None}

    if debug_provenance:
        result["debug_provenance"] = {
            "c2pa_validation_results": c2pa.get("validation_results"),
            "c2pa_manifest": c2pa.get("manifest"),
            "metadata_fields": metadata.get("metadata"),
        }
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-forensics", action="store_true")
    parser.add_argument("--debug-forensics", action="store_true")
    parser.add_argument("--debug-provenance", action="store_true")
    parser.add_argument("--calibration")
    args = parser.parse_args()
    print(json.dumps(analyze_tier1(args.image, args.output_dir,
                                   not args.skip_forensics, args.debug_forensics,
                                   args.debug_provenance,
                                   args.calibration),
                     indent=2, default=str))

if __name__ == "__main__":
    main()
