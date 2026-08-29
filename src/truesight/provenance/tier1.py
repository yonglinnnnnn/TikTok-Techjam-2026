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

AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositedwithtrainedalgorithmicmedia")
CAPTURE_SOURCE_TYPES = ("digitalcapture",)

def _source_has(c2pa: dict[str, Any], values: tuple[str, ...]) -> bool:
    origin_type = str((c2pa.get("history") or {}).get("origin_type") or "").lower()
    return any(value in origin_type for value in values)

def _weight(c2pa: dict[str, Any], metadata: dict[str, Any]) -> tuple[float, list[str]]:
    evidence: list[str] = []
    score = 0.0
    if c2pa["verified"] and _source_has(c2pa, AI_SOURCE_TYPES):
        score = .98
        evidence.append("trusted C2PA history declares trained-algorithmic media")
    elif c2pa["present"] and _source_has(c2pa, AI_SOURCE_TYPES):
        score = .35
        evidence.append("C2PA history declares trained-algorithmic media but is not trusted")
    elif c2pa["present"]:
        evidence.append("C2PA provenance present without a recognized AI claim")
    if metadata["ai_markers"]:
        score = max(score, .30)
        evidence.append("unverified AI-related metadata marker detected")
    return score, evidence

def _signals(c2pa: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize heterogeneous checks for fusion, logging and UI rendering."""
    metadata_score = .30 if metadata["ai_markers"] else 0.0
    ai_claim = _source_has(c2pa, AI_SOURCE_TYPES)
    c2pa_score = .98 if c2pa["verified"] and ai_claim else (
        .35 if c2pa["present"] and ai_claim else 0.0)
    return [
        {"kind": "content_credential", "provider": "c2pa",
         "status": c2pa["status"], "present": c2pa["present"],
         "verified": c2pa["verified"], "ai_claim": ai_claim,
         "source": (c2pa.get("history") or {}).get("origin_type"),
         "evidence_score": c2pa_score},
        {"kind": "embedded_metadata", "provider": "exif_xmp",
         "status": "observed" if metadata["checked"] else "error",
         "present": bool(metadata["metadata"]), "verified": False,
         "ai_claim": bool(metadata["ai_markers"]),
         "source": metadata["ai_markers"][0] if metadata["ai_markers"] else None,
         "evidence_score": metadata_score},
    ]

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

def analyze_tier1(image_path: str | Path,
                  output_dir: str | Path | None = None,
                  run_forensics: bool = True,
                  debug_forensics: bool = False,
                  debug_provenance: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    file_info = inspect_file(path)
    c2pa, metadata = check_c2pa(path), check_metadata(path)
    forensics = (analyze_blind_forensics(path, output_dir, debug_forensics)
                 if run_forensics else None)
    score, evidence = _weight(c2pa, metadata)
    signals = _signals(c2pa, metadata)
    history = c2pa.get("history") or {}
    verified_ai = bool(c2pa["verified"] and _source_has(c2pa, AI_SOURCE_TYPES))
    verified_capture = bool(c2pa["verified"] and _source_has(c2pa, CAPTURE_SOURCE_TYPES)
                            and not history.get("content_edited")
                            and not history.get("transformed")
                            and not history.get("has_unclassified_actions"))
    source = history.get("origin_type") or (
        metadata["ai_markers"][0] if metadata["ai_markers"] else None)
    status = "verified_ai" if verified_ai else "verified_capture" if verified_capture else (
        "ai_indicated" if score > 0 else "no_provenance_found")
    if forensics and forensics["candidate_regions"]:
        evidence.append("deterministic forensics produced candidate regions for downstream review")
    result = {"image_path": path.as_posix(),
        "is_ai_generated": True if verified_ai else False if verified_capture else None,
        "confidence": .98 if verified_ai else .95 if verified_capture else None,
        "source": source, "ai_coverage": None,
        "heatmap_path": None,
        "provenance": {"status": status,
            "conclusion": ("verified AI provenance" if verified_ai else
                "trusted digital-capture origin with no declared content edits" if verified_capture else
                "provenance signals are suggestive but unverified" if score > 0 else
                "no supported provenance signal found; this does not establish authenticity"),
            "signals": signals, "file": file_info,
            "c2pa": _public_c2pa(c2pa), "metadata": _public_metadata(metadata)},
        "forensics": forensics,
        "evidence": evidence, "latency_ms": round((time.perf_counter()-start)*1000),
        "tier1": {"watermark_detected": None,
                  "provenance_verified": verified_ai or verified_capture,
                  "severity_weight": round(score, 3),
                  "forensic_integrity_weight": (forensics["integrity_risk_weight"]
                                                  if forensics else 0.0),
                  "requires_tier2": not (verified_ai or verified_capture)}}

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
    args = parser.parse_args()
    print(json.dumps(analyze_tier1(args.image, args.output_dir,
                                   not args.skip_forensics, args.debug_forensics,
                                   args.debug_provenance),
                     indent=2, default=str))

if __name__ == "__main__":
    main()
