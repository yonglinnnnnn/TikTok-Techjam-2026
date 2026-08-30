"""Optional OpenAI image-provenance API adapter.

The original file bytes are uploaded unchanged. Missing or undetected signals
remain inconclusive; this checker never treats them as evidence of authenticity.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    fields = ("type", "outcome", "model", "issuer", "generated_at",
              "validation_state")
    return {field: getattr(value, field, None) for field in fields}


def check_openai_provenance(image_path: str | Path,
                            client: Any | None = None,
                            api_key: str | None = None,
                            timeout_seconds: float = 30.0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": False,
        "status": "not_checked",
        "present": None,
        "verified": False,
        "source": None,
        "results": [],
        "error": None,
    }
    load_dotenv()  
    if client is None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        print(os.getenv('OPENAI_API_KEY'))
        if not key:
            result.update(status="unavailable",
                          error="OPENAI_API_KEY is not configured")
            return result
        try:
            from openai import OpenAI
        except ImportError:
            result.update(status="unavailable",
                          error="openai package is not installed")
            return result
        client = OpenAI(api_key=key, timeout=timeout_seconds, max_retries=2)

    try:
        payload = Path(image_path).read_bytes()
        response = client.content_provenance_checks.create(file=payload)
        normalized: list[dict[str, Any]] = []
        for item in getattr(response, "results", []) or []:
            raw = _as_dict(item)
            signal_type = str(raw.get("type") or "unknown").lower()
            outcome = str(raw.get("outcome") or "not_detected").lower()
            validation = raw.get("validation_state")
            normalized.append({
                "type": signal_type,
                "outcome": outcome,
                "model": raw.get("model"),
                "issuer": raw.get("issuer"),
                "generated_at": raw.get("generated_at"),
                "validation_state": str(validation).lower() if validation else None,
            })
        detected = [item for item in normalized if item["outcome"] == "detected"]
        source = next((item["model"] for item in detected if item["model"]), None)
        result.update(
            checked=True,
            status="detected" if detected else "not_detected",
            present=bool(detected),
            # The normalized signal aggregator decides verification per signal:
            # trusted for C2PA and provider-confirmed for SynthID.
            verified=any(item["type"] == "synthid" or
                         item["validation_state"] == "trusted"
                         for item in detected),
            source=source or ("OpenAI" if detected else None),
            results=normalized,
        )
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result
