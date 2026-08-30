"""Normalize provenance providers into one stable signal interface."""
from __future__ import annotations

from typing import Any

AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositedwithtrainedalgorithmicmedia")
CAPTURE_SOURCE_TYPES = ("digitalcapture",)


def _source_values(c2pa: dict[str, Any]) -> list[str]:
    history = c2pa.get("history") or {}
    source_types = [history.get("origin_type")]
    source_types.extend(item.get("digital_source_type")
                        for item in history.get("timeline") or []
                        if isinstance(item, dict))
    return [str(value) for value in source_types if value]


def _matching_source(c2pa: dict[str, Any], values: tuple[str, ...]) -> str | None:
    for source in reversed(_source_values(c2pa)):
        if any(value in source.lower() for value in values):
            return source
    return None


def source_has(c2pa: dict[str, Any], values: tuple[str, ...]) -> bool:
    return _matching_source(c2pa, values) is not None


def _signal(*, kind: str, provider: str, status: str,
            present: bool | None, verified: bool, ai_claim: bool,
            source: str | None, verification_basis: str | None = None,
            details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "provider": provider,
        "status": status,
        "present": present,
        "verified": verified,
        "ai_claim": ai_claim,
        "source": source,
        "verification_basis": verification_basis,
        "evidence_score": 0.0,
        "details": details or {},
    }


def _local_c2pa_signal(c2pa: dict[str, Any]) -> dict[str, Any]:
    ai_claim = source_has(c2pa, AI_SOURCE_TYPES)
    if c2pa.get("status") in {"error", "unavailable", "not_checked"}:
        status = str(c2pa.get("status"))
        present: bool | None = None
    elif not c2pa.get("present"):
        status, present = "not_present", False
    elif c2pa.get("verified"):
        status, present = "verified", True
    elif c2pa.get("validation_state") == "invalid":
        status, present = "invalid", True
    else:
        status, present = "detected", True
    return _signal(
        kind="content_credential",
        provider="c2pa_local",
        status=status,
        present=present,
        verified=bool(c2pa.get("verified")),
        ai_claim=ai_claim,
        source=(_matching_source(c2pa, AI_SOURCE_TYPES) if ai_claim else
                (c2pa.get("history") or {}).get("origin_type")),
        verification_basis="c2pa_trust" if c2pa.get("verified") else None,
        details={"validation_state": c2pa.get("validation_state")},
    )


def _openai_signals(openai_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not openai_result.get("checked"):
        return [_signal(
            kind="provider_check",
            provider="openai",
            status=str(openai_result.get("status") or "not_checked"),
            present=None,
            verified=False,
            ai_claim=False,
            source=None,
            details={"error": openai_result.get("error")},
        )]

    signals: list[dict[str, Any]] = []
    for result in openai_result.get("results") or []:
        signal_type = str(result.get("type") or "unknown").lower()
        detected = result.get("outcome") == "detected"
        validation_state = result.get("validation_state")
        is_watermark = signal_type == "synthid"
        verified = bool(detected and (is_watermark or validation_state == "trusted"))
        status = "verified" if verified else "detected" if detected else "not_detected"
        signals.append(_signal(
            kind="watermark" if is_watermark else "content_credential",
            provider="openai",
            status=status,
            present=detected,
            verified=verified,
            ai_claim=detected,
            source=result.get("model") or result.get("issuer") or
                   ("OpenAI" if detected else None),
            verification_basis="provider_api" if is_watermark and detected else
                               "c2pa_trust" if verified else None,
            details={
                "signal_type": signal_type,
                "validation_state": validation_state,
                "issuer": result.get("issuer"),
                "model": result.get("model"),
                "generated_at": result.get("generated_at"),
            },
        ))
    if not signals:
        signals.append(_signal(
            kind="provider_check", provider="openai", status="not_detected",
            present=False, verified=False, ai_claim=False, source=None,
        ))
    return signals


def _metadata_signal(metadata: dict[str, Any]) -> dict[str, Any]:
    markers = metadata.get("ai_markers") or []
    if not metadata.get("checked"):
        status, present = "error", None
    else:
        status, present = ("detected", True) if markers else ("not_detected", False)
    return _signal(
        kind="embedded_metadata",
        provider="exif_xmp",
        status=status,
        present=present,
        verified=False,
        ai_claim=bool(markers),
        source=markers[0] if markers else None,
        details={"metadata_present": bool(metadata.get("metadata")),
                 "markers": markers},
    )


def aggregate_signals(c2pa: dict[str, Any], metadata: dict[str, Any],
                      openai_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [_local_c2pa_signal(c2pa), *_openai_signals(openai_result),
            _metadata_signal(metadata)]


def has_verified_ai_signal(signals: list[dict[str, Any]]) -> bool:
    return any(signal["verified"] and signal["ai_claim"] and
               signal["kind"] in {"content_credential", "watermark"}
               for signal in signals)


def has_verified_capture_signal(signals: list[dict[str, Any]],
                                c2pa: dict[str, Any]) -> bool:
    history = c2pa.get("history") or {}
    return bool(
        source_has(c2pa, CAPTURE_SOURCE_TYPES)
        and not source_has(c2pa, AI_SOURCE_TYPES)
        and any(signal["provider"] == "c2pa_local" and signal["verified"]
                for signal in signals)
        and not history.get("content_edited")
        and not history.get("transformed")
        and not history.get("has_unclassified_actions")
    )


def watermark_detection(signals: list[dict[str, Any]]) -> bool | None:
    checked = [signal for signal in signals
               if signal["kind"] == "watermark" and
               signal["status"] not in {"unavailable", "error", "not_checked"}]
    if not checked:
        return None
    return any(signal["present"] is True for signal in checked)


def best_ai_source(signals: list[dict[str, Any]]) -> str | None:
    candidates = [signal for signal in signals if signal["ai_claim"] and signal["source"]]
    candidates.sort(key=lambda item: (item["verified"], item["evidence_score"]), reverse=True)
    return candidates[0]["source"] if candidates else None
