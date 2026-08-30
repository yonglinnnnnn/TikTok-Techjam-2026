"""Conservative evidence scoring with optional fitted logistic calibration."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

FEATURE_NAMES = (
    "verified_ai_credential",
    "provider_watermark",
    "unverified_ai_credential",
    "metadata_ai_marker",
)

POLICY_WEIGHTS = {
    "verified_ai_credential": 0.98,
    "provider_watermark": 0.97,
    "unverified_ai_credential": 0.35,
    "metadata_ai_marker": 0.15,
}


def signal_features(signals: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "verified_ai_credential": float(any(
            signal["kind"] == "content_credential" and signal["verified"] and
            signal["ai_claim"] for signal in signals)),
        "provider_watermark": float(any(
            signal["kind"] == "watermark" and signal["verified"] and
            signal["ai_claim"] for signal in signals)),
        "unverified_ai_credential": float(any(
            signal["kind"] == "content_credential" and signal["present"] and
            signal["ai_claim"] and not signal["verified"] for signal in signals)),
        "metadata_ai_marker": float(any(
            signal["kind"] == "embedded_metadata" and signal["ai_claim"]
            for signal in signals)),
    }


def _signal_policy_score(signal: dict[str, Any]) -> float:
    if not signal["ai_claim"] or signal["present"] is not True:
        return 0.0
    if signal["kind"] == "watermark" and signal["verified"]:
        return POLICY_WEIGHTS["provider_watermark"]
    if signal["kind"] == "content_credential" and signal["verified"]:
        return POLICY_WEIGHTS["verified_ai_credential"]
    if signal["kind"] == "content_credential":
        return POLICY_WEIGHTS["unverified_ai_credential"]
    if signal["kind"] == "embedded_metadata":
        return POLICY_WEIGHTS["metadata_ai_marker"]
    return 0.0


def _calibration_path(path: str | Path | None) -> Path | None:
    value = str(path) if path is not None else os.getenv("TRUESIGHT_PROVENANCE_CALIBRATION")
    return Path(value) if value else None


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def score_signals(signals: list[dict[str, Any]],
                  calibration_path: str | Path | None = None
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = [{**signal, "evidence_score": round(_signal_policy_score(signal), 4)}
              for signal in signals]
    features = signal_features(scored)
    raw_score = max((signal["evidence_score"] for signal in scored), default=0.0)
    result: dict[str, Any] = {
        "score": raw_score,
        "raw_score": raw_score,
        "calibrated": False,
        "method": "policy_v1",
        "model_version": None,
        "features": features,
        "error": None,
    }

    path = _calibration_path(calibration_path)
    if path is None:
        return scored, result
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        feature_names = artifact["feature_names"]
        coefficients = artifact["coefficients"]
        intercept = float(artifact["intercept"])
        if set(feature_names) != set(FEATURE_NAMES):
            raise ValueError("calibration feature names do not match this pipeline")
        # No positive signal means no provenance evidence. Do not turn the
        # training-set prior/intercept into a positive provenance contribution.
        if any(features.values()):
            logit = intercept + sum(float(coefficients[name]) * features[name]
                                    for name in feature_names)
            result["score"] = round(_sigmoid(logit), 6)
        result.update(
            calibrated=True,
            method=str(artifact.get("method") or "logistic_regression"),
            model_version=str(artifact.get("version") or "unknown"),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return scored, result
