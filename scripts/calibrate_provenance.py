"""Fit provenance severity calibration from labelled normalized signals."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FEATURE_NAMES = [
    "verified_ai_credential",
    "provider_watermark",
    "unverified_ai_credential",
    "metadata_ai_marker",
]


def fit(input_csv: Path, output_json: Path) -> dict:
    frame = pd.read_csv(input_csv)
    required = {*FEATURE_NAMES, "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(missing)}")
    if len(frame) < 30 or frame["label"].nunique() != 2:
        raise ValueError("calibration requires at least 30 rows and both label classes")

    x = frame[FEATURE_NAMES].astype(float)
    y = frame["label"].astype(int)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]
    metrics = {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "brier_score_training": round(float(brier_score_loss(y, probabilities)), 6),
        "log_loss_training": round(float(log_loss(y, probabilities)), 6),
        "roc_auc_training": round(float(roc_auc_score(y, probabilities)), 6),
    }
    artifact = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "method": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "coefficients": dict(zip(FEATURE_NAMES, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "metrics": metrics,
        "warning": "Training metrics are descriptive; validate on a held-out set before deployment.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Tier 1 provenance calibration")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(args.input_csv, args.output_json), indent=2))


if __name__ == "__main__":
    main()
