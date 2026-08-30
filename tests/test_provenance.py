from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.provenance.c2pa_checker import _history
from truesight.provenance.openai_checker import check_openai_provenance
from truesight.provenance.scoring import score_signals
from truesight.provenance.signals import (
    aggregate_signals,
    has_verified_ai_signal,
    has_verified_capture_signal,
    watermark_detection,
)


def _action(name: str, source_type: str | None = None) -> dict:
    value = {"action": name}
    if source_type:
        value["digitalSourceType"] = source_type
    return value


def _manifest(actions: list[dict], ingredients: list[dict] | None = None) -> dict:
    return {
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
        "ingredients": ingredients or [],
        "signature_info": {"issuer": "Test signer", "alg": "ES256"},
    }


def test_c2pa_history_builds_ingredient_chain_oldest_first():
    capture = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    store = {
        "active_manifest": "m2",
        "manifests": {
            "m1": _manifest([_action("c2pa.created", capture)]),
            "m2": _manifest(
                [_action("c2pa.opened"), _action("c2pa.adjustedColor")],
                [{"label": "parent", "title": "original.jpg",
                  "relationship": "parentOf", "active_manifest": "m1"}],
            ),
        },
    }

    history = _history(store)

    assert history["chain"]["root_manifest"] == "m2"
    assert history["chain"]["complete"] is True
    assert history["chain"]["edges"][0]["to_manifest"] == "m1"
    assert [item["manifest"] for item in history["timeline"]] == ["m1", "m2", "m2"]
    assert history["origin_type"] == capture
    assert history["content_edited"] is True


def test_openai_checker_normalizes_c2pa_and_synthid(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"original-file-bytes")

    class Checks:
        def create(self, *, file):
            assert file == b"original-file-bytes"
            return SimpleNamespace(results=[
                {"type": "c2pa", "outcome": "detected", "model": "gpt-image-1",
                 "issuer": "OpenAI", "generated_at": "2026-01-01T00:00:00Z",
                 "validation_state": "trusted"},
                {"type": "synthid", "outcome": "detected", "model": "gpt-image-1",
                 "generated_at": None},
            ])

    client = SimpleNamespace(content_provenance_checks=Checks())
    result = check_openai_provenance(image_path, client=client)

    assert result["checked"] is True
    assert result["status"] == "detected"
    assert result["verified"] is True
    assert {item["type"] for item in result["results"]} == {"c2pa", "synthid"}


def test_general_signals_and_routing_for_provider_watermark():
    c2pa = {"status": "not_present", "present": False, "verified": False,
            "validation_state": None, "history": None}
    metadata = {"checked": True, "metadata": {}, "ai_markers": [], "error": None}
    openai_result = {
        "checked": True, "status": "detected", "present": True,
        "verified": True, "source": "gpt-image-1", "error": None,
        "results": [{"type": "synthid", "outcome": "detected",
                     "model": "gpt-image-1", "issuer": None,
                     "generated_at": None, "validation_state": None}],
    }

    signals, severity = score_signals(aggregate_signals(c2pa, metadata, openai_result))

    assert has_verified_ai_signal(signals) is True
    assert watermark_detection(signals) is True
    assert severity["score"] == 0.97
    assert severity["calibrated"] is False


def test_verified_capture_does_not_skip_downstream():
    capture = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    c2pa = {
        "status": "trusted", "present": True, "verified": True,
        "validation_state": "trusted",
        "history": {"origin_type": capture, "content_edited": False,
                    "transformed": False, "has_unclassified_actions": False},
    }
    metadata = {"checked": True, "metadata": {}, "ai_markers": [], "error": None}
    openai_result = {"checked": False, "status": "not_checked", "present": None,
                     "verified": False, "source": None, "results": [], "error": None}
    signals, _ = score_signals(aggregate_signals(c2pa, metadata, openai_result))

    assert has_verified_capture_signal(signals, c2pa) is True
    assert has_verified_ai_signal(signals) is False


def test_later_ai_composite_is_not_misclassified_as_verified_capture():
    capture = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    composite = ("http://cv.iptc.org/newscodes/digitalsourcetype/"
                 "compositedWithTrainedAlgorithmicMedia")
    c2pa = {
        "status": "trusted", "present": True, "verified": True,
        "validation_state": "trusted",
        "history": {
            "origin_type": capture,
            "timeline": [
                {"digital_source_type": capture},
                {"digital_source_type": composite},
            ],
            "content_edited": False, "transformed": False,
            "has_unclassified_actions": False,
        },
    }
    metadata = {"checked": True, "metadata": {}, "ai_markers": [], "error": None}
    openai_result = {"checked": False, "status": "not_checked", "present": None,
                     "verified": False, "source": None, "results": [], "error": None}
    signals, _ = score_signals(aggregate_signals(c2pa, metadata, openai_result))

    assert has_verified_capture_signal(signals, c2pa) is False
    assert has_verified_ai_signal(signals) is True


def test_optional_calibration_artifact(tmp_path):
    artifact = {
        "version": "test-v1",
        "method": "logistic_regression",
        "feature_names": ["verified_ai_credential", "provider_watermark",
                          "unverified_ai_credential", "metadata_ai_marker"],
        "coefficients": {"verified_ai_credential": 2.0,
                         "provider_watermark": 3.0,
                         "unverified_ai_credential": 0.5,
                         "metadata_ai_marker": 0.1},
        "intercept": -1.0,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    signal = {
        "kind": "watermark", "provider": "openai", "status": "verified",
        "present": True, "verified": True, "ai_claim": True,
        "source": "OpenAI", "verification_basis": "provider_api",
        "evidence_score": 0.0, "details": {},
    }

    _, severity = score_signals([signal], path)

    assert severity["calibrated"] is True
    assert severity["model_version"] == "test-v1"
    assert 0.87 < severity["score"] < 0.89


def test_unified_schema_accepts_tier1_output(tmp_path, monkeypatch):
    import jsonschema
    from referencing import Registry, Resource
    from truesight.provenance import tier1

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)
    no_c2pa = {
        "checked": True, "status": "not_present", "present": False,
        "verified": False, "active_manifest": None, "validation_state": None,
        "validation_results": None, "history": None, "manifest": None, "error": None,
    }
    no_metadata = {"checked": True, "ai_markers": [], "metadata": {}, "error": None}
    no_openai = {"checked": False, "status": "not_checked", "present": None,
                 "verified": False, "source": None, "results": [], "error": None}
    monkeypatch.setattr(tier1, "check_c2pa", lambda _path: no_c2pa)
    monkeypatch.setattr(tier1, "check_metadata", lambda _path: no_metadata)
    monkeypatch.setattr(tier1, "check_openai_provenance", lambda _path: no_openai)

    result = tier1.analyze_tier1(image_path, run_forensics=False)
    schema_path = ROOT / "schemas" / "prediction.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    provenance_schema = json.loads(
        (ROOT / "schemas" / "provenance.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        "provenance.schema.json", Resource.from_contents(provenance_schema))

    jsonschema.Draft202012Validator(schema, registry=registry).validate(result)
    assert result["tier1"]["requires_tier2"] is True
    assert result["is_ai_generated"] is None

    result["tier2"] = {
        "is_ai_generated": True, "confidence": 0.72, "source": "unknown",
        "ai_coverage": 0.35, "evidence": ["VLM analysis completed"],
        "latency_ms": 400,
    }
    result["tier3"] = {
        "probability": 0.81, "heatmap_path": "outputs/heatmaps/sample.png",
        "evidence": ["ConvNeXt visual classifier completed"], "latency_ms": 120,
    }
    result["fusion"] = {
        "method": "logistic_regression", "calibrated": True,
        "decision_threshold": 0.5,
        "inputs": {"provenance_severity": 0.0, "vlm_confidence": 0.72,
                   "convnext_probability": 0.81},
        "latency_ms": 2,
    }
    jsonschema.Draft202012Validator(schema, registry=registry).validate(result)
