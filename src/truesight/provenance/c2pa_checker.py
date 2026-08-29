"""C2PA/JUMBF provenance extraction and cryptographic validation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

CONTENT_EDIT_ACTIONS = {
    "c2pa.addedText", "c2pa.adjustedColor", "c2pa.cropped", "c2pa.deleted",
    "c2pa.drawing", "c2pa.dubbed", "c2pa.edited", "c2pa.filtered",
    "c2pa.placed", "c2pa.removed", "c2pa.replaced",
}
TRANSFORM_ACTIONS = {
    "c2pa.converted", "c2pa.enhanced", "c2pa.formatted", "c2pa.repackaged",
    "c2pa.resized", "c2pa.transcoded",
}
NON_EDIT_ACTIONS = {
    "c2pa.created", "c2pa.opened", "c2pa.published", "c2pa.saved",
    "c2pa.managed", "c2pa.produced", "c2pa.edited.metadata",
}

def _actions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for assertion in manifest.get("assertions") or []:
        if not isinstance(assertion, dict):
            continue
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            data = assertion.get("data") or {}
            if isinstance(data, dict):
                found.extend(x for x in data.get("actions", []) if isinstance(x, dict))
    return found

def _source_type(action: dict[str, Any] | None) -> str | None:
    if not action:
        return None
    value = action.get("digitalSourceType") or action.get("digital_source_type")
    return str(value) if value else None

def _history(store: dict[str, Any]) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    ingredient_count = 0
    for label, manifest in (store.get("manifests") or {}).items():
        if not isinstance(manifest, dict):
            continue
        ingredient_count += len(manifest.get("ingredients") or [])
        for action in _actions(manifest):
            timeline.append({"manifest": label, "action": action.get("action"),
                "when": action.get("when"), "software_agent": action.get("softwareAgent"),
                "digital_source_type": _source_type(action),
                "description": action.get("description"), "changes": action.get("changes")})
    origin = next((x for x in timeline if x["action"] in {"c2pa.created", "c2pa.opened"}), None)
    content_edits = [x for x in timeline if x["action"] in CONTENT_EDIT_ACTIONS]
    transforms = [x for x in timeline if x["action"] in TRANSFORM_ACTIONS]
    metadata_edits = [x for x in timeline if x["action"] == "c2pa.edited.metadata"]
    classified = CONTENT_EDIT_ACTIONS | TRANSFORM_ACTIONS | NON_EDIT_ACTIONS
    unclassified = [x for x in timeline if x["action"] not in classified]
    origin_type = origin["digital_source_type"] if origin else None
    return {"origin": origin, "origin_type": origin_type, "timeline": timeline,
        "content_edited": bool(content_edits), "content_edits": content_edits,
        "transformed": bool(transforms), "transforms": transforms,
        "metadata_edited": bool(metadata_edits), "metadata_edits": metadata_edits,
        "has_unclassified_actions": bool(unclassified),
        "unclassified_actions": unclassified,
        "ingredient_count": ingredient_count, "manifest_count": len(store.get("manifests") or {})}

def check_c2pa(image_path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {"checked": False, "status": "not_checked",
        "present": False, "verified": False,
        "active_manifest": None, "validation_state": None,
        "validation_results": None, "history": None, "manifest": None, "error": None}
    try:
        from c2pa import Reader
    except ImportError:
        result["status"] = "unavailable"
        result["error"] = "c2pa-python is not installed"
        return result
    try:
        with Reader(str(image_path)) as reader:
            store = json.loads(reader.json())
            label = store.get("active_manifest")
            manifest = (store.get("manifests") or {}).get(label, {})
            state = reader.get_validation_state()
            state_text = str(getattr(state, "value", state)).lower()
            # "Valid" has no validation errors but an untrusted signer. Only the
            # SDK's "Trusted" state satisfies this pipeline's verified policy.
            status = "not_present" if not label else state_text
            result.update(checked=True, status=status, present=bool(label),
                verified=state_text == "trusted",
                active_manifest=label,
                validation_state=state_text,
                validation_results=reader.get_validation_results(),
                history=_history(store), manifest=manifest)
    except Exception as exc:
        message = str(exc)
        if "ManifestNotFound" in message or "no JUMBF data found" in message:
            result.update(checked=True, status="not_present")
        else:
            result.update(checked=False, status="error",
                          error=f"{type(exc).__name__}: {exc}")
    return result
