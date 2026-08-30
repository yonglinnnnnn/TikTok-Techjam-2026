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

def _ingredient_manifest(ingredient: dict[str, Any]) -> str | None:
    value = ingredient.get("active_manifest") or ingredient.get("activeManifest")
    return str(value) if value else None

def _ingredient_refs(action: dict[str, Any]) -> list[str]:
    parameters = action.get("parameters") or {}
    references = parameters.get("ingredients") or parameters.get("ingredient") or []
    if isinstance(references, dict):
        references = [references]
    found: list[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        value = reference.get("url") or reference.get("identifier")
        if value:
            found.append(str(value))
    return found

def _normalized_action(label: str, action: dict[str, Any], depth: int) -> dict[str, Any]:
    return {
        "manifest": label,
        "depth": depth,
        "action": action.get("action"),
        "when": action.get("when"),
        "software_agent": action.get("softwareAgent") or action.get("software_agent"),
        "digital_source_type": _source_type(action),
        "description": action.get("description"),
        "changes": action.get("changes"),
        "ingredient_refs": _ingredient_refs(action),
    }

def _manifest_node(label: str, manifest: dict[str, Any], active_label: str,
                   depth: int) -> dict[str, Any]:
    signature = manifest.get("signature_info") or {}
    ingredients: list[dict[str, Any]] = []
    for ingredient in manifest.get("ingredients") or []:
        if not isinstance(ingredient, dict):
            continue
        ingredients.append({
            "label": ingredient.get("label"),
            "title": ingredient.get("title"),
            "format": ingredient.get("format"),
            "relationship": ingredient.get("relationship"),
            "instance_id": ingredient.get("instance_id") or ingredient.get("instanceId"),
            "manifest": _ingredient_manifest(ingredient),
            "has_manifest": bool(_ingredient_manifest(ingredient)),
        })
    return {
        "manifest": label,
        "active": label == active_label,
        "depth": depth,
        "title": manifest.get("title"),
        "format": manifest.get("format"),
        "instance_id": manifest.get("instance_id") or manifest.get("instanceId"),
        "claim_generator": manifest.get("claim_generator"),
        "claim_generator_info": manifest.get("claim_generator_info") or [],
        "signature": {
            "issuer": signature.get("issuer"),
            "time": signature.get("time"),
            "algorithm": signature.get("alg"),
        },
        "actions": [_normalized_action(label, action, depth) for action in _actions(manifest)],
        "ingredients": ingredients,
    }

def _history(store: dict[str, Any]) -> dict[str, Any]:
    """Build an ingredient-linked graph and an oldest-to-active action timeline."""
    manifests = store.get("manifests") or {}
    active_label = str(store.get("active_manifest") or "")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_manifests: list[str] = []
    cycles: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(label: str, depth: int) -> None:
        if label in visiting:
            cycles.append(label)
            return
        if label in visited:
            return
        manifest = manifests.get(label)
        if not isinstance(manifest, dict):
            missing_manifests.append(label)
            return
        visiting.add(label)
        node = _manifest_node(label, manifest, active_label, depth)
        nodes.append(node)
        for ingredient in node["ingredients"]:
            child = ingredient["manifest"]
            edges.append({
                "from_manifest": label,
                "to_manifest": child,
                "relationship": ingredient["relationship"],
                "ingredient_label": ingredient["label"],
                "title": ingredient["title"],
            })
            if child:
                visit(child, depth + 1)
        visiting.remove(label)
        visited.add(label)

    if active_label:
        visit(active_label, 0)
    # Retain disconnected manifests for audit, but do not insert their actions
    # into the active asset's declared history.
    disconnected = sorted(str(label) for label in manifests if str(label) not in visited)
    for label in disconnected:
        manifest = manifests.get(label)
        if isinstance(manifest, dict):
            nodes.append(_manifest_node(label, manifest, active_label, -1))

    timeline: list[dict[str, Any]] = []
    timeline_seen: set[str] = set()

    def append_history(label: str, depth: int) -> None:
        if label in timeline_seen:
            return
        timeline_seen.add(label)
        manifest = manifests.get(label)
        if not isinstance(manifest, dict):
            return
        for ingredient in manifest.get("ingredients") or []:
            if isinstance(ingredient, dict):
                child = _ingredient_manifest(ingredient)
                if child:
                    append_history(child, depth + 1)
        timeline.extend(_normalized_action(label, action, depth)
                        for action in _actions(manifest))

    if active_label:
        append_history(active_label, 0)

    origin = next((x for x in timeline if x["action"] in {"c2pa.created", "c2pa.opened"}), None)
    content_edits = [x for x in timeline if x["action"] in CONTENT_EDIT_ACTIONS]
    transforms = [x for x in timeline if x["action"] in TRANSFORM_ACTIONS]
    metadata_edits = [x for x in timeline if x["action"] == "c2pa.edited.metadata"]
    classified = CONTENT_EDIT_ACTIONS | TRANSFORM_ACTIONS | NON_EDIT_ACTIONS
    unclassified = [x for x in timeline if x["action"] not in classified]
    origin_type = origin["digital_source_type"] if origin else None
    unsigned_count = sum(1 for edge in edges if edge["to_manifest"] is None)
    return {"origin": origin, "origin_type": origin_type, "timeline": timeline,
        "content_edited": bool(content_edits), "content_edits": content_edits,
        "transformed": bool(transforms), "transforms": transforms,
        "metadata_edited": bool(metadata_edits), "metadata_edits": metadata_edits,
        "has_unclassified_actions": bool(unclassified),
        "unclassified_actions": unclassified,
        "ingredient_count": len(edges), "manifest_count": len(manifests),
        "chain": {
            "root_manifest": active_label or None,
            "nodes": nodes,
            "edges": edges,
            "complete": not missing_manifests and not cycles and not disconnected,
            "missing_manifests": sorted(set(missing_manifests)),
            "cycles": sorted(set(cycles)),
            "disconnected_manifests": disconnected,
            "unsigned_ingredient_count": unsigned_count,
        }}

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
