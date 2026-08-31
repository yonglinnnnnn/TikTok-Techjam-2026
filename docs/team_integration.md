# Team integration handoff

## Shared lifecycle

```text
Original uploaded bytes
        |
        +--> Tier 1: C2PA + metadata + blind forensics
        |        |
        |        +--> trusted C2PA AI claim: fusion may use the fast path
        |        +--> valid/untrusted C2PA: retain as detected and continue
        |        +--> otherwise: continue
        |
        +--> Normalized derivative (never overwrite original)
                 |
                 +--> Tier 2 VLM
                 +--> Tier 3 ConvNeXt + Grad-CAM
                              |
                              +--> final calibrated fusion
```

The shared JSON Schema is `schemas/prediction.schema.json`. Tier 1 initializes
the envelope. Every member preserves existing keys and writes only their owned
block.

## Tier 2 / VLM owner

Consume:

```python
vlm_input = {
    "image_path": normalized_image_path,
    "forensic_overlay": result["forensics"]["artifacts"]["vlm_overlay"],
    "candidate_regions": result["forensics"]["candidate_regions"],
    "provenance_status": result["provenance"]["status"],
    "provenance_detected": result["tier1"]["provenance_detected"],
    "provenance_verified": result["tier1"]["provenance_verified"],
}
```

Write `result["tier2"]` with the VLM verdict, confidence, source estimate,
AI-coverage estimate, evidence, and `latency_ms`. Do not interpret missing provenance
as authentic. Do not send raw manifests, individual forensic maps, or the human
review panel to the VLM.

## Tier 3 / ConvNeXt owner

Consume only the normalized RGB image. Keep this classifier independent of
provenance and handcrafted forensic inputs unless it is explicitly retrained as a
multi-input model.

Map the current Member 1 result as follows:

```python
result["tier3"] = {
    "probability": member1_prediction["pred"],
    "heatmap_path": member1_prediction["heatmap_path"],
    "evidence": ["ConvNeXt visual classifier completed"],
    "latency_ms": tier3_latency_ms
}
```

The orchestrator measures `tier3_latency_ms`; Member 1's existing prediction
object does not have to add that field.

Grad-CAM is an explanation, not pixel segmentation; do not turn it directly into
`ai_coverage`.

## Fusion/orchestration owner

Consume compact independent features:

```python
fusion_input = {
    "provenance_severity": result["tier1"]["severity_weight"],
    "provenance_verified_ai": result["tier1"]["verified_ai_signal"],
    "forensic_integrity": result["tier1"]["forensic_integrity_weight"],
    "vlm_confidence": result["tier2"]["confidence"],
    "convnext_probability": result["tier3"]["probability"],
}
```

The fusion owner sets the final top-level `is_ai_generated`, `confidence`,
`source`, `ai_coverage`, and `heatmap_path`, appends evidence, and writes
`result["fusion"]`. Do not add the three numeric scores directly; fit and validate
the final fusion/calibration policy on held-out data.

The fusion block records `method`, whether it is calibrated, its decision
threshold, the compact inputs used, and its own latency.

## UI/demo owner

Display the final verdict, confidence, source, concise provenance conclusion,
C2PA chain/actions, evidence, `artifacts.human_review`, and Tier 3 heatmap. Debug
payloads should be hidden unless developer mode is enabled.

## Null and routing rules

- `null`: unknown, unavailable, or not owned by the current tier.
- `false`: the relevant checker ran and returned a negative result.
- `provenance_detected`: at least one supported provenance signal is present.
- `provenance_verified`: at least one detected signal passes verification policy.
- `requires_tier2: false`: only a trusted, AI-attributed C2PA credential.
- verified capture: supporting negative evidence, but still continue.
- `provenance.status: detected`: provenance exists, but it is not a verified
  AI/capture result; continue.
- no provenance: no conclusion; still continue.

`signals[].evidence_score` and `tier1.severity_weight` measure AI-generation
evidence. Do not interpret `0.0` as "no C2PA" or "credential invalid"; inspect
`present`, `validation_state`, and `verified` for credential state.
