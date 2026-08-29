# Unified detection result

Every tier enriches the same object. Unknown values remain `null`, not `false`.
When Tier 1 has no decisive result, top-level `confidence` is also `null`; its
evidence contribution remains available as `tier1.severity_weight`.
Tier 1 also returns an internal orchestration block containing
`watermark_detected`, `provenance_verified`, both evidence weights, and
`requires_tier2`. Until a real watermark checker is connected,
`watermark_detected` is `null`, not a misleading `false`.

Policy: metadata is weak evidence. A present-but-invalid C2PA manifest is weak
evidence. Only a cryptographically verified and AI-attributed C2PA claim may
quick-exit Tier 1. Missing provenance never proves that an image is authentic.

`provenance.signals` is the stable fusion interface. Each checker reports its
kind, provider, status, presence, verification, AI claim, source and evidence
score. Raw checker payloads are retained for diagnostics. C2PA reports origin,
the declared action timeline, content edits, transforms, metadata edits and
ingredients; it does not infer origin from vendor-name aliases. `provenance.file`
contains SHA-256 and container facts for deduplication/audit, but carries no AI
evidence by itself.

`forensics` is a blind (no-original-required) manipulation analysis. Classical
recompression, noise, JPEG-grid and copy-move maps produce candidate regions, a
capped `integrity_risk_weight`, and a compact numeric `feature_vector`. No learned
model is used and this layer does not set an AI verdict or top-level confidence.
Tier 2 normally receives only the original image and `artifacts.vlm_overlay` plus
the compact JSON signals. Tier 3 receives the original image; handcrafted maps
are not ConvNeXt inputs unless that model is explicitly retrained for them.
Tampering alone does not establish that generative AI was used.

Production output omits raw C2PA manifests, validation diagnostics, raw metadata,
individual forensic maps and thresholds. Use `--debug-provenance` and
`--debug-forensics` to expose those fields during development.
