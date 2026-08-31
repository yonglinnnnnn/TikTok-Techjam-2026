# Unified result contract

The canonical contract is `schemas/prediction.schema.json`. Tier 1 creates the
complete envelope with `tier2`, `tier3`, and `fusion` set to `null`. Later owners
replace only the block they own and update final top-level fields during fusion.

Unknown is always `null`, never a false negative. In particular:

- missing C2PA does not establish authenticity;
- no watermark provider is registered, so `watermark_detected` is `null`;
- `candidate_coverage` is forensic anomaly coverage, not `ai_coverage`.

## Normalized provenance signal

Every checker contributes the same object:

```json
{
  "kind": "content_credential",
  "provider": "c2pa_local",
  "status": "detected",
  "present": true,
  "verified": false,
  "ai_claim": false,
  "source": null,
  "verification_basis": null,
  "evidence_score": 0.0,
  "details": {"validation_state": "valid"}
}
```

The stable statuses are `verified`, `detected`, `not_detected`, `not_present`,
`invalid`, `unavailable`, `error`, and `not_checked`.

For C2PA, `validation_state` is copied from the SDK. `valid` means the credential
integrity checks pass but the signer is not trusted; `trusted` additionally means
the certificate chains to a configured trust anchor. The normalized signal sets
`verified: true` only for `trusted`. A present `valid` credential uses signal and
overall status `detected`, so it is not incorrectly reported as
`no_provenance_found`.

Overall `provenance.status` may be `verified_ai`, `verified_capture`,
`ai_indicated`, `detected`, `invalid`, `no_provenance_found`, or `error`.

The Tier 1 summary deliberately exposes both presence and verification:

```json
{
  "provenance_detected": true,
  "provenance_verified": false,
  "verified_ai_signal": false,
  "verified_capture_signal": false
}
```

`provenance_detected` is true when any normalized signal has `present: true`.
`provenance_verified` is true only when at least one signal has `verified: true`.

## Routing policy

Only a trusted, AI-attributed C2PA credential sets `requires_tier2` to `false`.
A trusted digital-capture claim is retained as strong supporting evidence but
still routes to Tier 2 and Tier 3. Detected, valid-but-untrusted, invalid, and
missing credentials also continue downstream.

## Scores

`signals[].evidence_score` and `tier1.severity_weight` are AI-generation evidence,
not scores for credential integrity. A credential can be present and valid while
having `evidence_score: 0.0` when it makes no recognized AI claim. Correlated
signals for the same underlying credential are not added together. The default
`policy_v1` uses the strongest signal and reports
`calibrated: false`. A fitted artifact can replace it; its path is supplied with
`--calibration` or `TRUESIGHT_PROVENANCE_CALIBRATION`.

`tier1.forensic_integrity_weight` is separate weak integrity evidence. It never
sets the AI verdict by itself.

Raw C2PA manifests, validation diagnostics, metadata fields, thresholds, and
individual forensic maps are debug data. Enable them with `--debug-provenance`
and `--debug-forensics`; do not send them to ordinary downstream models.
