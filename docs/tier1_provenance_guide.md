# Tier 1 provenance implementation guide

## Installation

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The local C2PA, metadata, and forensic checks do not require an API key.

Tier 1 loads the official C2PA Conformance Trust List bundled at
`configs/c2pa/C2PA-TRUST-LIST.pem`. Override it in deployments with:

```powershell
$env:TRUESIGHT_C2PA_TRUST_ANCHORS = "C:\path\to\C2PA-TRUST-LIST.pem"
```

The bundled list comes from the C2PA Conformance Program repository:
https://github.com/c2pa-org/conformance-public/tree/main/trust-list
Refresh it deliberately as part of dependency/security maintenance; do not
silently fetch mutable trust anchors during image analysis.

## Running

```powershell
$env:PYTHONPATH = "src"
python -m truesight.provenance.tier1 data/image.jpg
```

Useful flags:

```text
--skip-forensics           Run provenance only
--debug-provenance         Include raw manifest/validation/metadata diagnostics
--debug-forensics          Include individual forensic maps and thresholds
--calibration PATH         Load a fitted severity calibration artifact
```

Always run Tier 1 on the untouched uploaded bytes. Any RGB conversion, EXIF
orientation, resize, or recompression must produce a separate file for Tier 2/3.

## C2PA result

The C2PA SDK validation state and the pipeline verification flag answer different
questions:

- `valid`: the signature, assertion hashes, and asset hash validate, but the
  signing identity is not anchored in the verifier's trust store;
- `trusted`: the credential is valid and its signer chains to an approved trust
  anchor;
- `invalid`: at least one validation check failed.

Tier 1 preserves the SDK value in `provenance.c2pa.validation_state`. It sets
`verified: true` only for `trusted`; it never rewrites `valid` as `trusted`.
A valid-but-untrusted credential is still reported as present and produces the
overall provenance status `detected`, not `no_provenance_found`.

`provenance.c2pa.history.chain` contains:

- the active/root manifest;
- one node per reachable manifest, plus disconnected nodes for audit;
- ingredient edges with relationship and title;
- missing references and cycles;
- unsigned ingredient count;
- an oldest-to-active declared-action timeline.

`content_edited` and `transformed` describe signed, declared actions. Their being
false does not prove that no undeclared edit ever occurred.

## Status and score interpretation

The overall `provenance.status` values used by Tier 1 are:

- `verified_ai`: a trusted C2PA credential declares an AI source type;
- `verified_capture`: a trusted credential declares digital capture and passes
  the capture-history policy;
- `ai_indicated`: unverified provenance or metadata indicates AI generation;
- `detected`: a provenance signal exists but has no verified AI/capture result;
- `invalid`: a credential is present but fails validation;
- `no_provenance_found`: no supported provenance signal is present.

`signals[].evidence_score` and `tier1.severity_weight` measure evidence of AI
generation. They do not measure credential integrity. A valid camera credential
with `ai_claim: false` therefore correctly has `evidence_score: 0.0`. Blind
forensic anomaly evidence remains separate in `tier1.forensic_integrity_weight`.

For routing and filtering, Tier 1 exposes two separate booleans:

- `provenance_detected`: at least one supported signal has `present: true`;
- `provenance_verified`: at least one signal passes its verification policy.

For example, valid-but-untrusted C2PA produces `provenance_detected: true` and
`provenance_verified: false`.

## Severity calibration

Create a CSV with these columns:

```text
verified_ai_credential,provider_watermark,unverified_ai_credential,metadata_ai_marker,label
```

Use labelled development data that was not used to train the ConvNeXt model:

```powershell
python scripts/calibrate_provenance.py data/provenance_calibration.csv configs/provenance_calibration.json
$env:TRUESIGHT_PROVENANCE_CALIBRATION = "configs/provenance_calibration.json"
```

At least 30 rows and both classes are required. The script fits a balanced
logistic model and records training metrics. Evaluate it on a held-out set before
calling it deployment-calibrated. Without an artifact, the output explicitly says
`method: policy_v1` and `calibrated: false`.

## Manual verification cases

1. Plain camera JPEG: no positive provenance; continue to Tier 2.
2. Original C2PA image: retain original bytes and inspect the credential.
3. Re-saved C2PA image: missing credentials remain inconclusive.
4. Valid but untrusted C2PA asset: status `detected`; continue downstream.
5. Trusted C2PA AI asset: verified AI signal; fast path allowed.
6. Trusted digital-capture asset: record capture claim but continue downstream.
7. Tampered signed asset: invalid credential; continue downstream.
8. Forged AI-related EXIF: weak unverified metadata only.

The repository includes `data/greencheckmark-trusted-sample.jpg` as a positive
trusted-capture fixture. With the bundled trust list it should produce
`validation_state: trusted`, `provenance_verified: true`, and
`verified_capture_signal: true`.
