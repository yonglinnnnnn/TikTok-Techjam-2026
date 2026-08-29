# Tier 1 provenance implementation guide

## Installation

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The local C2PA and forensic checks work without an API key. To enable OpenAI's
image provenance check, set the key in the process environment:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

The official endpoint checks supported OpenAI image signals and can return both
C2PA and image SynthID results:
https://developers.openai.com/api/reference/python/resources/content_provenance_checks/methods/create

The removed `synthid-text` package is not used. No separate key-dependent
watermark provider is registered.

## Running

```powershell
$env:PYTHONPATH = "src"
python -m truesight.provenance.tier1 data/image.jpg
```

Useful flags:

```text
--skip-openai-provenance   Avoid the remote provider check
--skip-forensics           Run provenance only
--debug-provenance         Include raw manifest/validation/metadata diagnostics
--debug-forensics          Include individual forensic maps and thresholds
--calibration PATH         Load a fitted severity calibration artifact
```

Always run Tier 1 on the untouched uploaded bytes. Any RGB conversion, EXIF
orientation, resize, or recompression must produce a separate file for Tier 2/3.

## C2PA result

`provenance.c2pa.history.chain` contains:

- the active/root manifest;
- one node per reachable manifest, plus disconnected nodes for audit;
- ingredient edges with relationship and title;
- missing references and cycles;
- unsigned ingredient count;
- an oldest-to-active declared-action timeline.

`content_edited` and `transformed` describe signed, declared actions. Their being
false does not prove that no undeclared edit ever occurred.

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
2. Original supported OpenAI image: retain original bytes; inspect separate C2PA
   and SynthID signals.
3. Re-saved OpenAI image: missing signals remain inconclusive.
4. Trusted C2PA AI asset: verified AI signal; fast path allowed.
5. Trusted digital-capture asset: record capture claim but continue downstream.
6. Tampered signed asset: invalid credential; continue downstream.
7. Forged `Software=OpenAI` EXIF: weak unverified metadata only.
8. No API key: OpenAI provider status `unavailable`, not `not_detected`.
