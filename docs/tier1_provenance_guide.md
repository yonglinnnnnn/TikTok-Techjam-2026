# Tier 1 provenance implementation guide

## 1. Install

Use Python 3.10 or newer from the repository root. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of
`python`; changing the machine execution policy is not required.

## 2. Run Tier 1

```powershell
$env:PYTHONPATH = "src"
python -m truesight.provenance.tier1 data/samples/example.jpg
```

`c2pa_checker.py` reads the JUMBF manifest store, extracts the active provenance
chain and asks the SDK for its validation state. `Trusted` means valid plus a
trusted active signer; `Valid` alone is deliberately not treated as verified.

`metadata_checker.py` reads EXIF/XMP/software strings. These strings are easy to
forge, so they contribute only a `0.30` severity weight. An unverified C2PA store
contributes `0.35`. A trusted, AI-attributed C2PA store contributes `0.98` and can
quick-exit. Tune these provisional weights against a labelled validation set,
then calibrate the final fused score (isotonic regression or Platt scaling are
reasonable choices). Do not add the weights together: correlated provenance
signals would inflate confidence.

The aggregator uses `null` for fields Tier 1 cannot know. Tier 2/3 should update
the same object rather than creating a different payload. In particular,
`ai_coverage` and `heatmap_path` remain null until the classifier/Grad-CAM stage.

## 3. Watermark and vendor checks

Vendor-specific image watermark checks should be added only when the team has an
official detector/API and any required detection key.

Meta Stable Signature is not a generic marker. Detection requires Meta's TorchScript
extractor and the exact bit key used to watermark the model. Its research repo was
built with an older PyTorch/CUDA stack and most of it is non-commercially licensed,
so isolate it in a separate environment/service and review the licence before use.
Run its documented bit-accuracy evaluation with your extractor and key, calibrate
a threshold on known positive and negative images, and expose only a small adapter:

```python
def check_stable_signature(path: str) -> dict:
    bit_accuracy = detector_score(path)  # your licensed model/key implementation
    return {
        "checked": True,
        "present": bit_accuracy >= CALIBRATED_THRESHOLD,
        "verified": bit_accuracy >= CALIBRATED_THRESHOLD,
        "source": "Meta Stable Signature",
        "score": bit_accuracy,
    }
```

Never search image bytes for vendor names and call that a binary watermark. Byte
inspection is useful only for locating metadata/container segments; cryptographic
verification and pixel watermark extraction require their respective verifier.

## 4. Manual verification checklist

1. **Plain negative:** use a camera/handmade JPEG with no credentials. Expect
   `is_ai_generated: null`, confidence `0`, and `requires_tier2: true`.
2. **Verified positive:** download an original OpenAI-generated image without
   screenshotting or re-saving it. Check it in the vendor verifier, then run the
   CLI. Expect C2PA present, ideally `Trusted`, an OpenAI/DALL-E source, and quick
   exit. Trust stores can affect whether the state is `Valid` or `Trusted`.
3. **Stripped copy:** re-save or screenshot that image. Expect provenance to be
   absent and escalation to Tier 2. This confirms that absence is not classified
   as authentic.
4. **Tampered copy:** modify bytes/pixels while retaining the manifest. Expect an
   invalid/unverified state and no quick exit.
5. **Forged metadata:** write `Software=OpenAI` into an ordinary image. Expect only
   weak metadata evidence, `is_ai_generated: null`, and no quick exit.
6. **Bad input:** pass a missing path (clear `FileNotFoundError`) and a text file
   renamed `.jpg` (provenance/metadata errors, never a positive verdict). Input
   MIME/dimension/size validation should eventually run before this module.
7. Record output and wall-clock time for a folder of labelled samples. Compare
   false positives and false negatives by signal type before changing weights.

For each manual case, validate that the top-level keys stay identical. Downstream
tiers may replace nulls and append evidence, but should not remove or rename keys.
