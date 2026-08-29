# TikTok TechJam 2026 — TrueSight

TrueSight is an early-stage image-analysis pipeline for detecting AI-generated
images. The current pipeline returns placeholder results while the detection
models are being integrated.

## Run predictions

Run commands from the repository root so Python can resolve the project
packages correctly:

```powershell
python -m scripts.predict --input_dir data/samples --output predictions.json
```

## Run tests

The test suite uses Python's built-in `unittest` module, so no extra test
dependency is required:

```powershell
python -m unittest discover -s tests -v
```
