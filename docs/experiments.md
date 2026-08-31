# Experiments and Evaluation

## Pipeline Evaluation

You can evaluate the full TrueSight integrated pipeline (Tier 1 to 3) on a directory of images using the `evaluate.py` script. 

### Running an Evaluation

```powershell
python scripts\evaluate.py --input_dir data\samples --output predictions.json
```

This will run the pipeline recursively on all supported images in the target directory. The results, including the final prediction confidence and metadata, will be stored in the specified JSON file.

## Dataset Management

The `scripts/data` directory provides tools for managing the datasets used in experiments:

- `download_datasets.py`: Downloads the CIFAKE dataset (from Kaggle) and the SID-Set (from Hugging Face) and creates CSV manifests in `data/processed/`.
- `make_experiment_manifests.py`: Tooling to organize and prepare datasets for different experiment splits.

## Running Calibrations

You can calibrate individual tiers of the system using the calibration scripts in the `scripts` folder:
- `scripts/calibrate_provenance.py`
- `scripts/calibrate_vlm.py`
