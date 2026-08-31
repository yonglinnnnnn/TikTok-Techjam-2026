# Setup Guide

## Prerequisites
- Python 3.12 (recommended)
- Virtual Environment

## Installation

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy the example environment file and add your API keys:
   ```powershell
   copy .env.example .env
   ```
   Edit `.env` to include your `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY` for Tier 2 VLM capabilities.

## Getting Datasets (Optional, for training/evaluation)
If you need to train the ConvNeXt model or evaluate locally, download the CIFAKE and SID-Set datasets:
```powershell
python scripts\data\download_datasets.py
```
*(Use `--skip-cifake` or `--skip-sid` to skip specific datasets).*

## Running the Application
To run the interactive Streamlit UI:
```powershell
python -m streamlit run apps\demo\app.py
```
