#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dabench.py
echo "Setup complete. Copy .env.example to .env and add your API keys."
