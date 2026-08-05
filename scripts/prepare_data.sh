#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw data/processed
python -c 'from pathlib import Path; Path("data/processed/READY").touch()'
