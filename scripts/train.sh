#!/usr/bin/env bash
set -euo pipefail
python -m coevo_drl.cli train --steps 1000000 --rollout-steps 2048 --device cuda --output coevo_drl.pt
