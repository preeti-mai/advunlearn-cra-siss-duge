#!/usr/bin/env bash
set -e
# Create and activate the exact environment used by AdvUnlearn
conda env create -f vendor/AdvUnlearn/environment.yaml
conda activate AdvUnlearn

# (Optional) also install extras we use in wrappers
pip install pyyaml tqdm matplotlib
echo "[ok] environment ready"
