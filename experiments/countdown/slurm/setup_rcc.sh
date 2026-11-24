#!/bin/bash
# One-time environment setup on RCC Midway3. Run on a login node:  bash experiments/countdown/slurm/setup_rcc.sh
# Creates a venv on scratch (home quota is small) and points the HF cache at scratch too.
set -euo pipefail
umask 002
: "${CD_ROOT:=/project/cmsc35428/mechshaping}"   # shared with the team (group cmsc35428)
mkdir -p "$CD_ROOT"/{hf,runs}
module load python/miniforge-25.3.0
python3 -m venv "$CD_ROOT/venv"
source "$CD_ROOT/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers>=5.20" peft einops pyyaml accelerate
# fast kernels for Qwen3.5's linear-attention layers (pure-triton; falls back to slow torch code without it)
pip install -q flash-linear-attention || echo "flash-linear-attention failed to install; will run on the slow path"
pip install -q -e "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cat > "$CD_ROOT/env.sh" <<EOF
module load python/miniforge-25.3.0
umask 002
source $CD_ROOT/venv/bin/activate
export HF_HOME=$CD_ROOT/hf
export CD_ROOT=$CD_ROOT
EOF
echo "done. Every job sources $CD_ROOT/env.sh"
