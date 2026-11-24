#!/bin/bash
# Generic GPU job: runs whatever command follows, inside the countdown env, from experiments/countdown/.
#   sbatch experiments/countdown/slurm/job.sbatch python base_sweep.py --model Qwen/Qwen3.5-2B --n 200 --out $CD_ROOT/runs/sweeps/q2b.pt
#   sbatch --time=12:00:00 experiments/countdown/slurm/job.sbatch python train_grpo.py --run_dir $CD_ROOT/runs/grpo_q2b_s0 ...
# train_grpo.py resumes from runs/<run>/state/ automatically, so a timed-out training job can simply be resubmitted.
#SBATCH --account=cmsc35428
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# gpu partition = V100 32GB nodes (fp32 here) + one A100 node: add  --constraint=a100  to sbatch to ask for it
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
: "${CD_ROOT:=/project/cmsc35428/mechshaping}"
source "$CD_ROOT/env.sh"
cd "$SLURM_SUBMIT_DIR/experiments/countdown" 2>/dev/null || cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "cmd: $*"
"$@"
