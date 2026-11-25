#!/bin/bash
# Submits the whole reproduction on A100s: probe data + probe + timing smoke, then 3 training runs (afterok).
# Run on a login node from the repo root:  bash experiments/countdown/slurm/pipeline_9b.sh
set -euo pipefail
R=/project/cmsc35428/mechshaping/runs
J=experiments/countdown/slurm/job.sbatch
A="--constraint=a100 --mem=96G"
P=$R/probes/q9B
PROBE="--shaping.probe_path $P/probe_linear.pt --monitor.heldout_path $P/heldout.pt"
prep=$(sbatch --parsable -J prep9B -o ../logs/prep9B-%j.out $A --time=03:00:00 $J bash -c "
  python base_sweep.py --model Qwen/Qwen3.5-9B --n 300 --seed 99 --out $R/sweeps/q9B_b.pt &&
  python fit_probe.py --sweep $R/sweeps/q9B.pt $R/sweeps/q9B_b.pt --out $P --kind linear &&
  python fit_probe.py --sweep $R/sweeps/q9B.pt $R/sweeps/q9B_b.pt --out $P --kind dom &&
  python train_grpo.py --run_dir $R/smoke9B --train.updates 1 --train.n_groups 2 --train.group_size 8 --shaping.score_mode grad $PROBE")
echo "prep job $prep"
for mode in none detached grad; do
  j=$(sbatch --parsable -J g9B_$mode -o ../logs/grpo9B_$mode-%j.out $A --time=30:00:00 --dependency=afterok:$prep \
      $J python train_grpo.py --run_dir $R/grpo9B_${mode}_s0 --shaping.score_mode $mode $PROBE)
  echo "train $mode job $j"
done
