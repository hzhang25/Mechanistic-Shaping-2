# Countdown expression shaping

GRPO on agentic Countdown with expression shaping, written from Fel et al. 2026 Sec. 4 / App. H. The upstream repo has no RL code.

## Run

```bash
source /project/cmsc35428/mechshaping/env.sh   # shared env + runs, already set up
cd experiments/countdown
python base_sweep.py --model Qwen/Qwen3.5-2B --n 400 --out $CD_ROOT/runs/sweeps/q2b.pt
python fit_probe.py --sweep $CD_ROOT/runs/sweeps/q2b.pt --out $CD_ROOT/runs/probes/q2b
python train_grpo.py --run_dir $CD_ROOT/runs/<name> \
  --shaping.score_mode detached \
  --shaping.probe_path $CD_ROOT/runs/probes/q2b/probe_linear.pt \
  --monitor.heldout_path $CD_ROOT/runs/probes/q2b/heldout.pt
```

On Slurm, prefix any command with `sbatch slurm/job.sbatch`. A training job that hits the time limit resumes when you resubmit it.

## Flags

- `--shaping.score_mode`: `none` (plain GRPO) | `detached` (i) | `grad` (ii) | `frozen_ref` (iii)
- `--shaping.baseline`: `none` | `random_v` | `filter`
- `--shaping.lam` (0.4), `--shaping.refit_every_n` (0 = never)
- `--env.target_position`: `lowest` | `uniform`; `--env.target_file`: `true` | `false`
- Any other key in `configs/default.yaml`, dotted. Values not from the paper are marked `NOT FROM PAPER`.
- New task variant: add a field to `EnvConfig` and pass it as `--env.<field>`.

## Outputs (`run_dir/`)

- `metrics.jsonl`: per update. Peek rate, honest solve rate, reward, shaped reward, probe score on peeking vs non-peeking rollouts, histogram of the turn of the first peek.
- `rollouts/update_XXXX.jsonl`: per rollout. Every tool call with its turn and the token position of its `<tool_call>`, plus reward, `M`, `s`, `lambda_g`, and advantage.
- `ckpt/update_XXXX/`: LoRA adapter and the probe in use at that point, every 10 updates.
- `acts/update_XXXX.pt`: held-out base-model rollouts re-encoded by that checkpoint. `X` has shape (n_calls, 1, c), `y` = 1 for a prohibited call, `group` = rollout id.
- `checkpoint_metrics.jsonl`: original-probe accuracy (L1), refit-probe accuracy (L2), and the refit probe's cosine to the original.

## Deviations from the paper

- Model: smaller than Qwen3.5-9B, chosen from the base sweep.
- Scale: 60 updates × 128 rollouts (8 groups of 16), not 150 × 512 (16 groups of 32).
- Frozen reference: the base policy with LoRA disabled, not Qwen3.6-35B.
- DoM: fitted on our base rollouts, not Bergen et al.'s SWE data.
- Layer: `round(19/32 · n_layers)`.
- Thinking: off.
- LoRA r=64, lr 2e-5, no KL. The paper doesn't report these.
