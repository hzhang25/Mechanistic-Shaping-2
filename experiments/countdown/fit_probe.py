"""Fit the 'prohibited tool call' probe on base-model rollouts and write the held-out monitor set.

  python fit_probe.py --sweep runs/sweeps/q2b_lowest.pt --out runs/probes/q2b --kind linear [--layer 14]

Rollouts are split 50/50 by rollout:
  half A -> fit the probe (activations at <tool_call> tokens; label = the call reads target.txt)
  half B -> held-out monitor set (token ids + call positions). train_grpo.py re-encodes it with every
            checkpoint: original-probe accuracy (paper's L1), refit-probe accuracy and cosine (L2, Fig. 42).
Writes: <out>/probe_<kind>.pt  {'v','threshold','kind','layer','model','metrics'}
        <out>/heldout.pt        {'rollouts': [...], 'layer', 'model'}
        <out>/probe_<kind>.json  human-readable metrics
"""
import argparse
import json
from pathlib import Path

import torch

from model_utils import load_policy, read_layer, check_hook_alignment, n_layers
from probe import call_dataset, fit_and_eval, split_by_group
from rollout import Tok


def default_layer(model):
    return max(1, round(19 / 32 * n_layers(model)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', nargs='+', required=True, help='one or more base_sweep.py outputs (same model)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--kind', default='linear', choices=['linear', 'dom'])
    ap.add_argument('--layer', type=int)
    ap.add_argument('--positions', default='tool_call', choices=['tool_call', 'tool_span'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    data = [torch.load(p, weights_only=False) for p in args.sweep]
    model_name = data[0]['meta']['model']
    rolls = [r for d in data for r in d['rollouts']]
    model, tk = load_policy(model_name)
    model.eval()
    tok = Tok(tk)
    layer = args.layer or default_layer(model)
    err = check_hook_alignment(model, rolls[0]['ids'][:512], layer, tok.pad, 'cuda')
    print(f'layer {layer}/{n_layers(model)}; hook alignment rel err {err:.2e}')

    acts = read_layer(model, [r['ids'] for r in rolls], layer, tok.pad, 'cuda')
    X, y, g = call_dataset([r['summary'] for r in rolls], acts, args.positions)
    print(f'{len(y)} tool calls: {int(y.sum())} prohibited, {int((1 - y).sum())} permitted, from {len(rolls)} rollouts')
    if y.sum() < 20:
        print('WARNING: fewer than 20 prohibited calls; the probe will be noisy. Collect more base rollouts '
              '(or a variant that peeks more, e.g. --env.forbid_text "") before trusting it.')

    a_mask, b_mask = split_by_group(g, 0.5, args.seed)
    a_ids = sorted(set(g[a_mask].tolist()))
    b_ids = sorted(set(g[b_mask].tolist()))
    from probe import fit, evaluate
    try:  # within-half-A split, for an honest train/test number
        _, metrics = fit_and_eval(X[a_mask], y[a_mask], g[a_mask], args.kind, args.seed)
    except ValueError as e:
        metrics = {'within_A_split': f'skipped: {e}'}
    full = fit(X[a_mask], y[a_mask], args.kind)
    metrics['train_A'] = evaluate(full, X[a_mask], y[a_mask])
    metrics['heldout_B'] = evaluate(full, X[b_mask], y[b_mask])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {'v': full['v'], 'threshold': full['threshold'], 'kind': args.kind, 'layer': layer, 'model': model_name,
           'positions': args.positions, 'metrics': metrics, 'sweeps': args.sweep}
    torch.save(rec, out / f'probe_{args.kind}.pt')
    torch.save({'rollouts': [rolls[i] for i in b_ids], 'layer': layer, 'model': model_name,
                'positions': args.positions}, out / 'heldout.pt')
    (out / f'probe_{args.kind}.json').write_text(json.dumps({k: v for k, v in rec.items() if k != 'v'}, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f'wrote {out}/probe_{args.kind}.pt and heldout.pt ({len(b_ids)} rollouts)')


if __name__ == '__main__':
    main()
