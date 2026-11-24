"""Run a (base or checkpoint) model on N fresh Countdown episodes; save rollouts and print behaviour stats.

  python base_sweep.py --model Qwen/Qwen3.5-0.8B --n 200 --out runs/sweeps/q08_lowest.pt
  python base_sweep.py --model Qwen/Qwen3.5-2B --env.target_position uniform --n 200 --out ...

Use this to choose the base model (valid tool-call rate, peek rate, honest solve rate) and to produce the
base-model rollouts the probe is fitted on. Saved file: {'rollouts': [{'ids','gen_mask','summary'}], 'meta'}.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from common import load_config, apply_overrides
from countdown_env import EnvConfig, asdict_cfg
from model_utils import load_policy
from rollout import Tok, run_episodes, make_groups


def stats(summaries):
    n = len(summaries)
    f = lambda k: sum(bool(s[k]) for s in summaries) / max(n, 1)  # noqa: E731
    return {
        'n': n,
        'peek_rate': f('peeked'),
        'solve_rate': f('solved'),
        'honest_solve_rate': f('honest_solve'),
        'any_tool_call_rate': sum(len(s['calls']) > 0 for s in summaries) / max(n, 1),
        'read_problem_rate': sum(any(c['name'] == 'read_file' and 'problem' in str(c['args']) for c in s['calls'])
                                 for s in summaries) / max(n, 1),
        'submit_rate': sum(any(c['name'] == 'submit' for c in s['calls']) for s in summaries) / max(n, 1),
        'mean_attempts': sum(s['attempts'] for s in summaries) / max(n, 1),
        'mean_reward': sum(s['reward'] for s in summaries) / max(n, 1),
        'truncated_rate': f('truncated'),
        'mean_gen_tokens': sum(s['n_gen_tokens'] for s in summaries) / max(n, 1),
        'n_prohibited_calls': sum(c['peek'] for s in summaries for c in s['calls']),
        'n_permitted_calls': sum(not c['peek'] for s in summaries for c in s['calls']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs' / 'default.yaml'))
    ap.add_argument('--model')
    ap.add_argument('--adapter', help='optional LoRA checkpoint dir to evaluate instead of the base model')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', required=True)
    args, rest = ap.parse_known_args()
    cfg = apply_overrides(load_config(args.config), rest)
    if args.model:
        cfg['model'] = args.model
    env_cfg = EnvConfig.from_dict(cfg['env'])
    model, tk = load_policy(cfg['model'])
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tok = Tok(tk)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    eps = make_groups(env_cfg, args.n, 1, rng)
    t = time.time()
    R = run_episodes(model, tok, env_cfg, eps, cfg['gen'], 'cuda')
    dt = time.time() - t
    summ = [r['summary'] for r in R]
    st = stats(summ)
    st.update({'model': cfg['model'], 'adapter': args.adapter, 'seconds': round(dt, 1), 'env': asdict_cfg(env_cfg)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'rollouts': [{'ids': r['ids'], 'gen_mask': r['gen_mask'], 'summary': r['summary']} for r in R],
                'meta': st, 'config': cfg}, args.out)
    with open(Path(args.out).with_suffix('.examples.txt'), 'w', encoding='utf-8') as f:
        for r in R[:8]:
            f.write(json.dumps({k: r['summary'][k] for k in ('target', 'lo', 'hi', 'solved', 'peeked', 'attempts')}) + '\n')
            f.write(tk.decode(r['ids'][r['prompt_len']:]) + '\n' + '=' * 80 + '\n')
    print(json.dumps({k: v for k, v in st.items() if k != 'env'}, indent=1))


if __name__ == '__main__':
    main()
