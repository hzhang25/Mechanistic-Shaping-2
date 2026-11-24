"""GRPO on agentic Countdown with expression shaping (Fel et al. 2026, Sec. 4 / App. H).

  python train_grpo.py --run_dir runs/grpo_q2b_s0 [--config configs/default.yaml] [--shaping.score_mode detached ...]

Per update:
  1. sample n_groups problems x group_size rollouts from the current policy (multi-turn, tools executed)
  2. task reward r_i = 1 - w + w/n_attempts if solved else 0
  3. probe score M_i = max_k v.z_{i,k} at layer L over the scored positions (paper: "max over token k")
     s_i = (M_i - mean_g M) / std_g M  within the group
     lambda_g = lam * min(1, std_g(r) / sigma_ref)         (vanishes when a group's rewards are all equal)
     r'_i = r_i - lambda_g * s_i
       score_mode detached   : (i)   score read from the current policy, no gradient through it (reward shaping)
       score_mode grad       : (ii)  same, plus the score term is differentiated: loss += lambda_g * s_i(theta) / N
       score_mode frozen_ref : (iii) score read by the frozen base model (LoRA disabled) over the policy's rollouts
                               (paper used a separate 35B reference; ours is the frozen base -- deviation)
     baseline random_v : score with a random unit direction (paper's control)
     baseline filter   : drop peeking rollouts before computing advantages (paper's control)
  4. GRPO advantage A_i = (r'_i - mean_g r') / (std_g r' + eps); one optimiser step per update (fully on-policy,
     so the PPO ratio is 1 and no clipping is needed). Loss = -sum_i A_i sum_t log p(y_it) / total gen tokens.
Outputs (see README): config.yaml, metrics.jsonl, rollouts/update_XXXX.jsonl, ckpt/update_XXXX/,
                      acts/update_XXXX.pt, checkpoint_metrics.jsonl, state/ (resume).
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from common import load_config, apply_overrides
from countdown_env import EnvConfig
from model_utils import load_policy, read_layer, train_forward, check_hook_alignment, n_layers
from probe import call_dataset, evaluate as probe_eval, refit as probe_refit, random_direction, cosine
from rollout import Tok, run_episodes, make_groups

EPS = 1e-6


def score_positions(r, mode):
    s, gm = r['summary'], r['gen_mask']
    pos = []
    if mode in ('tool_call', 'tool_span'):
        for c in s['calls']:
            if c.get('pos') is None:
                continue
            if mode == 'tool_span' and c.get('end_pos') is not None:
                pos += list(range(c['pos'], c['end_pos'] + 1))
            else:
                pos.append(c['pos'])
    if not pos:  # no tool call: fall back to every generated token (NOT FROM PAPER)
        pos = [i for i, m in enumerate(gm) if m]
    return pos or [len(gm) - 1]


def group_std(x):
    return float(torch.tensor(x).std(unbiased=False)) if len(x) > 1 else 0.0


def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


class Monitor:
    """Re-encodes the fixed held-out base-model rollouts with a checkpoint (paper Fig. 17 / 42)."""

    def __init__(self, path, tok, layer):
        d = torch.load(path, weights_only=False)
        assert d['layer'] == layer, f"held-out set was built at layer {d['layer']}, run uses {layer}"
        self.rolls = d['rollouts']
        self.positions = d.get('positions', 'tool_call')
        self.tok, self.layer = tok, layer

    def encode(self, model, frozen=False):
        seqs = [r['ids'] for r in self.rolls]
        if frozen and hasattr(model, 'disable_adapter'):
            with model.disable_adapter():
                acts = read_layer(model, seqs, self.layer, self.tok.pad, 'cuda')
        else:
            acts = read_layer(model, seqs, self.layer, self.tok.pad, 'cuda')
        return call_dataset([r['summary'] for r in self.rolls], acts, self.positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent / 'configs' / 'default.yaml'))
    ap.add_argument('--run_dir', required=True)
    args, rest = ap.parse_known_args()
    run = Path(args.run_dir)
    state_path = run / 'state' / 'state.pt'
    resuming = state_path.exists()
    if resuming:
        cfg = load_config(run / 'config.yaml')
        print(f'resuming {run}')
    else:
        cfg = apply_overrides(load_config(args.config), rest)
    T, S = cfg['train'], cfg['shaping']
    env_cfg = EnvConfig.from_dict(cfg['env'])
    seed = T['seed']
    random.seed(seed)
    torch.manual_seed(seed)

    model, tk = load_policy(cfg['model'], lora=T['lora'], grad_ckpt=T['grad_ckpt'])
    tok = Tok(tk)
    layer = cfg['layer'] or max(1, round(19 / 32 * n_layers(model)))
    cfg['layer'] = layer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T['lr'], weight_decay=0.0)
    if T.get('kl_beta', 0):
        raise NotImplementedError('kl_beta > 0 is not implemented (paper does not specify a KL term)')

    # ---- probe / direction
    probe = None
    if S['score_mode'] != 'none' or S['baseline'] == 'random_v':
        if S['baseline'] == 'random_v':
            probe = random_direction(model.config.get_text_config().hidden_size if hasattr(model.config, 'get_text_config')
                                     else model.config.hidden_size, seed)
        else:
            p = torch.load(S['probe_path'], weights_only=False)
            assert p['layer'] == layer, f"probe layer {p['layer']} != run layer {layer}"
            assert p['model'] == cfg['model'], f"probe fitted on {p['model']}, run uses {cfg['model']}"
            probe = {k: p[k] for k in ('v', 'threshold', 'kind')}
        if S['score_mode'] == 'none':
            S['score_mode'] = 'detached'
    original_probe = dict(probe) if probe else None
    monitor = Monitor(cfg['monitor']['heldout_path'], tok, layer) if cfg['monitor'].get('heldout_path') else None
    if monitor is not None and original_probe is None:
        p = torch.load(S['probe_path'], weights_only=False) if S.get('probe_path') else None
        original_probe = {k: p[k] for k in ('v', 'threshold', 'kind')} if p else None

    start, sigma_ref = 0, S.get('sigma_ref')
    if resuming:
        st = torch.load(state_path, weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, st['adapter'])
        opt.load_state_dict(st['opt'])
        start, sigma_ref, probe = st['update'] + 1, st['sigma_ref'], st['probe']
        random.setstate(st['py_rng'])
        torch.set_rng_state(st['torch_rng'])
        torch.cuda.set_rng_state(st['cuda_rng'])
    else:
        run.mkdir(parents=True, exist_ok=True)
        cfg['git'] = git_hash()
        (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    for d in ('rollouts', 'ckpt', 'acts', 'state'):
        (run / d).mkdir(exist_ok=True)
    rng = random.Random(seed * 100003 + start)

    def checkpoint(u):
        """Save adapter + probe, and re-encode the held-out set (monitorability numbers)."""
        cdir = run / 'ckpt' / f'update_{u:04d}'
        if u > 0:
            model.save_pretrained(cdir)
        cdir.mkdir(parents=True, exist_ok=True)
        if probe is not None:
            torch.save({**probe, 'layer': layer, 'model': cfg['model']}, cdir / 'shaping_probe.pt')
        if monitor is None or original_probe is None:
            return
        model.eval()
        X, y, g = monitor.encode(model)
        torch.save({'X': X.half().unsqueeze(1), 'y': y, 'group': g, 'layer': layer, 'layout': '(n_calls, 1, c)',
                    'update': u}, run / 'acts' / f'update_{u:04d}.pt')
        rec = {'update': u, 'L1_original_probe': probe_eval(original_probe, X, y)}
        try:
            newp, m = probe_refit(X, y, g, original_probe['v'], kind='linear')
            rec['L2_refit_probe'] = m['test']
            rec['refit_cos_to_original'] = m['cos_to_reference']
        except ValueError as e:
            rec['L2_refit_probe'] = str(e)
        Xb = X.float()
        rec['mean_act_norm'] = float(Xb.norm(dim=-1).mean())
        with open(run / 'checkpoint_metrics.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print('[ckpt]', json.dumps(rec))

    if not resuming:
        if probe is not None:
            check_hook_alignment(model, tok.prompt_ids(env_cfg, make_groups(env_cfg, 1, 1, random.Random(0))[0]),
                                 layer, tok.pad, 'cuda')
        checkpoint(0)

    for u in range(start, T['updates']):
        t0 = time.time()
        # ---------------- refit the shaping probe (candidate new experiment)
        refit_info = None
        if S.get('refit_every_n') and u > 0 and u % S['refit_every_n'] == 0 and monitor is not None:
            model.eval()
            X, y, g = monitor.encode(model)
            newp, m = probe_refit(X, y, g, original_probe['v'], kind=probe['kind'] if probe['kind'] != 'random' else 'linear')
            probe = {'v': newp['v'], 'threshold': newp['threshold'], 'kind': newp['kind']}
            refit_info = {'test': m['test'], 'cos_to_original': m['cos_to_reference']}

        # ---------------- rollouts
        model.eval()
        eps = make_groups(env_cfg, T['n_groups'], T['group_size'], rng)
        R = run_episodes(model, tok, env_cfg, eps, cfg['gen'], 'cuda')
        t_gen = time.time() - t0
        N, G = len(R), T['group_size']
        r = [x['summary']['reward'] for x in R]
        if sigma_ref is None and u == 0:
            sigma_ref = group_std(r) or None  # paper: initial reward std; None until rewards vary
        if sigma_ref is None and group_std(r) > 0:
            sigma_ref = group_std(r)

        # ---------------- probe scores
        pos = [score_positions(x, S['score_positions']) for x in R]
        M = [0.0] * N
        if probe is not None:
            seqs = [x['ids'] for x in R]
            if S['score_mode'] == 'frozen_ref':
                with model.disable_adapter():
                    acts = read_layer(model, seqs, layer, tok.pad, 'cuda')
            else:
                acts = read_layer(model, seqs, layer, tok.pad, 'cuda')
            M = [float((a[p] @ probe['v']).max()) for a, p in zip(acts, pos)]

        # ---------------- shaped rewards and advantages, per group
        keep = [not (S['baseline'] == 'filter' and x['summary']['peeked']) for x in R]
        s_std, lam_g, rp, A = [0.0] * N, [0.0] * N, list(r), [0.0] * N
        M_mean, M_sd = [0.0] * N, [1.0] * N
        for g0 in range(0, N, G):
            idx = [i for i in range(g0, g0 + G) if keep[i]]
            if not idx:
                continue
            if probe is not None:
                mm = sum(M[i] for i in idx) / len(idx)
                sd = group_std([M[i] for i in idx])
                lg = S['lam'] * min(1.0, group_std([r[i] for i in idx]) / sigma_ref) if sigma_ref else 0.0
                for i in idx:
                    s_std[i] = (M[i] - mm) / (sd + EPS)
                    lam_g[i], M_mean[i], M_sd[i] = lg, mm, sd + EPS
                    rp[i] = r[i] - lg * s_std[i]
            mu = sum(rp[i] for i in idx) / len(idx)
            sd = group_std([rp[i] for i in idx])
            for i in idx:
                A[i] = (rp[i] - mu) / (sd + 1e-4) if sd > 0 else 0.0

        # ---------------- policy-gradient step (one optimiser step per update)
        model.train()
        opt.zero_grad(set_to_none=True)
        active = [i for i in range(N) if A[i] != 0.0 or (S['score_mode'] == 'grad' and lam_g[i] > 0)]
        tot_tok = max(1, sum(sum(R[i]['gen_mask']) for i in active))
        mb = T['micro_batch']
        loss_pg, loss_sc = 0.0, 0.0
        for k in range(0, len(active), mb):
            ids_ = active[k:k + mb]
            grad_score = S['score_mode'] == 'grad' and probe is not None
            lp, nt, sc = train_forward(model, [R[i]['ids'] for i in ids_], [R[i]['gen_mask'] for i in ids_], layer,
                                       tok.pad, 'cuda', score_positions=[pos[i] for i in ids_] if grad_score else None,
                                       v=probe['v'] if grad_score else None)
            adv = torch.tensor([A[i] for i in ids_], device=lp.device)
            loss = -(adv * lp).sum() / tot_tok
            loss_pg += float(loss)
            if grad_score:
                lgs = torch.tensor([lam_g[i] / M_sd[i] for i in ids_], device=sc.device)
                ls = (lgs * sc).sum() / N
                loss = loss + ls
                loss_sc += float(ls)
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], T['max_grad_norm'])
        if math.isfinite(float(gn)):
            opt.step()
        else:
            print(f'update {u}: non-finite grad norm, step skipped')

        # ---------------- logging
        summ = [x['summary'] for x in R]
        peek_turns = Counter(min(s['peek_steps']) for s in summ if s['peek_steps'])
        met = {
            'update': u, 'peek_rate': sum(s['peeked'] for s in summ) / N,
            'honest_solve_rate': sum(s['honest_solve'] for s in summ) / N,
            'solve_rate': sum(s['solved'] for s in summ) / N,
            'mean_reward': sum(r) / N, 'mean_shaped_reward': sum(rp) / N,
            'mean_attempts': sum(s['attempts'] for s in summ) / N,
            'truncated_rate': sum(s['truncated'] for s in summ) / N,
            'mean_gen_tokens': sum(s['n_gen_tokens'] for s in summ) / N,
            'mean_M': sum(M) / N, 'mean_M_peek': (sum(M[i] for i in range(N) if summ[i]['peeked']) /
                                                  max(1, sum(s['peeked'] for s in summ))),
            'mean_M_nopeek': (sum(M[i] for i in range(N) if not summ[i]['peeked']) /
                              max(1, sum(not s['peeked'] for s in summ))),
            'mean_lambda_g': sum(lam_g) / N, 'sigma_ref': sigma_ref,
            'frac_zero_adv': sum(a == 0.0 for a in A) / N,
            'first_peek_turn_hist': dict(sorted(peek_turns.items())),
            'loss_pg': loss_pg, 'loss_score': loss_sc, 'grad_norm': float(gn),
            'refit': refit_info, 'sec_gen': round(t_gen, 1), 'sec_total': round(time.time() - t0, 1),
        }
        with open(run / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(met) + '\n')
        with open(run / 'rollouts' / f'update_{u:04d}.jsonl', 'w') as f:
            for i, s in enumerate(summ):
                f.write(json.dumps({**s, 'group': i // G, 'M': M[i], 's': s_std[i], 'lambda_g': lam_g[i],
                                    'shaped_reward': rp[i], 'advantage': A[i]}) + '\n')
        print(f"[u{u:03d}] peek {met['peek_rate']:.2f} honest {met['honest_solve_rate']:.2f} "
              f"r {met['mean_reward']:.3f} r' {met['mean_shaped_reward']:.3f} M(peek/no) "
              f"{met['mean_M_peek']:.2f}/{met['mean_M_nopeek']:.2f} lam_g {met['mean_lambda_g']:.3f} "
              f"gen {t_gen:.0f}s tot {met['sec_total']:.0f}s", flush=True)

        if (u + 1) % T['ckpt_every'] == 0 or u + 1 == T['updates']:
            checkpoint(u + 1)
        from peft import get_peft_model_state_dict
        torch.save({'adapter': get_peft_model_state_dict(model), 'opt': opt.state_dict(), 'update': u,
                    'sigma_ref': sigma_ref, 'probe': probe, 'py_rng': random.getstate(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()},
                   str(state_path) + '.tmp')
        os.replace(str(state_path) + '.tmp', state_path)
        rng = random.Random(seed * 100003 + u + 1)
    (run / 'DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
