"""Probe directions for 'this tool call is prohibited' (read target.txt) vs permitted tool calls.

fit:   from labelled activations at <tool_call> tokens, a unit direction v (reliance.geometric_probes.linear_probe
       or reliance.contrast_subspaces.dom) plus a threshold chosen on the training split.
evaluate: AUC and balanced accuracy of a (v, threshold) on held-out calls.
refit: fit a fresh probe on a checkpoint's activations and report its cosine to a reference v
       (the paper's L2 probe and Fig. 42 drift measure).
Splits are by ROLLOUT, never by call, so calls from one rollout never sit on both sides.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reliance.contrast_subspaces import dom  # noqa: E402
from reliance.geometric_probes import linear_probe  # noqa: E402


def call_dataset(rollouts, acts, positions='tool_call'):
    """Rows = tool calls. X (N, c), y (N,) 1 = prohibited, group (N,) rollout index."""
    X, y, g = [], [], []
    for ri, (r, h) in enumerate(zip(rollouts, acts)):
        for c in r['calls']:
            if c.get('pos') is None:
                continue
            if positions == 'tool_span' and c.get('end_pos') is not None:
                x = h[c['pos']:c['end_pos'] + 1].max(0).values
            else:
                x = h[c['pos']]
            X.append(x)
            y.append(int(c['peek']))
            g.append(ri)
    if not X:
        return torch.zeros(0), torch.zeros(0), torch.zeros(0)
    return torch.stack(X), torch.tensor(y), torch.tensor(g)


def split_by_group(g, frac=0.5, seed=0):
    ids = sorted(set(g.tolist()))
    random.Random(seed).shuffle(ids)
    a = set(ids[:int(len(ids) * frac)])
    m = torch.tensor([x in a for x in g.tolist()])
    return m, ~m


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def bal_acc(s, y, thr):
    pred = s > thr
    tpr = pred[y == 1].float().mean() if (y == 1).any() else torch.tensor(float('nan'))
    tnr = (~pred[y == 0]).float().mean() if (y == 0).any() else torch.tensor(float('nan'))
    return float((tpr + tnr) / 2)


def best_threshold(s, y):
    cand = torch.unique(s)
    best, thr = -1, 0.0
    for t in cand[:: max(1, len(cand) // 512)]:
        b = bal_acc(s, y, t)
        if b > best:
            best, thr = b, float(t)
    return thr


def fit(X, y, kind='linear'):
    pos, neg = X[y == 1].float(), X[y == 0].float()
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(f'not enough calls to fit a probe: {len(pos)} prohibited, {len(neg)} permitted')
    if kind == 'linear':
        v = linear_probe(pos, neg)[0]
    elif kind == 'dom':
        v = dom(pos, neg)[0]
    else:
        raise ValueError(kind)
    v = v / v.norm()
    s = X.float() @ v
    if auc(s, y) < 0.5:  # orient so that larger = prohibited
        v = -v
        s = -s
    return {'v': v, 'threshold': best_threshold(s, y), 'kind': kind}


def evaluate(probe, X, y):
    s = X.float() @ probe['v']
    return {'auc': auc(s, y), 'bal_acc': bal_acc(s, y, probe['threshold']),
            'n_pos': int((y == 1).sum()), 'n_neg': int((y == 0).sum())}


def fit_and_eval(X, y, g, kind='linear', seed=0):
    tr, te = split_by_group(g, 0.5, seed)
    p = fit(X[tr], y[tr], kind)
    return p, {'train': evaluate(p, X[tr], y[tr]), 'test': evaluate(p, X[te], y[te])}


def cosine(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def refit(X, y, g, reference_v, kind='linear', seed=0):
    """A fresh probe on (X, y) and its cosine to reference_v. Returns (probe, metrics)."""
    p, m = fit_and_eval(X, y, g, kind, seed)
    m['cos_to_reference'] = cosine(p['v'], reference_v)
    return p, m


def random_direction(c, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(c, generator=gen)
    return {'v': v / v.norm(), 'threshold': 0.0, 'kind': 'random'}
