"""Config loading. YAML file + dotted command-line overrides (--env.target_position uniform)."""
import copy
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _parse(v):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def apply_overrides(cfg, rest):
    """rest: leftover argv like ['--env.target_position', 'uniform', '--shaping.lam=0.4']."""
    cfg = copy.deepcopy(cfg)
    i = 0
    while i < len(rest):
        a = rest[i]
        if not a.startswith('--'):
            raise SystemExit(f'bad override {a}')
        if '=' in a:
            k, v = a[2:].split('=', 1)
            i += 1
        else:
            k, v = a[2:], rest[i + 1]
            i += 2
        d = cfg
        parts = k.split('.')
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _parse(v)
    return cfg
