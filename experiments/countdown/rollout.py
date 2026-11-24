"""Batched multi-turn rollouts with tool execution, token-exact bookkeeping.

Each rollout is kept as one token sequence (prompt, then alternating policy turns and tool responses),
built by concatenating token ids rather than re-rendering the chat template, so the sequence the
trainer scores is exactly the one the policy sampled. `gen_mask[t] = 1` marks tokens the policy
generated (the only ones that get a policy-gradient term). Every tool call records the absolute
position of its `<tool_call>` token and of its `</tool_call>` token, which is where the probe reads.
"""
import random
import torch

from countdown_env import TOOLS, system_prompt, user_prompt, make_episode, step, parse_tool_calls


class Tok:
    """Token ids and the exact string pieces the Qwen3.5 template uses between turns."""

    def __init__(self, tokenizer):
        self.tk = tokenizer
        c = tokenizer.convert_tokens_to_ids
        self.im_end = c('<|im_end|>')
        self.eot = c('<|endoftext|>')
        self.call_open = c('<tool_call>')
        self.call_close = c('</tool_call>')
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eot

    def prompt_ids(self, cfg, ep):
        msgs = [{'role': 'system', 'content': system_prompt(cfg)}, {'role': 'user', 'content': user_prompt(ep)}]
        text = self.tk.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)  # NOT FROM PAPER: thinking off
        return self.tk(text, add_special_tokens=False).input_ids

    def tool_suffix_ids(self, responses):
        s = '\n<|im_start|>user' + ''.join(f'\n<tool_response>\n{r}\n</tool_response>' for r in responses)
        s += '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return self.tk(s, add_special_tokens=False).input_ids


@torch.no_grad()
def generate_batch(model, tok, seqs, max_new_tokens, temperature, top_p, device):
    """Left-pad, sample one turn for each sequence, return the new tokens (cut at <|im_end|>)."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        att[i, L - len(s):] = 1
    out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                         eos_token_id=[tok.im_end, tok.eot], pad_token_id=tok.pad)
    new = out[:, L:].tolist()
    res = []
    for row in new:
        cut = []
        for t in row:
            cut.append(t)
            if t in (tok.im_end, tok.eot):
                break
        res.append(cut)
    return res


def run_episodes(model, tok, env_cfg, episodes, gen_cfg, device):
    """Play every episode to completion. Returns one dict per rollout."""
    R = []
    for ep in episodes:
        ids = tok.prompt_ids(env_cfg, ep)
        R.append({'ep': ep, 'ids': list(ids), 'gen_mask': [0] * len(ids), 'prompt_len': len(ids),
                  'call_pos': [], 'truncated': False})
    active = list(range(len(R)))
    bs = gen_cfg.get('gen_batch_size', 64)
    while active:
        news = []
        for k in range(0, len(active), bs):
            chunk = active[k:k + bs]
            news += generate_batch(model, tok, [R[i]['ids'] for i in chunk], gen_cfg['max_new_tokens'],
                                   gen_cfg['temperature'], gen_cfg['top_p'], device)
        still = []
        for i, new in zip(active, news):
            r, ep = R[i], R[i]['ep']
            start = len(r['ids'])
            r['ids'] += new
            r['gen_mask'] += [1] * len(new)
            ep.turns += 1
            opens = [start + j for j, t in enumerate(new) if t == tok.call_open]
            closes = [start + j for j, t in enumerate(new) if t == tok.call_close]
            calls = parse_tool_calls(tok.tk.decode(new, skip_special_tokens=False))
            ended = new and new[-1] in (tok.im_end, tok.eot)
            if not ended:
                r['truncated'] = True
            if not calls or not ended:
                ep.done = True
                continue
            responses = []
            for c_idx, (name, args) in enumerate(calls):
                if ep.done:
                    break
                n_before = len(ep.calls)
                responses.append(step(ep, name, args))
                rec = ep.calls[n_before]
                rec['pos'] = opens[c_idx] if c_idx < len(opens) else None
                rec['end_pos'] = closes[c_idx] if c_idx < len(closes) else None
            if ep.done or ep.turns >= env_cfg.max_turns:
                ep.done = True
                continue
            suffix = tok.tool_suffix_ids(responses)
            if len(r['ids']) + len(suffix) + gen_cfg['max_new_tokens'] > gen_cfg['max_seq_len']:
                ep.done, r['truncated'] = True, True
                continue
            r['ids'] += suffix
            r['gen_mask'] += [0] * len(suffix)
            still.append(i)
        active = still
    for r in R:
        # trim a trailing turn-end so the sequence ends on the last policy token
        s = r['ep'].summary()
        s['truncated'] = r['truncated']
        s['n_tokens'] = len(r['ids'])
        s['n_gen_tokens'] = sum(r['gen_mask'])
        s['peek_steps'] = [c['turn'] for c in s['calls'] if c['peek']]
        s['first_peek_pos'] = next((c['pos'] for c in s['calls'] if c['peek']), None)
        r['summary'] = s
    return R


def make_groups(env_cfg, n_groups, group_size, rng):
    """GRPO groups: group_size rollouts of the same problem."""
    eps = []
    for g in range(n_groups):
        base = make_episode(env_cfg, rng)
        for _ in range(group_size):
            eps.append(type(base)(numbers=list(base.numbers), target=base.target, lo=base.lo, hi=base.hi,
                                  solution=base.solution, cfg=env_cfg))
    return eps
