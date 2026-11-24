"""Model loading, layer access, activation reads and the training forward pass.

Layer convention: `layer = L` means the residual stream at the OUTPUT of decoder block L (1-indexed),
i.e. `hidden_states[L]` from `output_hidden_states=True`. The paper's "layer 19" of Qwen3.5-9B (32 blocks)
is read here as block 19's output (ambiguous in the paper; flagged in the README).
"""
import torch


def pick_dtype(dtype='auto'):
    """bf16 where the GPU supports it (A100/H100/RTX 30+); fp32 on V100 (no bf16; fp16 overflows Qwen3.5)."""
    if dtype != 'auto':
        return getattr(torch, dtype)
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)         else torch.float32


def load_policy(name, dtype='auto', lora=None, device='cuda', grad_ckpt=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = pick_dtype(dtype) if isinstance(dtype, str) else dtype
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    if lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=0.0,
                         target_modules=lora.get('target_modules', 'all-linear'), task_type='CAUSAL_LM')
        model = get_peft_model(model, cfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if lora:
            model.enable_input_require_grads()
    return model, tok


def causal_lm(model):
    """The underlying *ForCausalLM (unwraps peft). LoRA layers stay active when called through it."""
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def layers(model):
    return causal_lm(model).model.layers


def n_layers(model):
    return len(layers(model))


class _Stop(Exception):
    pass


def _pad_right(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att


@torch.no_grad()
def read_layer(model, seqs, layer, pad, device, batch_size=8):
    """Residual stream at `layer` for every sequence, stopping the forward pass there.
    Returns a list of (len_i, c) float32 CPU tensors."""
    blk = layers(model)[layer - 1]
    store = {}

    def hook(_m, _i, out):
        store['h'] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    hnd = blk.register_forward_hook(hook)
    res = []
    try:
        for k in range(0, len(seqs), batch_size):
            chunk = seqs[k:k + batch_size]
            ids, att = _pad_right(chunk, pad)
            try:
                causal_lm(model).model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False)
            except _Stop:
                pass
            h = store.pop('h').float().cpu()
            res += [h[i, :len(s)] for i, s in enumerate(chunk)]
    finally:
        hnd.remove()
    return res


def check_hook_alignment(model, seq, layer, pad, device):
    """Guard: the hooked/early-exit read must equal output_hidden_states[layer] from a full pass."""
    a = read_layer(model, [seq], layer, pad, device)[0]
    with torch.no_grad():
        out = causal_lm(model).model(input_ids=torch.tensor([seq], device=device), output_hidden_states=True,
                                     use_cache=False)
    b = out.hidden_states[layer][0].float().cpu()
    err = (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)
    assert err < 1e-3, f'hook misaligned with hidden_states[{layer}]: rel err {err}'
    return err


def train_forward(model, seqs, gen_masks, layer, pad, device, score_positions=None, v=None, chunk=512):
    """Differentiable forward for one microbatch.

    Returns (sum of log p(gen token) per sequence (B,), number of gen tokens per sequence (B,),
             per-sequence max probe score with grad (B,) or None).
    Log-probs are computed only at generated positions, in chunks, so the B x T x |V| logits tensor
    (|V| = 248k for Qwen3.5) is never materialised.
    """
    lm = causal_lm(model)
    ids, att = _pad_right(seqs, pad)
    ids, att = ids.to(device), att.to(device)
    store = {}
    hnd = None
    if v is not None:
        def hook(_m, _i, out):
            store['h'] = out[0] if isinstance(out, tuple) else out
        hnd = layers(model)[layer - 1].register_forward_hook(hook)
    try:
        h = lm.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    finally:
        if hnd is not None:
            hnd.remove()
    B = len(seqs)
    lp_sum = torch.zeros(B, device=device, dtype=torch.float32)
    n_tok = torch.zeros(B, device=device)
    for i in range(B):
        pos = torch.tensor([p for p, m in enumerate(gen_masks[i]) if m and p > 0], device=device)
        n_tok[i] = len(pos)
        for k in range(0, len(pos), chunk):
            p = pos[k:k + chunk]
            logits = lm.lm_head(h[i, p - 1]).float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[i, p][:, None])[:, 0]
            lp_sum[i] = lp_sum[i] + lp.sum()
    scores = None
    if v is not None:
        z = store['h']
        vv = v.to(device=device, dtype=z.dtype)
        scores = torch.stack([(z[i, torch.tensor(score_positions[i], device=device)] @ vv).float().max()
                              for i in range(B)])
    return lp_sum, n_tok, scores
