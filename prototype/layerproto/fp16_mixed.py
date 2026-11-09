"""Run the Qwen-Image(-Layered) transformer on GPUs without bf16 (Kaggle T4, P100, RTX 20xx) without NaNs.

Qwen-Image is trained for bf16; its hidden values exceed the fp16 limit (65504), so a plain fp16 run gives all-NaN
latents (ComfyUI also lists Qwen-Image as bf16/fp32 only). This keeps the fast fp16 tensor-core matmuls but holds
everything else in fp32:

  * every Linear: input scaled down to absmax 32, fp16 matmul, result scaled back in fp32
    (falls back to a fp32 matmul for that call if the fp16 result is still non-finite)
  * attention: q/k in fp16 (RMS-normalised, small), v scaled the same way (attention is linear in v)
  * residual stream, norms, gates, embeddings: fp32
  * the model's output is returned in fp16, so the diffusers pipeline is unchanged

Measured on Kaggle 2x T4 (Q4_K_S GGUF, 640 px, 4 layers): no NaN, 42 s/step, peak 8.8 / 8.4 GB.

    from layerproto.fp16_mixed import apply_fp16_mixed
    apply_fp16_mixed(transformer)        # once, after loading; idempotent
"""
import types

import torch
import torch.nn.functional as F

LINEAR_LIMIT = 32.0  # absmax the fp16 matmul input is scaled to (1024 was still too high: NaNs)
ATTN_LIMIT = 32.0


def _dequant(w):
    if type(w).__name__ == "GGUFParameter":
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        return dequantize_gguf_tensor(w)
    return w


def _mixed_linear(self, x):
    x = x.float()
    s = (x.abs().amax() / LINEAR_LIMIT).clamp_(min=1.0)
    y = self._fp16_orig_forward((x / s).to(torch.float16)).float()   # = W x/s + b
    b = getattr(self, "bias", None)
    if not torch.isfinite(y).all():                                  # rare: this call in fp32
        return F.linear(x, _dequant(self.weight).float(), None if b is None else b.float())
    if b is None:
        return y * s
    b = b.float()
    return (y - b) * s + b


def _patch_attention():
    import diffusers.models.transformers.transformer_qwenimage as tq
    if getattr(tq, "_fp16_orig_attn", None) is not None:
        return
    orig = tq.dispatch_attention_fn

    def attn_fp16(q, k, v, attn_mask=None, **kw):
        s = (v.float().abs().amax() / ATTN_LIMIT).clamp_(min=1.0)
        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(torch.float16)
        o = orig(q.to(torch.float16), k.to(torch.float16), (v.float() / s).to(torch.float16), attn_mask=attn_mask, **kw)
        return o.float() * s

    tq._fp16_orig_attn = orig
    tq.dispatch_attention_fn = attn_fp16


def apply_fp16_mixed(model):
    """Patch a loaded QwenImageTransformer2DModel in place. Returns (linears patched, params moved to fp32)."""
    _patch_attention()
    n_lin = n_par = 0
    for _, m in model.named_modules():
        if "Linear" in type(m).__name__:
            if not hasattr(m, "_fp16_orig_forward"):
                m._fp16_orig_forward = m.forward
            m.forward = types.MethodType(_mixed_linear, m)
            n_lin += 1
        else:
            for _, p in m.named_parameters(recurse=False):
                if p.dtype == torch.float16:
                    p.data = p.data.float()
                    n_par += 1
    if not hasattr(model, "_fp16_orig_model_forward"):
        model._fp16_orig_model_forward = model.forward

        def fwd(*a, **k):
            o = model._fp16_orig_model_forward(*a, **k)
            if isinstance(o, tuple):
                return (o[0].to(torch.float16),) + tuple(o[1:])
            if hasattr(o, "sample"):
                o.sample = o.sample.to(torch.float16)
            return o

        model.forward = fwd
    return n_lin, n_par
