# MLX attention/FFN bridge: runs bs_roformer transformer blocks (rotary + gate)
# on MPS by round-tripping torch<->mx. Rotary has a Metal-kernel fast path.
import torch
from ..mlx_backend import (torch_dtype,)
from ..mlx_backend import (gelu as _gelu,)
from ..mlx_backend import (linear as _linear,)
from ..mlx_backend import (mx_dtype as _mlx_dtype,)
from ..mlx_backend import (rms_norm as _rms_norm,)
from ..mlx_backend import (sigmoid as _sigmoid,)
from ..mlx_backend import (to_mx as _torch_to_mlx_array,)
from ..mlx_backend import (to_mx_raw as torch_mps_to_mlx,)
from ..mlx_backend import (to_torch as mlx_to_torch_mps,)
_COMPUTE_DTYPE,_ROTARY_METAL_KERNEL,_ROTARY_METAL_UNAVAILABLE = torch.float16, None, False
def mlx_bridge_sdpa(q, k, v): import mlx.core as mx; out = mx.fast.scaled_dot_product_attention(torch_mps_to_mlx(q), torch_mps_to_mlx(k), torch_mps_to_mlx(v), scale=q.shape[-1] ** -0.5); return mlx_to_torch_mps(out, q)
def _rotary_metal_kernel():
    global _ROTARY_METAL_KERNEL, _ROTARY_METAL_UNAVAILABLE
    if _ROTARY_METAL_KERNEL is not None or _ROTARY_METAL_UNAVAILABLE: return _ROTARY_METAL_KERNEL
    metal_kernel = getattr(getattr(__import__("mlx.core", fromlist=["fast"]), "fast", None), "metal_kernel", None)
    if metal_kernel is None: _ROTARY_METAL_UNAVAILABLE = True; return None
    try:
        _ROTARY_METAL_KERNEL = metal_kernel(
            name="pymss_rotary_qk",
            input_names=["q", "k", "cos", "sin"],
            output_names=["q_out", "k_out"],
            source="""
                uint elem = thread_position_in_grid.x;
                uint d = elem % D;
                uint pair = d >> 1;
                uint pair_base = elem - (d & 1);
                uint seq = (elem / (H * D)) % N;
                uint trig_idx = seq * HALF_D + pair;
                T q_even = q[pair_base];
                T q_odd = q[pair_base + 1];
                T k_even = k[pair_base];
                T k_odd = k[pair_base + 1];
                T c = cos[trig_idx];
                T s = sin[trig_idx];
                if ((d & 1) == 0) { q_out[elem] = q_even * c - q_odd * s; k_out[elem] = k_even * c - k_odd * s; } else { q_out[elem] = q_odd * c + q_even * s; k_out[elem] = k_odd * c + k_even * s; }
            """,
            ensure_row_contiguous=True,
        )
    except Exception:
        _ROTARY_METAL_UNAVAILABLE = True
        return None
    return _ROTARY_METAL_KERNEL
def _apply_rotary_fallback(q, k, cos, sin):
    import mlx.core as mx
    def rotate(t): even, odd = t[..., ::2], t[..., 1::2]; return mx.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1).reshape(t.shape)
    return rotate(q), rotate(k)
def _apply_rotary_metal(q, k, cos, sin, dtype):
    kernel = _rotary_metal_kernel()
    if kernel is None or q.ndim != 4 or k.shape != q.shape or q.shape[-1] % 2: return None
    _, n, heads, dim = q.shape
    outputs = kernel(inputs=[q, k, cos, sin], template=[("T", dtype), ("N", int(n)), ("H", int(heads)), ("D", int(dim)), ("HALF_D", int(dim // 2))], grid=(q.size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[q.shape, k.shape], output_dtypes=[q.dtype, k.dtype])
    return outputs[0], outputs[1]
def _rotary_cos_sin(rotary_embed, seq_len, dtype):
    import mlx.core as mx
    cache = getattr(rotary_embed, "_pymss_mlx_cos_sin_cache", None)
    if cache is None: rotary_embed._pymss_mlx_cos_sin_cache = cache = {}
    key = (seq_len, dtype, rotary_embed.freqs.data_ptr(), rotary_embed.freqs._version)
    if key not in cache:
        freqs = _torch_to_mlx_array(rotary_embed.freqs, torch.float32)
        angles = mx.arange(seq_len, dtype=mx.float32)[:, None] * freqs[None]
        cache.clear()
        cache[key] = (mx.cos(angles).astype(dtype)[None, :, None, :], mx.sin(angles).astype(dtype)[None, :, None, :])
    return cache[key]
def _apply_rotary(q, k, rotary_embed, dtype):
    cos, sin = _rotary_cos_sin(rotary_embed, q.shape[1], dtype)
    if not getattr(rotary_embed, "_pymss_mlx_disable_metal_rotary", False):
        try:
            out = _apply_rotary_metal(q, k, cos, sin, dtype)
            if out is not None: return out
        except Exception as exc:
            rotary_embed._pymss_mlx_disable_metal_rotary = True
            rotary_embed._pymss_mlx_metal_rotary_error = repr(exc)
    return _apply_rotary_fallback(q, k, cos, sin)
def _attention_cache(module, dtype):
    cache = getattr(module, "_pymss_mlx_attention_cache", None)
    params = (module.norm.gamma, module.to_qkv.weight, module.to_qkv.bias, module.to_gates.weight, module.to_gates.bias, module.to_out[0].weight, module.to_out[0].bias)
    key = tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)
    if cache is not None and cache.get("key") == key: return cache
    cache = {
        "key": key,
        "norm_gamma": _torch_to_mlx_array(module.norm.gamma, dtype),
        "qkv_weight": _torch_to_mlx_array(module.to_qkv.weight, dtype),
        "qkv_bias": None if module.to_qkv.bias is None else _torch_to_mlx_array(module.to_qkv.bias, dtype),
        "gate_weight": _torch_to_mlx_array(module.to_gates.weight, dtype),
        "gate_bias": _torch_to_mlx_array(module.to_gates.bias, dtype),
        "out_weight": _torch_to_mlx_array(module.to_out[0].weight, dtype),
        "out_bias": None if module.to_out[0].bias is None else _torch_to_mlx_array(module.to_out[0].bias, dtype),
    }
    module._pymss_mlx_attention_cache = cache
    return cache
_SDPA_QUERY_CHUNK = 512
def _sdpa_chunked(q, k, v, scale):  # query-axis streaming: workspace bounded at chunk/T; tail chunks of 1-8 queries take the small-M kernel path (measured <=1 fp16 ulp vs fused)
    import mlx.core as mx
    T = q.shape[2]
    if T <= _SDPA_QUERY_CHUNK: return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return mx.concatenate([mx.fast.scaled_dot_product_attention(q[:, :, s : s + _SDPA_QUERY_CHUNK], k, v, scale=scale) for s in range(0, T, _SDPA_QUERY_CHUNK)], axis=2)
def _mlx_attention(module, x, dtype):
    import mlx.core as mx
    dtype = _mlx_dtype(torch_dtype(dtype))
    cache = _attention_cache(module, dtype)
    # Q/K/V evaluation also materializes normalization and projection. Keep the
    # three stage boundaries without synchronizing those dependencies separately.
    x_norm = _rms_norm(x, cache["norm_gamma"])
    if x_norm.dtype != dtype: x_norm = x_norm.astype(dtype)
    qkv = _linear(x_norm, cache["qkv_weight"], cache["qkv_bias"])
    b, n, _ = qkv.shape
    qkv = qkv.reshape(b, n, 3, module.heads, -1)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if module.rotary_embed is not None: q, k = _apply_rotary(q, k, module.rotary_embed, dtype)
    q, k, v = mx.swapaxes(q, 1, 2), mx.swapaxes(k, 1, 2), mx.swapaxes(v, 1, 2); mx.eval(q, k, v)
    out = _sdpa_chunked(q, k, v, scale=q.shape[-1] ** -0.5); mx.eval(out)
    del q, k, v, qkv
    out = mx.swapaxes(out, 1, 2)
    gates = _sigmoid(_linear(x_norm, cache["gate_weight"], cache["gate_bias"]))
    out = _linear((out * gates[..., None]).reshape(b, n, -1), cache["out_weight"], cache["out_bias"])
    del x_norm, gates
    mx.eval(out)
    return out
def mlx_bridge_attention(module, x): x_mx = torch_mps_to_mlx(x).astype(_mlx_dtype(_COMPUTE_DTYPE)); out = _mlx_attention(module, x_mx, _COMPUTE_DTYPE); return mlx_to_torch_mps(out, x)
def _feed_forward_cache(module, dtype):
    norm, linear_in, activation, _, linear_out, _ = module.net
    if not isinstance(activation, torch.nn.GELU) or activation.approximate != "none": raise TypeError("MLX feed-forward bridge only supports torch.nn.GELU(approximate='none')")
    cache = getattr(module, "_pymss_mlx_feed_forward_cache", None)
    params = (norm.gamma, linear_in.weight, linear_in.bias, linear_out.weight, linear_out.bias)
    key = tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)
    if cache is not None and cache.get("key") == key: return cache
    cache = { "key": key, "norm_gamma": _torch_to_mlx_array(norm.gamma, dtype), "linear_in_weight": _torch_to_mlx_array(linear_in.weight, dtype), "linear_in_bias": None if linear_in.bias is None else _torch_to_mlx_array(linear_in.bias, dtype), "linear_out_weight": _torch_to_mlx_array(linear_out.weight, dtype), "linear_out_bias": None if linear_out.bias is None else _torch_to_mlx_array(linear_out.bias, dtype), }
    module._pymss_mlx_feed_forward_cache = cache
    return cache
def _mlx_feed_forward(module, x, dtype):
    import mlx.core as mx
    cache = _feed_forward_cache(module, dtype)
    if not getattr(module, "_pymss_mlx_disable_compiled_ffn", False):
        compiled_cache = getattr(module, "_pymss_mlx_compiled_feed_forward_cache", None)
        if compiled_cache is None: module._pymss_mlx_compiled_feed_forward_cache = compiled_cache = {}
        key = (tuple(x.shape), x.dtype, cache["key"])
        if compiled_cache.get("key") != key:
            # One graph or failure record; new shapes/weights may compile successfully.
            compiled_cache.clear()
            compiled_cache["key"] = key
        if "error" not in compiled_cache:
            try:
                return _mlx_feed_forward_compiled(x, cache, compiled_cache)
            except Exception as exc:
                # Do not retain a failed graph or permanently disable other cache keys.
                compiled_cache.clear()
                compiled_cache.update(key=key, error=repr(exc))
    # Keep the bounded eager path for unsupported compiler/backend operations.
    y = _rms_norm(x, cache["norm_gamma"])
    if y.dtype != cache["norm_gamma"].dtype: y = y.astype(cache["norm_gamma"].dtype)
    mx.eval(y)
    y = _gelu(_linear(y, cache["linear_in_weight"], cache["linear_in_bias"])); mx.eval(y)
    out = _linear(y, cache["linear_out_weight"], cache["linear_out_bias"]); mx.eval(out)
    return out
def _mlx_feed_forward_compiled(x, cache, compiled_cache):
    import mlx.core as mx
    has_in_bias, has_out_bias = cache["linear_in_bias"] is not None, cache["linear_out_bias"] is not None
    if "fn" not in compiled_cache:
        def ffn_core(x_arg, gamma, weight_in, bias_in, weight_out, bias_out):
            y = _rms_norm(x_arg, gamma)
            if y.dtype != gamma.dtype: y = y.astype(gamma.dtype)
            y = _gelu(_linear(y, weight_in, bias_in if has_in_bias else None))
            return _linear(y, weight_out, bias_out if has_out_bias else None)
        compiled_cache["fn"] = mx.compile(ffn_core)
    dummy = mx.zeros((0,), dtype=x.dtype)
    out = compiled_cache["fn"](x, cache["norm_gamma"], cache["linear_in_weight"], cache["linear_in_bias"] if has_in_bias else dummy,
                               cache["linear_out_weight"], cache["linear_out_bias"] if has_out_bias else dummy)
    if out.shape != x.shape: raise ValueError(f"compiled FFN returned shape {out.shape}, expected {x.shape}")
    # Materialize here so FFN intermediates die before the next attention stage.
    mx.eval(out)
    return out
def _norm_gamma_cache(module, dtype):
    if isinstance(module, torch.nn.Identity): return None
    if not hasattr(module, "gamma"): raise TypeError("MLX transformer bridge only supports RMSNorm or Identity output norm")
    cache = getattr(module, "_pymss_mlx_norm_cache", None)
    key = (module.gamma.data_ptr(), module.gamma._version, tuple(module.gamma.shape), dtype)
    if cache is not None and cache.get("key") == key: return cache["gamma"]
    cache = {"key": key, "gamma": _torch_to_mlx_array(module.gamma, dtype)}
    module._pymss_mlx_norm_cache = cache
    return cache["gamma"]
def _mlx_output_norm(module, x, dtype): gamma = _norm_gamma_cache(module, dtype); return x if gamma is None else _rms_norm(x, gamma)
def mlx_bridge_transformer(module, x):
    x_mx = torch_mps_to_mlx(x).astype(_mlx_dtype(_COMPUTE_DTYPE))
    for attn, ff in module.layers: x_mx = _mlx_attention(attn, x_mx, _COMPUTE_DTYPE) + x_mx; x_mx = _mlx_feed_forward(ff, x_mx, _COMPUTE_DTYPE) + x_mx
    x_mx = _mlx_output_norm(module.norm, x_mx, _COMPUTE_DTYPE)
    return mlx_to_torch_mps(x_mx, x)
