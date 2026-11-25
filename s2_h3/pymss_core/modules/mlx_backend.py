# Shared MLX full-backend kernel. One copy of every torch->mx primitive used by
# the per-family adapters (mdx23c/demucs/bandit/scnet/apollo/roformer).
# mx stays a call-time import: MLX is an optional dependency.
import numpy as np
import torch
def mx_dtype(dtype):
    import mlx.core as mx
    if dtype == torch.float16: return mx.float16
    if dtype == torch.float32: return mx.float32
    raise TypeError(f"unsupported MLX bridge dtype: {dtype}")
def torch_dtype(dtype):  # accept torch or mx dtype, return torch dtype
    import mlx.core as mx
    return {mx.float16: torch.float16, mx.float32: torch.float32}.get(dtype, dtype)
def to_mx(tensor, dtype):  # dtype: torch or mx
    import mlx.core as mx
    return mx.array(tensor.detach().to(dtype=torch_dtype(dtype)).cpu().numpy())
def to_mx_raw(tensor):  # no dtype cast, keeps fp32 (MPS bridge path)
    import mlx.core as mx
    return mx.array(tensor.detach().cpu().numpy())
def to_torch(array, reference): return torch.from_numpy(np.array(array, copy=False)).to(device=reference.device, dtype=reference.dtype)
def check_dtype(dtype, name):
    if dtype not in (torch.float16, torch.float32): raise TypeError(f"MLX full {name} supports torch.float16 or torch.float32 compute dtype")
def param(module, name, tensor, dtype, transpose=None):  # memoize converted weights on the torch module; dtype: torch or mx
    cache = getattr(module, "_pymss_mlx_full_param_cache", None)
    if cache is None: cache = {}; module._pymss_mlx_full_param_cache = cache
    cache_key = name if transpose is None else f"{name}_{transpose}"
    key = (name, tensor.data_ptr(), tensor._version, tuple(tensor.shape), str(dtype), transpose)
    cached = cache.get(cache_key)
    if cached is not None and cached[0] == key: return cached[1]
    value = to_mx(tensor, dtype)
    if transpose is not None: value = value.transpose(transpose)
    cache[cache_key] = (key, value)
    return value
def linear(x, weight, bias=None):
    import mlx.core as mx
    transposed = mx.swapaxes(weight, -1, -2)
    # Match mlx.nn.Linear: fuse the bias into GEMM instead of retaining a full
    # projection temporary for a separate addition.
    return mx.matmul(x, transposed) if bias is None else mx.addmm(bias, x, transposed)
def linear_layer(module, x, dtype): return linear(x, param(module, "weight", module.weight, dtype), None if module.bias is None else param(module, "bias", module.bias, dtype))
def generic_activation(module, x, extra_swish=False):
    # shared activation dispatch: Tanh/ReLU/GELU/ELU/Identity/Swish, ordered by per-family frequency
    import mlx.core as mx
    if isinstance(module, torch.nn.Tanh): return mx.tanh(x)
    if isinstance(module, torch.nn.ReLU): return relu(x)
    if isinstance(module, torch.nn.GELU):
        if module.approximate != "none": raise TypeError("MLX bridge only supports exact GELU")
        return gelu(x)
    if isinstance(module, torch.nn.SiLU): return silu(x)
    if isinstance(module, torch.nn.Identity): return x
    if extra_swish and type(module).__name__ == "Swish": return swish(x)
    if isinstance(module, torch.nn.ELU): return mx.where(x > 0, x, module.alpha * (mx.exp(x) - 1))
    raise TypeError(f"unsupported activation for MLX full backend: {type(module).__name__}")
def generic_module_forward(module, x, dtype, norm_fn, swish_cls=None, extra=()):
    # shared isinstance dispatch for conv/linear/norm/activation trees used by every per-family adapter.
    # norm_fn handles the family-specific norm set; extra maps family classes to handlers (type, fn) pairs.
    for klass, fn in extra:
        if isinstance(module, klass): return fn(module, x, dtype)
    if isinstance(module, torch.nn.Sequential):
        for child in module: x = generic_module_forward(child, x, dtype, norm_fn, swish_cls, extra)
        return x
    if isinstance(module, torch.nn.Conv1d): return conv1d(module, x, dtype)
    if isinstance(module, torch.nn.Conv2d): return conv2d(module, x, dtype)
    if isinstance(module, torch.nn.ConvTranspose1d): return conv_transpose1d(module, x, dtype)
    if isinstance(module, torch.nn.ConvTranspose2d): return conv_transpose2d(module, x, dtype)
    if isinstance(module, torch.nn.Linear): return linear_layer(module, x, dtype)
    if isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d)): return norm_fn(module, x, dtype)
    if isinstance(module, torch.nn.GLU): return glu(x, module.dim)
    if isinstance(module, torch.nn.SiLU): return silu(x)
    if swish_cls is not None and isinstance(module, swish_cls): return swish(x)
    return generic_activation(module, x)
def rms_norm(x, gamma):
    import mlx.core as mx
    if x.dtype == mx.float16:
        # Promote before squaring even when gamma is FP32: multiplying by a
        # wider weight later cannot repair an FP16 underflow/overflow.
        if getattr(gamma, "dtype", None) == mx.float16 and gamma.ndim == 1 and gamma.shape[0] == x.shape[-1]:
            return mx.fast.rms_norm(x, gamma, eps=1e-12)
        promoted = x.astype(mx.float32)
        normalized = promoted * mx.rsqrt(mx.mean(mx.square(promoted), axis=-1, keepdims=True) + 1e-12)
        # Read native promotion from a lazy expression; do not narrow FP32 gamma.
        out, output_dtype = normalized * gamma, (x * gamma).dtype
        return out if out.dtype == output_dtype else out.astype(output_dtype)
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-12) * gamma
def sigmoid(x): import mlx.core as mx; return mx.sigmoid(x)
def gelu(x): import mlx.core as mx; return 0.5 * x * (1 + mx.erf(x * (2**-0.5)))
def relu(x): import mlx.core as mx; return mx.maximum(x, 0)
def glu(x, axis=-1): import mlx.core as mx; a, b = mx.split(x, 2, axis=axis); return a * mx.sigmoid(b)
def swish(x): import mlx.core as mx; return x * mx.sigmoid(x)
def silu(x): return swish(x)
def elu(module, x): import mlx.core as mx; return mx.where(x > 0, x, module.alpha * (mx.exp(x) - 1))
_HANN_MX_CACHE = {}
def periodic_hann_window(length, dtype):
    import mlx.core as mx
    length = int(length)
    key = (length, dtype)
    w = _HANN_MX_CACHE.get(key)
    if w is not None: return w
    if length <= 0: w = mx.zeros((0,), dtype=dtype)
    elif length == 1: w = mx.ones((1,), dtype=dtype)
    else: w = mx.hanning(length + 1)[:-1].astype(dtype)
    _HANN_MX_CACHE[key] = w
    return w
_CACHE_LIMIT_DONE = False
def _cache_limit():  # allocator cache is a monotonic ledger of dead-but-cached buffers; cap it so the peak tracks the live set, not the sum of all stages (PYMSS_MLX_CACHE_LIMIT_MB, -1=off)
    global _CACHE_LIMIT_DONE
    if _CACHE_LIMIT_DONE: return
    _CACHE_LIMIT_DONE = True
    try:
        import os
        import mlx.core as mx
        if (mb := int(os.environ.get("PYMSS_MLX_CACHE_LIMIT_MB", "1024"))) >= 0: mx.metal.set_cache_limit(mb * 1024 * 1024)
    except Exception: pass
def maybe_eval(*arrays):  # rolling eval: free each stage before the next builds; no-op when mlx is unavailable (cross-platform unit tests mock the forward path)
    try:
        import mlx.core as mx
    except Exception:
        return
    _cache_limit()
    mx.eval(*arrays)
def compile_cached(module, cache_name, key, fn):
    import mlx.core as mx
    cache = getattr(module, cache_name, None)
    if cache is None: cache = {}; setattr(module, cache_name, cache)
    compiled = cache.get(key)
    if compiled is None: compiled = mx.compile(fn); cache[key] = compiled
    return compiled
def reflect_pad_last(x, left=0, right=0):
    import mlx.core as mx
    if left <= 0 and right <= 0: return x
    if x.shape[-1] <= max(left, right): raise ValueError("reflect padding requires input length greater than padding")
    parts = []
    if left > 0: parts.append(x[..., 1:left + 1][..., ::-1])
    parts.append(x)
    if right > 0: parts.append(x[..., -right - 1:-1][..., ::-1])
    return mx.concatenate(parts, axis=-1)
def pad_last(x, left, right, mode="constant", value=0.0, extend=False):
    import mlx.core as mx
    if left <= 0 and right <= 0: return x
    if mode == "constant": return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], constant_values=value)
    if mode != "reflect": raise TypeError(f"unsupported padding mode for MLX full backend: {mode!r}")
    if extend:  # demucs: pad flat with zeros until reflect windows fit
        length = x.shape[-1]
        max_pad = max(left, right)
        if length <= max_pad:
            extra = max_pad - length + 1
            extra_right = min(right, extra)
            extra_left = extra - extra_right
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(extra_left, extra_right)])
            left, right = left - extra_left, right - extra_right
    return reflect_pad_last(x, left, right)
def conv_padding(conv, ndim=2):
    padding = conv.padding
    if isinstance(padding, str): kernel = conv.kernel_size; return (kernel[0] // 2, kernel[1] // 2) if ndim == 2 else kernel[0] // 2
    if isinstance(padding, int): return (padding,) * ndim
    return padding[0] if ndim == 1 and len(padding) == 1 else padding
def conv1d(conv, x, dtype):  # NCL in/out
    import mlx.core as mx
    y = mx.conv1d(x.transpose(0, 2, 1), param(conv, "weight_mlx1d", conv.weight, dtype, (0, 2, 1)), stride=conv.stride[0], padding=conv_padding(conv, 1), dilation=conv.dilation[0], groups=conv.groups)
    if conv.bias is not None: y = y + param(conv, 'bias', conv.bias, dtype)
    return y.transpose(0, 2, 1)
def conv_transpose1d(conv, x, dtype):  # NCL in/out
    import mlx.core as mx
    y = mx.conv_transpose1d(x.transpose(0, 2, 1), param(conv, "weight_mlxt1d", conv.weight, dtype, (1, 2, 0)), stride=conv.stride[0], padding=conv.padding[0], dilation=conv.dilation[0], output_padding=conv.output_padding[0], groups=conv.groups)
    if conv.bias is not None: y = y + param(conv, 'bias', conv.bias, dtype)
    return y.transpose(0, 2, 1)
def conv2d(conv, x, dtype, padding=None):  # NCHW in/out
    import mlx.core as mx
    y = mx.conv2d(x.transpose(0, 2, 3, 1), param(conv, "weight_mlx2d", conv.weight, dtype, (0, 2, 3, 1)), stride=conv.stride, padding=conv_padding(conv) if padding is None else padding, dilation=conv.dilation, groups=conv.groups)
    if conv.bias is not None: y = y + param(conv, 'bias', conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)
def conv_transpose2d(conv, x, dtype):  # NCHW in/out
    import mlx.core as mx
    y = mx.conv_transpose2d(x.transpose(0, 2, 3, 1), param(conv, "weight_mlxt2d", conv.weight, dtype, (1, 2, 3, 0)), stride=conv.stride, padding=conv.padding, dilation=conv.dilation, output_padding=conv.output_padding, groups=conv.groups)
    if conv.bias is not None: y = y + param(conv, 'bias', conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)
def group_norm(module, x, dtype):  # NCHW / NC(*)
    import mlx.core as mx
    b, c = x.shape[:2]
    rest = x.shape[2:]
    y = x.astype(mx.float32).reshape(b, int(module.num_groups), c // module.num_groups, *rest)
    axes = tuple(range(2, y.ndim))
    mean = mx.mean(y, axis=axes, keepdims=True)
    var = mx.mean(mx.square(y - mean), axis=axes, keepdims=True)
    y = ((y - mean) * mx.rsqrt(var + module.eps)).reshape(x.shape).astype(x.dtype)
    if module.affine: shape = (1, -1) + (1,) * len(rest); y = y * param(module, 'weight', module.weight, dtype).reshape(*shape); y = y + param(module, 'bias', module.bias, dtype).reshape(*shape)
    return y
def layer_norm(module, x, dtype):
    import mlx.core as mx
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    y = ((x32 - mean) * mx.rsqrt(var + module.eps)).astype(x.dtype)
    if module.elementwise_affine:
        y = y * param(module, "weight", module.weight, dtype)
        if module.bias is not None: y = y + param(module, 'bias', module.bias, dtype)
    return y
def instance_norm2d(module, x, dtype):
    import mlx.core as mx
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=(2, 3), keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=(2, 3), keepdims=True)
    y = ((x32 - mean) * mx.rsqrt(var + module.eps)).astype(x.dtype)
    if module.affine: y = y * param(module, 'weight', module.weight, dtype).reshape(1, -1, 1, 1); y = y + param(module, 'bias', module.bias, dtype).reshape(1, -1, 1, 1)
    return y
def batch_norm(module, x, dtype):  # eval mode only (inference package); 1d/2d/any-ndim
    import mlx.core as mx
    if module.training: raise TypeError("MLX BatchNorm supports eval mode only")
    shape = (1, -1) + (1,) * (x.ndim - 2)
    y = x.astype(mx.float32)
    mean = to_mx(module.running_mean, torch.float32).reshape(shape)
    var = to_mx(module.running_var, torch.float32).reshape(shape)
    y = (y - mean) * mx.rsqrt(var + module.eps)
    if module.affine: y = y.astype(x.dtype) * param(module, 'weight', module.weight, dtype).reshape(shape); y = y + param(module, 'bias', module.bias, dtype).reshape(shape)
    return y.astype(x.dtype)
def _rnn_params(rnn, suffix, dtype): return { key: param(rnn, f"{key}_l0{suffix}", getattr(rnn, f"{key}_l0{suffix}"), dtype) for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh") if rnn.bias or not key.startswith("bias") }
def lstm(rnn, x, dtype):
    import mlx.core as mx
    def run(p, reverse=False):
        h = mx.zeros((x.shape[0], rnn.hidden_size), dtype=x.dtype); c, outs = mx.zeros_like(h), []
        wx = linear(x, p["weight_ih"], p.get("bias_ih"))
        for t in range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1]):
            gates = wx[:, t] + linear(h, p["weight_hh"], p.get("bias_hh"))
            i, f, g, o = mx.split(gates, 4, axis=-1)
            i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
            c = f * c + i * mx.tanh(g); h = o * mx.tanh(c); outs.append(h)
        if reverse: outs.reverse()
        return mx.stack(outs, axis=1)
    if rnn.num_layers != 1 or not rnn.batch_first: raise TypeError("MLX RNN supports one-layer batch_first RNNs only")
    forward = run(_rnn_params(rnn, "", dtype))
    if not rnn.bidirectional: return forward
    return mx.concatenate((forward, run(_rnn_params(rnn, "_reverse", dtype), reverse=True)), axis=-1)
def gru(rnn, x, dtype):
    import mlx.core as mx
    def run(p, reverse=False):
        h = mx.zeros((x.shape[0], rnn.hidden_size), dtype=x.dtype)
        outs = []
        wx = linear(x, p["weight_ih"], p.get("bias_ih"))
        for t in range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1]):
            gi = wx[:, t]; gh = linear(h, p["weight_hh"], p.get("bias_hh"))
            i_r, i_z, i_n = mx.split(gi, 3, axis=-1); h_r, h_z, h_n = mx.split(gh, 3, axis=-1)
            reset, update = sigmoid(i_r + h_r), sigmoid(i_z + h_z)
            h = (1 - update) * mx.tanh(i_n + reset * h_n) + update * h; outs.append(h)
        if reverse: outs.reverse()
        return mx.stack(outs, axis=1)
    if rnn.num_layers != 1 or not rnn.batch_first: raise TypeError("MLX RNN supports one-layer batch_first RNNs only")
    forward = run(_rnn_params(rnn, "", dtype))
    if not rnn.bidirectional: return forward
    return mx.concatenate((forward, run(_rnn_params(rnn, "_reverse", dtype), reverse=True)), axis=-1)
def rnn_forward(rnn, x, dtype):
    if isinstance(rnn, torch.nn.LSTM): return lstm(rnn, x, dtype)
    if isinstance(rnn, torch.nn.GRU): return gru(rnn, x, dtype)
    raise TypeError(f"unsupported RNN for MLX full backend: {type(rnn).__name__}")
def stft(x, n_fft, hop, window, dtype, center=True, pad_mode="reflect", normalized=False, pad_fn=None):
    # x: (..., L) -> (..., F, T) complex spec
    import mlx.core as mx
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).astype(dtype)
    if center: flat = pad_fn(flat, n_fft // 2, n_fft // 2) if pad_fn else pad_last(flat, n_fft // 2, n_fft // 2, pad_mode)
    frames = 1 + (flat.shape[-1] - n_fft) // hop
    framed = mx.as_strided(flat, shape=(flat.shape[0], frames, n_fft), strides=(flat.shape[-1], hop, 1))
    spec = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    if normalized: spec = spec / np.sqrt(n_fft)
    spec = mx.moveaxis(spec, -1, -2)  # (n, T, F) -> (n, F, T)
    return spec.reshape(*leading, spec.shape[-2], spec.shape[-1])
_OVERLAP_ADD_CACHE = {}
def overlap_add(frames, window, hop):  # weighted overlap-add, 1e-11 denom floor (matches torch istft)
    import mlx.core as mx
    n_fft = window.shape[-1]
    count = frames.shape[1]
    key = (id(window), hop, count)
    cached = _OVERLAP_ADD_CACHE.get(key)
    if cached is None or cached[0] is not window:
        if len(_OVERLAP_ADD_CACHE) > 64: _OVERLAP_ADD_CACHE.clear()
        positions = mx.arange(n_fft)[None, :] + hop * mx.arange(count)[:, None]
        denom = mx.zeros((n_fft + hop * (count - 1),), dtype=frames.dtype).at[positions].add(mx.broadcast_to(mx.square(window).astype(frames.dtype)[None, :], (count, n_fft)))
        _OVERLAP_ADD_CACHE[key] = cached = (window, positions, denom)
    _, positions, denom = cached
    full_length = denom.shape[-1]
    audio = mx.zeros((frames.shape[0], full_length), dtype=frames.dtype).at[:, positions].add(frames)
    return audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=frames.dtype))
def istft(spec, window, hop, length, dtype, n_fft=None, center=True, normalized=False):
    # spec: (..., F, T) complex -> (..., L); n_fft inferred for even sizes only
    import mlx.core as mx
    leading = spec.shape[:-2]
    freqs, count = spec.shape[-2:]
    if n_fft is None: n_fft = 2 * freqs - 2
    flat = mx.moveaxis(spec.reshape(-1, freqs, count), -2, -1)
    if normalized: flat = flat * np.sqrt(n_fft)
    frames = mx.fft.irfft(flat, n=n_fft, axis=-1).astype(dtype) * window
    audio = overlap_add(frames, window, hop)
    if center:
        audio = audio[..., n_fft // 2 : n_fft // 2 + length]
    elif length is not None: audio = audio[..., :length]
    return audio.reshape(*leading, audio.shape[-1])
class MpsBackendMixin:
    # One copy of the mps/mlx backend switch shared by every model family.
    mps_model_backend = "torch"
    mps_model_compute_dtype = torch.float16
    def set_mps_model_backend(self, backend=None, compute_dtype=None):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx_full"): raise ValueError("mps_model_backend must be 'torch' or 'mlx_full'")
        self.mps_model_backend = backend
        if compute_dtype is None: return
        if isinstance(compute_dtype, str): compute_dtype = {'float16': torch.float16, 'fp16': torch.float16, 'float32': torch.float32, 'fp32': torch.float32}.get(compute_dtype.lower(), compute_dtype)
        if compute_dtype not in (torch.float16, torch.float32): raise ValueError("mps_model_compute_dtype must be 'float16' or 'float32'")
        self.mps_model_compute_dtype = compute_dtype
    def _use_mlx_full_forward(self, x): return not self.training and self.mps_model_backend == "mlx_full" and x.device.type == "mps"
