import torch
from pymss_core.runtime_cache import _clear_cache_dicts
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList
from .attend import Attend
_CUDA_ATTENTION_BACKEND_ALIASES = {"auto": "auto", "torch": "default", "default": "default", "sdpa": "default", "flash": "flash", "flash_attention": "flash", "cudnn": "cudnn", "cudnn_attn": "cudnn", "cudnn_attention": "cudnn", "efficient": "efficient", "mem_efficient": "efficient", "memory_efficient": "efficient", "math": "math", "xformers": "xformers"}
_SDPA_BACKEND_ENUM_NAMES = {"flash": "FLASH_ATTENTION", "cudnn": "CUDNN_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}
_MPS_BACKENDS = ("torch", "mlx", "mlx_attention", "mlx_transformer")
def resolve_cuda_attention_backend(backend):
    # single source of truth for cuda_attention_backend parsing:
    # None/empty -> platform default (default_cuda_attention_backend), alias table -> canonical name, unknown -> ValueError
    backend = str(backend or default_cuda_attention_backend()).lower().replace("-", "_")
    if backend not in _CUDA_ATTENTION_BACKEND_ALIASES: raise ValueError("cuda_attention_backend must be one of: auto, default, flash, cudnn, efficient, math, xformers")
    return _CUDA_ATTENTION_BACKEND_ALIASES[backend]
normalize_cuda_attention_backend = resolve_cuda_attention_backend
def _sdpa_backend_enum(backend): enum_cls = getattr(getattr(torch.nn, "attention", None), "SDPBackend", None); enum_name = _SDPA_BACKEND_ENUM_NAMES.get(backend); return None if enum_cls is None or enum_name is None else getattr(enum_cls, enum_name, None)
def default_cuda_attention_backend():
    # On ROCm (HIP) builds the cuDNN SDPA backend is typically not compiled in,
    # while the AOTriton memory-efficient kernel works across precisions and GPU
    # architectures (and matches the flash kernel there anyway), so it is the
    # direct default under HIP. Runtime failures still fall back to default SDPA.
    if getattr(torch.version, "hip", None): return "efficient"
    return "cudnn" if _sdpa_backend_enum("cudnn") is not None else "default"
def default_mps_attention_backend(): return "torch"
def set_mps_attention_backend(module, backend=None, min_tokens=128, children=(), keep_mlx_transformer=False):
    # one switch shared by Attention / Transformer / Conformer; mlx_transformer maps to torch at leaf level
    backend = (backend or default_mps_attention_backend()).lower()
    if backend not in _MPS_BACKENDS: raise ValueError("mps_attention_backend must be 'torch', 'mlx', 'mlx_attention', or 'mlx_transformer'")
    if getattr(module, "use_pope", False): backend = "torch"
    module.mps_attention_backend = backend if keep_mlx_transformer else ("torch" if backend == "mlx_transformer" else backend)
    module.mps_mlx_min_tokens = 128 if min_tokens is None else int(min_tokens)
    for child in children: child.set_mps_attention_backend("torch" if backend == "mlx_transformer" else backend, module.mps_mlx_min_tokens)
def set_cuda_attention_backend(module, backend=None, children=()):
    module.cuda_attention_backend = normalize_cuda_attention_backend(backend)
    if hasattr(module, "_disabled_cuda_attention_backends"): module._disabled_cuda_attention_backends.clear()
    for child in children: child.set_cuda_attention_backend(module.cuda_attention_backend)
def _sdpa_with_backend(q, k, v, dropout_p, backend, scale=None):
    kernel = getattr(getattr(torch.nn, "attention", None), "sdpa_kernel", None)
    enum = _sdpa_backend_enum(backend)
    if kernel is None or enum is None: raise RuntimeError(f"SDPA backend {backend!r} is not available in this PyTorch build")
    with kernel(enum):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
def _xformers_attention(q, k, v, dropout_p): import xformers.ops as xops; return xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), p=dropout_p).transpose(1, 2)
#: Devices where the complex-multiply path measured fastest (4.7x on CUDA/MPS).
_COMPLEX_ROTARY_DEVICES = frozenset({"cuda", "mps"})


def apply_rotary_emb_fast(cos, sin, t):
    """Rotate ``t`` and return ``[even | odd]`` halves instead of interleaved pairs.

    Only q and k are ever rotated, and both get the same permutation of their head
    dimension, so every dot product -- and therefore every attention score -- is
    unchanged.  What does change is the write pattern: interleaved output needs
    strided stores, which cost 35.7 ms for the (62, 862, 8, 64) tensors a production
    roformer rotates, while writing two contiguous halves costs 9.4 ms.  CUDA and MPS
    measured the complex route fastest (4.7x), so those keep it.

    Args:
        cos: Cosine table, broadcastable to ``t[..., ::2]``.
        sin: Sine table of the same shape as ``cos``.
        t: Tensor whose last dimension is the head dimension.

    Returns:
        The rotated tensor with the head dimension permuted to ``[even | odd]``.
    """
    if t.device.type in _COMPLEX_ROTARY_DEVICES and (
        t.dtype == torch.float32 or (t.is_cuda and t.dtype == torch.float16)
    ):
        rot = torch.complex(cos, sin)
        rotated = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2)) * rot
        return torch.view_as_real(rotated).reshape_as(t)
    even = t[..., ::2] * cos - t[..., 1::2] * sin
    odd = t[..., 1::2] * cos + t[..., ::2] * sin
    return torch.cat((even, odd), dim=-1)


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, "_pymss_cos_sin_cache", None)
    if cache is None: rotary_embed._pymss_cos_sin_cache = cache = {}
    key = (seq_len, device.type, device.index, dtype)
    if (cached := cache.get(key)) is not None: return cached
    phase_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    freqs = rotary_embed.forward(lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=phase_dtype, offset=0), cache_key=f"freqs:{seq_len}|offset:0")[None, :, None, :].to(device=device, dtype=phase_dtype)
    cache[key] = cached = (freqs[..., ::2].cos().to(dtype=dtype), freqs[..., ::2].sin().to(dtype=dtype))
    return cached
def rotate_qk_fast_bnhd(rotary_embed, q, k): cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype); return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)
def qkv_to_bnhd(qkv, heads): b, n, _ = qkv.shape; return qkv.view(b, n, 3, heads, -1).unbind(dim=2)
class RMSNorm(Module):
    def __init__(self, dim): super().__init__(); self.scale,self.gamma,self._gamma_dtype_cache = dim**0.5, nn.Parameter(torch.ones(dim)), {}
    def clear_runtime_cache(self): _clear_cache_dicts(self, "_gamma_dtype_cache")
    def forward(self, x):
        if not self.training and x.dtype in (torch.float16, torch.bfloat16):
            if torch.compiler.is_compiling():
                # The dict cache below is keyed by ``data_ptr()``, which dynamo
                # cannot trace, so a compiled graph takes the plain cast instead.
                # XLA constant-folds it per layer, so the cache is not needed.
                return F.rms_norm(x, (x.shape[-1],), self.gamma.to(device=x.device, dtype=x.dtype), eps=1e-12)
            key = (x.device.type, x.device.index, x.dtype, self.gamma.data_ptr(), self.gamma._version)
            gamma = self._gamma_dtype_cache.get(key)
            if gamma is None: gamma = self.gamma.detach().to(device=x.device, dtype=x.dtype); self._gamma_dtype_cache.clear(); self._gamma_dtype_cache[key] = gamma
            return F.rms_norm(x, (x.shape[-1],), gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma
class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0): super().__init__(); self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, int(dim * mult)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(dim * mult), dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)
class MoEFFN(Module):
    """Sparse MoE feed-forward: content-routed top-k experts + one shared expert.

    Router sees the token itself (no external conditioning).  The switch-style
    load-balancing auxiliary loss is stashed on ``self.aux_loss`` during
    training and collected by ``BSRoformer.pop_moe_aux_loss``.
    """

    def __init__(self, *, dim, mult=4, dropout=0.0, num_experts=8, top_k=2, shared_expert=True):
        super().__init__()
        self.experts = ModuleList([FeedForward(dim, mult=mult, dropout=dropout) for _ in range(num_experts)])
        self.shared = FeedForward(dim, mult=mult, dropout=dropout) if shared_expert else None
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.num_experts, self.top_k = num_experts, top_k
        self.aux_loss = None

    def forward(self, x):
        shape = x.shape
        probs = self.router(x).softmax(dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        out = self.shared(x) if self.shared is not None else torch.zeros_like(x)
        flat_x = x.reshape(-1, shape[-1])
        flat_w, flat_i = w.reshape(-1, self.top_k), idx.reshape(-1, self.top_k)
        flat_o = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            m = (flat_i == e).any(dim=-1)
            if m.any():
                sel = flat_i[m] == e
                wsum = (flat_w[m] * sel).sum(dim=-1, keepdim=True)
                flat_o[m] += self.experts[e](flat_x[m]) * wsum
        out = out + flat_o.reshape(shape)
        if self.training:
            imp = probs.mean(dim=tuple(range(probs.ndim - 1)))
            usage = torch.zeros_like(imp)
            for e in range(self.num_experts):
                usage[e] = (flat_i == e).float().sum()
            usage = usage / usage.sum().clamp(min=1.0)
            self.aux_loss = self.num_experts * (imp * usage).sum()
        return out


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shared_qkv_bias=None, shared_out_bias=None, rotary_embed=None, flash=True, pope_embed=None):
        super().__init__()
        if rotary_embed is not None and pope_embed is not None: raise ValueError("Rotary and PoPE embeddings are mutually exclusive")
        self.pope_embed, self.use_pope = pope_embed, pope_embed is not None
        self.heads, self.flash, self.dropout, self.rotary_embed = heads, flash, dropout, rotary_embed
        self.mps_attention_backend, self.mps_mlx_min_tokens = default_mps_attention_backend(), 128
        self.cuda_attention_backend, self._disabled_cuda_attention_backends = default_cuda_attention_backend(), set()
        self.attend, self.norm = Attend(flash=False, dropout=dropout, scale=dim_head ** -0.5 if self.use_pope else None), RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None: self.to_qkv.bias = shared_qkv_bias
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim, bias=(shared_out_bias is not None)), nn.Dropout(dropout))
        if shared_out_bias is not None: self.to_out[0].bias = shared_out_bias
    def set_mps_attention_backend(self, backend=None, min_tokens=128): set_mps_attention_backend(self, backend, min_tokens)
    def set_cuda_attention_backend(self, backend=None): set_cuda_attention_backend(self, backend)
    def _use_mlx_attention_layer(self, x): return (self.flash and not self.training and self.mps_attention_backend == "mlx_attention" and x.device.type == "mps" and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps")) and x.shape[-2] >= self.mps_mlx_min_tokens)
    def _use_mlx_sdpa(self, q): return (self.flash and not self.training and self.mps_attention_backend == "mlx" and q.device.type == "mps" and q.dtype == torch.float16 and q.shape[-2] >= self.mps_mlx_min_tokens)
    def _attention(self, q, k, v):
        if self._use_mlx_sdpa(q):
            try:
                from .mlx_attention import mlx_bridge_sdpa
                return mlx_bridge_sdpa(q, k, v)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = default_mps_attention_backend()
        if self.flash: return self._cuda_or_default_attention(q, k, v)
        return self.attend(q, k, v)
    def _cuda_or_default_attention(self, q, k, v):
        dropout_p = self.dropout if self.training else 0.0
        scale = self.attend.scale
        backend = self.cuda_attention_backend
        if not q.is_cuda or backend == "default": return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
        if backend == "auto":
            for candidate in ("cudnn", "efficient"):
                if candidate in self._disabled_cuda_attention_backends: continue
                try:
                    return _sdpa_with_backend(q, k, v, dropout_p, candidate, scale=scale)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    self._pymss_cuda_attention_backend_error = f"{candidate}: {exc!r}"
                    self._disabled_cuda_attention_backends.add(candidate)
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
        try:
            # PoPE doubles Q/K width; compensate xformers' default scale to keep dim_head**-0.5.
            if backend == "xformers": return _xformers_attention(q * (2 ** 0.5) if self.use_pope else q, k, v, dropout_p)
            return _sdpa_with_backend(q, k, v, dropout_p, backend, scale=scale)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            self._pymss_cuda_attention_backend_error = f"{backend}: {exc!r}"
            self.cuda_attention_backend = "default"
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
    def forward(self, x):
        if self._use_mlx_attention_layer(x):
            try:
                from .mlx_attention import mlx_bridge_attention
                return mlx_bridge_attention(self, x)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = default_mps_attention_backend()
        x = self.norm(x)
        q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)
        if self.rotary_embed is not None: q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.pope_embed is not None: q, k = self.pope_embed(q, k)
        out = self._attention(q, k, v)
        # Some MPS SDPA versions return the expanded Q/K width instead of V's width.
        if self.use_pope and out.shape[-1] != v.shape[-1]: out = out[..., :v.shape[-1]]
        # NOTE(ascend-kernels): fused_gated_out from pymss_core.npu_kernels is
        # numerically equivalent but triton-ascend 3.2.1 memory kernels run ~3x
        # slower than aclnn elementwise on 950PR (unvectorized loads), so the
        # eager sigmoid+mul path stays until triton-ascend closes that gap.
        return self.to_out((out.transpose(1, 2) * self.to_gates(x).sigmoid().unsqueeze(-1)).flatten(start_dim=-2))
class ConvFFN(Module):
    """Gated FFN with a depthwise *temporal* conv mid-layer.

    Replaces the time-axis transformer stack when ``conv_ffn_kernel`` is set:
    local time context at O(T * d * k) instead of O(T^2 * d) attention.
    Operates on the (b, t, f, d) grid used by the band-split roformer core.
    """

    def __init__(self, *, dim, mult=4, k=7, dropout=0.0):
        super().__init__()
        h = dim * mult // 2
        self.proj_in = nn.Linear(dim, h * 2)
        self.conv = nn.Conv1d(h, h, k, padding=k // 2, groups=h)
        self.proj_out = nn.Linear(h, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, f, d = x.shape
        u, v = self.proj_in(x).chunk(2, dim=-1)
        h = u * torch.sigmoid(v)
        hid = h.shape[-1]
        h = h.permute(0, 2, 1, 3).reshape(b * f, t, hid).transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2).reshape(b, f, t, hid).permute(0, 2, 1, 3)
        return self.drop(self.proj_out(h))


class Transformer(Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, norm_output=True, rotary_embed=None, flash_attn=True, shared_qkv_bias=None, shared_out_bias=None, pope_embed=None, ff_moe=None):
        super().__init__()
        self.use_pope = pope_embed is not None
        self.layers = ModuleList([ModuleList([ Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias, rotary_embed=rotary_embed, pope_embed=pope_embed, flash=flash_attn), (MoEFFN(dim=dim, mult=ff_mult, dropout=ff_dropout, **ff_moe) if ff_moe else FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))]) for _ in range(depth)])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.mps_attention_backend, self.mps_mlx_min_tokens, self.cuda_attention_backend = default_mps_attention_backend(), 128, default_cuda_attention_backend()
    def set_mps_attention_backend(self, backend=None, min_tokens=128): set_mps_attention_backend(self, backend, min_tokens, [attn for attn, _ in self.layers], keep_mlx_transformer=True)
    def set_cuda_attention_backend(self, backend=None): set_cuda_attention_backend(self, backend, [attn for attn, _ in self.layers])
    def _use_mlx_transformer(self, x): return (self.mps_attention_backend == "mlx_transformer" and not self.training and x.device.type == "mps" and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps")) and x.shape[-2] >= self.mps_mlx_min_tokens)
    def forward(self, x):
        if self._use_mlx_transformer(x):
            try:
                from .mlx_attention import mlx_bridge_transformer
                return mlx_bridge_transformer(self, x)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.set_mps_attention_backend(default_mps_attention_backend(), self.mps_mlx_min_tokens)
        for attn, ff in self.layers: x = attn(x) + x; x = ff(x) + x
        return self.norm(x)
