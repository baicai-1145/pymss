import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend
from .triton_kernels import (
    attention_gate_out_inplace_triton,
    attention_gate_out_rope_inplace_triton,
    attention_gate_out_triton,
    attention_gate_packed64_5090_triton,
    attention_gate_triton,
    attention_gate_varlen_triton,
    copy_rms_norm_4d_triton,
    rms_norm_triton,
    rotate_qk_inplace_triton,
)


_CUDA_ATTENTION_BACKEND_ALIASES = {
    "auto": "auto",
    "torch": "default",
    "default": "default",
    "sdpa": "default",
    "flash": "flash",
    "flash_attention": "flash",
    "cudnn": "cudnn",
    "cudnn_attn": "cudnn",
    "cudnn_attention": "cudnn",
    "efficient": "efficient",
    "mem_efficient": "efficient",
    "memory_efficient": "efficient",
    "math": "math",
    "xformers": "xformers",
}

_CUDA_TRITON_BACKEND_ALIASES = {
    "auto": "auto",
    "off": "default",
    "torch": "default",
    "default": "default",
    "triton": "auto",
    "attention_gate": "attention_gate",
    "attn_gate": "attention_gate",
    "attention_gate_out": "attention_gate_out",
    "attn_gate_out": "attention_gate_out",
    "atomic_out": "attention_gate_out",
    "freq_atomic_out": "freq_atomic_out",
    "short_atomic_out": "freq_atomic_out",
}

_SDPA_BACKEND_ENUM_NAMES = {
    "flash": "FLASH_ATTENTION",
    "cudnn": "CUDNN_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
}


def normalize_cuda_attention_backend(backend):
    backend = str(backend or "cudnn").lower().replace("-", "_")
    if backend not in _CUDA_ATTENTION_BACKEND_ALIASES:
        raise ValueError(
            "cuda_attention_backend must be one of: auto, default, flash, cudnn, "
            "efficient, math, xformers"
        )
    return _CUDA_ATTENTION_BACKEND_ALIASES[backend]


def _sdpa_backend_enum(backend):
    attention = getattr(torch.nn, "attention", None)
    enum_cls = getattr(attention, "SDPBackend", None)
    enum_name = _SDPA_BACKEND_ENUM_NAMES.get(backend)
    return None if enum_cls is None or enum_name is None else getattr(enum_cls, enum_name, None)


def default_cuda_attention_backend():
    return "cudnn" if _sdpa_backend_enum("cudnn") is not None else "default"


def normalize_cuda_triton_backend(backend):
    backend = str(default_cuda_triton_backend() if backend is None else backend).lower().replace("-", "_")
    if backend not in _CUDA_TRITON_BACKEND_ALIASES:
        raise ValueError("cuda_triton_backend must be one of: auto, default, off, torch, triton, attention_gate, attention_gate_out, freq_atomic_out")
    return _CUDA_TRITON_BACKEND_ALIASES[backend]


def default_cuda_triton_backend():
    return "auto"


def _sdpa_with_backend(q, k, v, dropout_p, backend):
    kernel = getattr(getattr(torch.nn, "attention", None), "sdpa_kernel", None)
    enum = _sdpa_backend_enum(backend)
    if kernel is None or enum is None:
        raise RuntimeError(f"SDPA backend {backend!r} is not available in this PyTorch build")
    with kernel(enum):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def _xformers_attention(q, k, v, dropout_p):
    import xformers.ops as xops

    return xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), p=dropout_p).transpose(1, 2)


def apply_rotary_emb_fast(cos, sin, t):
    if t.is_cuda and t.dtype == torch.float16:
        rot = torch.complex(cos[..., ::2], sin[..., ::2])
        rotated = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2)) * rot
        return torch.view_as_real(rotated).reshape_as(t)

    cos, sin, t_even, t_odd = cos[..., ::2], sin[..., ::2], t[..., ::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., ::2] = t_even * cos - t_odd * sin
    out[..., 1::2] = t_odd * cos + t_even * sin
    return out


def apply_rotary_emb_inplace_with_rot(rot, t):
    torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2)).mul_(rot)
    return t


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, '_pymss_cos_sin_cache', None)
    if cache is None:
        rotary_embed._pymss_cos_sin_cache = cache = {}

    key = (seq_len, device.type, device.index, dtype)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0),
        cache_key=f'freqs:{seq_len}|offset:0'
    )[None, :, None, :].to(device=device, dtype=dtype)
    cos, sin = freqs.cos(), freqs.sin()
    cached = (cos, sin)
    cache[key] = cached
    return cached


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def rotate_qk_bnhd(rotary_embed, q, k, backend="default"):
    backend = normalize_cuda_triton_backend(backend)
    if (
        backend == "auto"
        and not torch.is_grad_enabled()
        and q.is_cuda
        and q.dtype == torch.float16
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q.shape[-1] % 2 == 0
    ):
        cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
        rotated = rotate_qk_inplace_triton(q, k, cos, sin)
        if rotated is not None:
            return rotated
        rot = torch.complex(cos[..., ::2], sin[..., ::2])
        return apply_rotary_emb_inplace_with_rot(rot, q), apply_rotary_emb_inplace_with_rot(rot, k)
    return rotate_qk_fast_bnhd(rotary_embed, q, k)


def qkv_to_bnhd(qkv, heads):
    b, n, _ = qkv.shape
    return qkv.view(b, n, 3, heads, -1).unbind(dim=2)


def pool_kv_sequence(x, stride, mode):
    if stride <= 1 or x.shape[2] < stride * 2:
        return x
    if mode == "avg":
        pad = (-x.shape[2]) % stride
        if pad:
            x = torch.cat((x, x[:, :, -1:, :].expand(-1, -1, pad, -1)), dim=2)
        return x.reshape(x.shape[0], x.shape[1], -1, stride, x.shape[-1]).mean(dim=3).contiguous()
    return x[:, :, ::stride, :].contiguous()


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self._gamma_dtype_cache = {}

    def forward(self, x):
        if not self.training and x.dtype in (torch.float16, torch.bfloat16):
            key = (x.device.type, x.device.index, x.dtype, self.gamma.data_ptr(), self.gamma._version)
            gamma = self._gamma_dtype_cache.get(key)
            if gamma is None:
                gamma = self.gamma.detach().to(device=x.device, dtype=x.dtype)
                self._gamma_dtype_cache.clear()
                self._gamma_dtype_cache[key] = gamma
            if x.is_cuda and x.dtype == torch.float16 and x.is_contiguous() and x.shape[-1] in (256, 384) and x.numel() // x.shape[-1] >= 65536:
                out = rms_norm_triton(x, gamma, eps=1e-12)
                if out is not None:
                    return out
            return F.rms_norm(x, (x.shape[-1],), gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )
        self._dropout_identity = isinstance(self.net[3], nn.Dropout) and self.net[3].p == 0.
        self._out_dropout_identity = isinstance(self.net[5], nn.Dropout) and self.net[5].p == 0.
        self._fused_linear_gelu_dim_candidate = dim in (256, 384, 512)
        self._fc1_dtype_cache = {}

    def _fc1_params(self, dtype, device):
        linear = self.net[1]
        key = (device.type, device.index, dtype, linear.weight.data_ptr(), linear.weight._version, None if linear.bias is None else linear.bias.data_ptr(), None if linear.bias is None else linear.bias._version)
        cached = self._fc1_dtype_cache.get(key)
        if cached is None:
            cached = (
                linear.weight.detach().to(device=device, dtype=dtype),
                None if linear.bias is None else linear.bias.detach().to(device=device, dtype=dtype),
            )
            self._fc1_dtype_cache.clear()
            self._fc1_dtype_cache[key] = cached
        return cached

    def _use_fused_linear_gelu(self, x):
        if not self._fused_linear_gelu_dim_candidate:
            return False
        rows = x.numel() // x.shape[-1]
        return (
            not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.ndim == 3
            and x.is_contiguous()
            and x.shape[-1] in (256, 384, 512)
            and 16384 <= rows <= 230000
            and isinstance(self.net[2], nn.GELU)
            and self.net[2].approximate == "none"
            and hasattr(torch.ops.aten, "_addmm_activation")
        )

    def _fused_linear_gelu(self, x):
        linear = self.net[1]
        weight, bias = self._fc1_params(x.dtype, x.device)
        return torch.ops.aten._addmm_activation(bias, x.reshape(-1, x.shape[-1]), weight.t(), beta=1, alpha=1, use_gelu=True).reshape(*x.shape[:-1], linear.out_features)

    def forward(self, x):
        if not self._fused_linear_gelu_dim_candidate:
            return self.net(x)
        if not self._use_fused_linear_gelu(x):
            return self.net(x)
        normed = self.net[0](x)
        return self.net[5](self.net[4](self._fused_linear_gelu(normed)))

    def forward_residual(self, x):
        if isinstance(x, tuple):
            normed, residual = x
            if not self._fused_linear_gelu_dim_candidate:
                return self.net[1:](normed) + residual
            return (self.net[5](self.net[4](self._fused_linear_gelu(normed))) if self._use_fused_linear_gelu(normed) else self.net[1:](normed)) + residual
        if not self._fused_linear_gelu_dim_candidate:
            if (
                not self.training
                and not torch.is_grad_enabled()
                and x.is_cuda
                and x.dtype in (torch.float16, torch.bfloat16)
                and x.ndim == 3
                and x.is_contiguous()
            ):
                return x.add_(self.net(x))
            return self.net(x) + x
        if (
            not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.ndim == 3
            and x.is_contiguous()
        ):
            if not self._use_fused_linear_gelu(x):
                return x.add_(self.net(x))
            return x.add_(self.net[5](self.net[4](self._fused_linear_gelu(self.net[0](x)))))
        return self.forward(x) + x


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            shared_qkv_bias=None,
            shared_out_bias=None,
            rotary_embed=None,
            flash=True,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head
        self.flash = flash
        self.dropout = dropout
        self.rotary_embed = rotary_embed
        self.mps_attention_backend = "torch"
        self.mps_mlx_min_tokens = 128
        self.cuda_attention_backend = default_cuda_attention_backend()
        self.cuda_triton_backend = default_cuda_triton_backend()
        self.approx_kv_stride = 1
        self.approx_kv_stride_mode = "avg"
        self._disabled_cuda_attention_backends = set()
        self.attend = Attend(flash=False, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout)
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias
        self._to_out_dtype_cache = {}

    def set_mps_attention_backend(self, backend=None, min_tokens=128):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx", "mlx_attention", "mlx_transformer"):
            raise ValueError("mps_attention_backend must be 'torch', 'mlx', 'mlx_attention', or 'mlx_transformer'")
        self.mps_attention_backend = "torch" if backend == "mlx_transformer" else backend
        self.mps_mlx_min_tokens = 128 if min_tokens is None else int(min_tokens)

    def set_cuda_attention_backend(self, backend=None):
        self.cuda_attention_backend = normalize_cuda_attention_backend(backend)
        self._disabled_cuda_attention_backends.clear()

    def set_cuda_triton_backend(self, backend=None):
        self.cuda_triton_backend = normalize_cuda_triton_backend(backend)

    def set_approx_kv_stride(self, stride=None, mode="avg"):
        stride = 1 if stride is None else int(stride)
        mode = (mode or "avg").lower()
        if stride < 1:
            raise ValueError("approx_kv_stride must be >= 1")
        if mode not in ("sample", "avg"):
            raise ValueError("approx_kv_stride_mode must be 'sample' or 'avg'")
        self.approx_kv_stride = stride
        self.approx_kv_stride_mode = mode

    def _to_out_params(self, dtype, device):
        linear = self.to_out[0]
        key = (device.type, device.index, dtype, linear.weight.data_ptr(), linear.weight._version, None if linear.bias is None else linear.bias.data_ptr(), None if linear.bias is None else linear.bias._version)
        cached = self._to_out_dtype_cache.get(key)
        if cached is None:
            cached = (
                linear.weight.detach().to(device=device, dtype=dtype),
                None if linear.bias is None else linear.bias.detach().to(device=device, dtype=dtype),
            )
            self._to_out_dtype_cache.clear()
            self._to_out_dtype_cache[key] = cached
        return cached

    def _use_mlx_attention_layer(self, x):
        return (
            self.flash
            and not self.training
            and self.mps_attention_backend == "mlx_attention"
            and x.device.type == "mps"
            and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps"))
            and x.shape[-2] >= self.mps_mlx_min_tokens
        )

    def _use_mlx_sdpa(self, q):
        return (
            self.flash
            and not self.training
            and self.mps_attention_backend == "mlx"
            and q.device.type == "mps"
            and q.dtype == torch.float16
            and q.shape[-2] >= self.mps_mlx_min_tokens
        )

    def _attention(self, q, k, v):
        if self._use_mlx_sdpa(q):
            try:
                from .mlx_attention import mlx_bridge_sdpa

                return mlx_bridge_sdpa(q, k, v)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = "torch"

        if self.flash:
            return self._cuda_or_default_attention(q, k, v)
        return self.attend(q, k, v)

    def _cuda_or_default_attention(self, q, k, v):
        dropout_p = self.dropout if self.training else 0.
        backend = self.cuda_attention_backend
        if not q.is_cuda or backend == "default":
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        if backend == "auto":
            for candidate in ("cudnn", "efficient"):
                if candidate in self._disabled_cuda_attention_backends:
                    continue
                try:
                    return _sdpa_with_backend(q, k, v, dropout_p, candidate)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    self._pymss_cuda_attention_backend_error = f"{candidate}: {exc!r}"
                    self._disabled_cuda_attention_backends.add(candidate)
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        try:
            if backend == "xformers":
                return _xformers_attention(q, k, v, dropout_p)
            return _sdpa_with_backend(q, k, v, dropout_p, backend)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            self._pymss_cuda_attention_backend_error = f"{backend}: {exc!r}"
            self.cuda_attention_backend = "default"
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    def _project_out(self, x, residual=None):
        linear, dropout = self.to_out
        if (
            residual is not None
            and not self.training
            and not torch.is_grad_enabled()
            and linear.bias is None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.ndim == 3
            and residual.is_contiguous()
            and residual.shape == x.shape[:-1] + (linear.out_features,)
        ):
            weight, _ = self._to_out_params(x.dtype, x.device)
            residual_flat = residual.reshape(-1, residual.shape[-1])
            torch.addmm(
                residual_flat,
                x.reshape(-1, x.shape[-1]),
                weight.t(),
                out=residual_flat,
            )
            return dropout(residual)
        projected = self.to_out(x)
        return projected if residual is None else projected + residual

    def forward(self, x, residual=None, already_normed=False):
        if self._use_mlx_attention_layer(x):
            try:
                from .mlx_attention import mlx_bridge_attention

                out = mlx_bridge_attention(self, x)
                return out if residual is None else out + residual
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = "torch"

        x = x if already_normed else self.norm(x)
        qkv, gates = self.to_qkv(x), self.to_gates(x)
        q, k, v = qkv_to_bnhd(qkv, self.heads)

        inline_short_rope = (
            self.rotary_embed is not None
            and self.cuda_triton_backend in ("auto", "freq_atomic_out")
            and not self.training
            and not torch.is_grad_enabled()
            and q.is_cuda
            and q.dtype == torch.float16
            and q.shape[1] <= 128
            and q.shape[-1] % 2 == 0
        )
        if self.rotary_embed is not None and not inline_short_rope:
            q, k = rotate_qk_bnhd(self.rotary_embed, q, k, self.cuda_triton_backend)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        kv_stride = self.approx_kv_stride if not self.training and not torch.is_grad_enabled() else 1
        if kv_stride > 1:
            if inline_short_rope:
                q, k = rotate_qk_bnhd(self.rotary_embed, q.transpose(1, 2), k.transpose(1, 2), self.cuda_triton_backend)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                inline_short_rope = False
            k, v = pool_kv_sequence(k, kv_stride, self.approx_kv_stride_mode), pool_kv_sequence(v, kv_stride, self.approx_kv_stride_mode)

        if self.cuda_triton_backend in ("attention_gate_out",) and not self.training and not torch.is_grad_enabled():
            fused = attention_gate_out_triton(q, k, v, gates, *self._to_out_params(q.dtype, q.device), residual=residual)
            if fused is not None:
                return self.to_out[1](fused)
            raise RuntimeError("cuda_triton_backend='attention_gate_out' requires Triton attention+out support for this shape/dtype")

        if self.cuda_triton_backend in ("auto", "freq_atomic_out") and not self.training and not torch.is_grad_enabled():
            try:
                if q.shape[2] <= 128:
                    weight, bias = self._to_out_params(q.dtype, q.device)
                    fused = None
                    if residual is not None and isinstance(self.to_out[1], nn.Dropout):
                        if inline_short_rope:
                            cos, sin = cached_rotary_cos_sin(self.rotary_embed, q.shape[2], q.device, q.dtype)
                            fused = attention_gate_out_rope_inplace_triton(q, k, v, gates, weight, residual, cos, sin, bias)
                            if fused is None:
                                q, k = rotate_qk_bnhd(self.rotary_embed, q.transpose(1, 2), k.transpose(1, 2), self.cuda_triton_backend)
                                q, k = q.transpose(1, 2), k.transpose(1, 2)
                                inline_short_rope = False
                        if fused is None:
                            fused = attention_gate_out_inplace_triton(q, k, v, gates, weight, residual, bias)
                    if fused is None:
                        if inline_short_rope:
                            q, k = rotate_qk_bnhd(self.rotary_embed, q.transpose(1, 2), k.transpose(1, 2), self.cuda_triton_backend)
                            q, k = q.transpose(1, 2), k.transpose(1, 2)
                            inline_short_rope = False
                        fused = attention_gate_out_triton(q, k, v, gates, weight, bias, residual=residual)
                    if fused is not None:
                        return self.to_out[1](fused)
                gated = attention_gate_packed64_5090_triton(q, k, v, gates)
                if gated is None:
                    gated = attention_gate_triton(q, k, v, gates)
                if gated is None:
                    gated = attention_gate_varlen_triton(q, k, v, gates)
                if gated is not None:
                    return self._project_out(gated, residual)
                if self.cuda_triton_backend == "freq_atomic_out":
                    raise RuntimeError("cuda_triton_backend='freq_atomic_out' requires Triton attention support for this shape/dtype")
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                if self.cuda_triton_backend == "freq_atomic_out":
                    raise
                if inline_short_rope:
                    q, k = rotate_qk_bnhd(self.rotary_embed, q.transpose(1, 2), k.transpose(1, 2), self.cuda_triton_backend)
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                    inline_short_rope = False
                self._pymss_cuda_triton_backend_error = repr(exc)
                self.cuda_triton_backend = "default"

        if self.cuda_triton_backend == "attention_gate" and not self.training and not torch.is_grad_enabled():
            gated = attention_gate_packed64_5090_triton(q, k, v, gates)
            if gated is None:
                gated = attention_gate_triton(q, k, v, gates)
            if gated is None:
                gated = attention_gate_varlen_triton(q, k, v, gates)
            if gated is not None:
                return self._project_out(gated, residual)
            raise RuntimeError("cuda_triton_backend='attention_gate' requires Triton attention support for this shape/dtype")

        out = self._attention(q, k, v)
        return self._project_out((out.transpose(1, 2) * gates.unsqueeze(-1).sigmoid()).flatten(start_dim=-2), residual)

    def forward_residual(self, x):
        return self.forward(x, residual=x)

    def forward_normed_residual(self, normed, residual):
        return self.forward(normed, residual=residual, already_normed=True)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            shared_qkv_bias=None,
            shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList([
            ModuleList([
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ])
            for _ in range(depth)
        ])

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

        self.mps_attention_backend = "torch"
        self.mps_mlx_min_tokens = 128
        self.cuda_attention_backend = default_cuda_attention_backend()
        self.cuda_triton_backend = default_cuda_triton_backend()
        self.approx_kv_stride = 1
        self.approx_kv_stride_mode = "avg"

    def set_mps_attention_backend(self, backend=None, min_tokens=128):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx", "mlx_attention", "mlx_transformer"):
            raise ValueError("mps_attention_backend must be 'torch', 'mlx', 'mlx_attention', or 'mlx_transformer'")
        self.mps_attention_backend = backend
        self.mps_mlx_min_tokens = 128 if min_tokens is None else int(min_tokens)
        child_backend = "torch" if backend == "mlx_transformer" else backend
        for attn, _ in self.layers:
            attn.set_mps_attention_backend(child_backend, self.mps_mlx_min_tokens)

    def set_cuda_attention_backend(self, backend=None):
        self.cuda_attention_backend = normalize_cuda_attention_backend(backend)
        for attn, _ in self.layers:
            attn.set_cuda_attention_backend(self.cuda_attention_backend)

    def set_cuda_triton_backend(self, backend=None):
        self.cuda_triton_backend = normalize_cuda_triton_backend(backend)
        for attn, _ in self.layers:
            attn.set_cuda_triton_backend(self.cuda_triton_backend)

    def set_approx_kv_stride(self, stride=None, mode="avg"):
        self.approx_kv_stride = 1 if stride is None else int(stride)
        self.approx_kv_stride_mode = (mode or "avg").lower()
        for attn, _ in self.layers:
            attn.set_approx_kv_stride(self.approx_kv_stride, self.approx_kv_stride_mode)

    def _use_mlx_transformer(self, x):
        return (
            self.mps_attention_backend == "mlx_transformer"
            and not self.training
            and x.device.type == "mps"
            and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps"))
            and x.shape[-2] >= self.mps_mlx_min_tokens
        )

    def forward(self, x):
        if self._use_mlx_transformer(x):
            try:
                from .mlx_attention import mlx_bridge_transformer

                return mlx_bridge_transformer(self, x)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.set_mps_attention_backend("torch", self.mps_mlx_min_tokens)

        for attn, ff in self.layers:
            x = attn.forward_residual(x)
            x = ff.forward_residual(x)
        return self.norm(x)

    def forward_from_normed_first(self, normed, residual):
        attn, ff = self.layers[0]
        x = ff.forward_residual(attn.forward_normed_residual(normed, residual))
        for attn, ff in self.layers[1:]:
            x = attn.forward_residual(x)
            x = ff.forward_residual(x)
        return self.norm(x)
