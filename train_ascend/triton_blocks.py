"""Triton kernels for the parts of the U-Net blocks that PyTorch runs inefficiently.

Measured on this device at bs24 (bf16, 690 frames): a ``LocalBandBlock`` costs 29.9 ms
forward, of which only 3.8 ms is GEMM. The rest is RMSNorm (4.3 ms per call, as
``F.normalize * scale * gamma`` with an fp32 divide), the layout transposes needed to feed
``Conv1d`` along the band axis (5.2 ms) and the elementwise chain. A Triton kernel that
does band-RMSNorm + depthwise band conv in one pass removes those passes.

Everything here is math-equivalent to the PyTorch path; ``triton_fused_check.py``
verifies forward and gradient against it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl
    import triton.language.extra.cann.libdevice as tld

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None


def available() -> bool:
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(X, GAMMA, OUT, M, F, C, eps, BF: tl.constexpr, BC: tl.constexpr):
        """out[m, f, :] = x[m, f, :] / sqrt(mean(x^2) + eps) * gamma."""
        pid_m = tl.program_id(0)
        pid_f = tl.program_id(1)
        offs_f = pid_f * BF + tl.arange(0, BF)
        offs_c = tl.arange(0, BC)
        mask = (offs_f[:, None] < F) & (offs_c[None, :] < C)
        ptr = X + pid_m * F * C + offs_f[:, None] * C + offs_c[None, :]
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / C
        # tl.sqrt/rsqrt are the approximate (fast-math) variants here: 7e-05 error.
        r = tld.rsqrt_rn(ms + eps)
        y = x * r[:, None] * gamma[None, :]
        tl.store(OUT + pid_m * F * C + offs_f[:, None] * C + offs_c[None, :], y.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _dwconv_band_kernel(N, W, B, OUT, M, F, C, K: tl.constexpr, BF: tl.constexpr, BC: tl.constexpr):
        """out[m, f, c] = sum_k w[c, k] * n[m, f + k - K//2, c] + b[c]  (conv along bands)."""
        pid_m = tl.program_id(0)
        pid_f = tl.program_id(1)
        offs_f = pid_f * BF + tl.arange(0, BF)
        offs_c = tl.arange(0, BC)
        acc = tl.zeros((BF, BC), dtype=tl.float32)
        half = K // 2
        for k in tl.static_range(K):
            src = offs_f + k - half
            mask = (src[:, None] >= 0) & (src[:, None] < F) & (offs_c[None, :] < C)
            ptr = N + pid_m * F * C + src[:, None] * C + offs_c[None, :]
            n = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + offs_c * K + k, mask=offs_c < C, other=0.0).to(tl.float32)
            acc += n * w[None, :]
        b = tl.load(B + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += b[None, :]
        mask_o = (offs_f[:, None] < F) & (offs_c[None, :] < C)
        tl.store(OUT + pid_m * F * C + offs_f[:, None] * C + offs_c[None, :], acc.to(OUT.dtype.element_ty), mask=mask_o)

    @triton.jit
    def _dwconv_band_bwd_kernel(DH, W, DN, M, F, C, K: tl.constexpr, BF: tl.constexpr, BC: tl.constexpr):
        """Adjoint of the band conv: dn[m, f, c] = sum_k dh[m, f - k + K//2, c] * w[c, k]."""
        pid_m = tl.program_id(0)
        pid_f = tl.program_id(1)
        offs_f = pid_f * BF + tl.arange(0, BF)
        offs_c = tl.arange(0, BC)
        acc = tl.zeros((BF, BC), dtype=tl.float32)
        half = K // 2
        for k in tl.static_range(K):
            src = offs_f - k + half
            mask = (src[:, None] >= 0) & (src[:, None] < F) & (offs_c[None, :] < C)
            ptr = DH + pid_m * F * C + src[:, None] * C + offs_c[None, :]
            d = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + offs_c * K + k, mask=offs_c < C, other=0.0).to(tl.float32)
            acc += d * w[None, :]
        mask_o = (offs_f[:, None] < F) & (offs_c[None, :] < C)
        tl.store(DN + pid_m * F * C + offs_f[:, None] * C + offs_c[None, :], acc.to(DN.dtype.element_ty), mask=mask_o)

    @triton.jit
    def _rmsnorm_bwd_kernel(X, GAMMA, DN, DX, M, F, C, eps,
                            BF: tl.constexpr, BC: tl.constexpr):
        """dx = r*gamma*dn - r^3/C * x * (x . (gamma*dn)), r = rsqrt(mean(x^2)+eps)."""
        pid_m = tl.program_id(0)
        pid_f = tl.program_id(1)
        of = pid_f * BF + tl.arange(0, BF)
        oc = tl.arange(0, BC)
        mask = (of[:, None] < F) & (oc[None, :] < C)
        base = pid_m * F * C + of[:, None] * C + oc[None, :]
        x = tl.load(X + base, mask=mask, other=0.0).to(tl.float32)
        dn = tl.load(DN + base, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(GAMMA + oc, mask=oc < C, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / C
        r = tld.rsqrt_rn(ms + eps)
        gd = gamma[None, :] * dn
        proj = tl.sum(x * gd, axis=1)[:, None] / C
        r3 = r * r * r
        dx = r[:, None] * gd - r3[:, None] * x * proj
        tl.store(DX + base, dx.to(DX.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_mul_kernel(A, G, OUT, n_elem, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
        y = (a * tl.sigmoid(a)) * g
        tl.store(OUT + offs, y.to(OUT.dtype.element_ty), mask=mask)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _grid(M: int, F: int, BF: int) -> tuple[int, int]:
    return (M, triton.cdiv(F, BF))


def fused_norm_dwconv(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor | None, eps: float = 1e-12,
                      bf: int = 8, bc: int | None = None) -> torch.Tensor:
    """RMSNorm over the channel axis then depthwise conv along the band axis.

    Args:
        x: (M, F, C) contiguous.
        gamma: (C,) RMSNorm gain.
        weight: (C, 1, K) depthwise conv weight (groups=C).
        bias: (C,) optional.
    """
    assert x.dim() == 3 and x.is_contiguous()
    M, Fb, C = x.shape
    bc = _pow2(C) if bc is None else bc
    k = weight.shape[-1]
    w2 = weight.reshape(C, k).contiguous()
    b = bias if bias is not None else torch.zeros(C, device=x.device, dtype=x.dtype)
    n = torch.empty_like(x)
    _rmsnorm_kernel[_grid(M, Fb, bf)](x, gamma, n, M, Fb, C, eps, BF=bf, BC=bc, num_warps=1)
    out = torch.empty_like(x)
    _dwconv_band_kernel[_grid(M, Fb, bf)](n, w2, b, out, M, Fb, C, K=k, BF=bf, BC=bc, num_warps=1)
    return out


class _SiluMul(torch.autograd.Function):
    """silu(a) * g with an explicit (exact) backward.

    A bare Triton call returns a tensor with no grad_fn, which silently cuts the graph for
    every parameter upstream of it - it cost the first overnight run 3100 steps before the
    zero gradient norm was noticed.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        n = a.numel()
        block = 8192
        _silu_mul_kernel[(triton.cdiv(n, block),)](a, g, out, n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(a, g)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, g = ctx.saved_tensors
        s = torch.sigmoid(a)
        # d/da [a*s*g] = g*(s + a*s*(1-s));  d/dg = a*s
        da = grad_out * g * (s * (1 + a * (1 - s)))
        dg = grad_out * (a * s)
        return da, dg


def fused_silu_mul(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Autograd-capable silu(a)*g (one kernel forward, exact analytic backward)."""
    return _SiluMul.apply(a, g)


class _NormDwconv(torch.autograd.Function):
    """RMSNorm -> depthwise band conv, forward in Triton, analytic backward."""

    @staticmethod
    def forward(ctx, x, gamma, weight, bias, eps, bf, bc):
        M, Fb, C = x.shape
        k = weight.shape[-1]
        w2 = weight.reshape(C, k).contiguous()
        b = bias if bias is not None else torch.zeros(C, device=x.device, dtype=x.dtype)
        n = torch.empty_like(x)
        _rmsnorm_kernel[_grid(M, Fb, bf)](x, gamma, n, M, Fb, C, eps, BF=bf, BC=bc, num_warps=1)
        out = torch.empty_like(x)
        _dwconv_band_kernel[_grid(M, Fb, bf)](n, w2, b, out, M, Fb, C, K=k, BF=bf, BC=bc, num_warps=1)
        ctx.save_for_backward(x, gamma, w2, n)
        ctx.cfg = (M, Fb, C, k, eps, bf, bc)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, gamma, w2, n = ctx.saved_tensors
        M, Fb, C, k, eps, bf, bc = ctx.cfg
        dn = torch.empty_like(n)
        _dwconv_band_bwd_kernel[_grid(M, Fb, bf)](
            grad_out.contiguous(), w2, dn, M, Fb, C, K=k, BF=bf, BC=bc, num_warps=1)
        dx = torch.empty_like(x)
        _rmsnorm_bwd_kernel[_grid(M, Fb, bf)](
            x, gamma, dn, dx, M, Fb, C, eps, BF=bf, BC=bc, num_warps=1)
        # dgamma needs x*r, which the kernel does not emit; one extra elementwise pass
        # (cheaper and more readable than atomics across 130k programs).
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        r = torch.rsqrt(ms + eps)
        dgamma = ((x * r) * dn).sum(dim=(0, 1))
        # weight/bias grads: small reductions, done with torch (exact, cheap)
        # weight/bias grads use the gradient w.r.t. the conv OUTPUT, not dn.
        pad = k // 2
        npad = F.pad(n, (0, 0, pad, pad))
        go = grad_out
        dw = torch.stack([(go * npad[:, j:j + Fb, :]).sum(dim=(0, 1)) for j in range(k)], dim=-1)
        db = go.sum(dim=(0, 1))
        return dx, dgamma, dw.reshape(C, 1, k), db, None, None, None


def norm_dwconv(x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, eps: float = 1e-12,
                bf: int = 8, bc: int | None = None) -> torch.Tensor:
    """Autograd-capable fused RMSNorm + band depthwise conv."""
    bc = _pow2(x.shape[-1]) if bc is None else bc
    return _NormDwconv.apply(x, gamma, weight, bias, eps, bf, bc)
