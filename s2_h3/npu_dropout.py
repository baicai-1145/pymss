"""Graph-safe dropout for Ascend: the mask is drawn on-device from a seed tensor, never from host RNG.

`nn.Dropout` raises inside an ACLGraph capture (`aclnnDropoutV3Tensor` is unavailable) and `torch.rand` is a host
generator, so a captured step can never see a fresh mask.  Here the mask comes from a 1-element int64 NPU tensor:
increment `seed_buf` **outside** the graph before each replay and every replay gets an independent mask while the
captured program stays identical.  Forward writes `x * mask * scale`, backward multiplies by the same mask, so the
function matches `nn.Dropout` exactly.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = tl = None


if HAS_TRITON:
    @triton.jit
    def _dropout_fwd(X, Y, M, SEED, N, P: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        seed = tl.load(SEED).to(tl.uint32)
        h = offs.to(tl.uint32) * tl.full((), 2654435761, tl.uint32) + seed * tl.full((), 0x9E3779B1, tl.uint32)
        h = (h ^ (h >> 13)) * tl.full((), 0x85EBCA6B, tl.uint32)
        h = h ^ (h >> 16)
        keep = (h.to(tl.float32) * 2.3283064365386963e-10) >= P          # uniform in [0,1)
        scale: tl.constexpr = 1.0 / (1.0 - P)
        x = tl.load(X + offs, mask=mask, other=0.0)
        k = tl.where(keep, 1.0, 0.0) * scale
        tl.store(M + offs, k.to(M.dtype.element_ty), mask=mask)
        tl.store(Y + offs, (x * k).to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _dropout_bwd(DY, M, DX, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        dy = tl.load(DY + offs, mask=mask, other=0.0)
        m = tl.load(M + offs, mask=mask, other=0.0)
        tl.store(DX + offs, (dy * m).to(DX.dtype.element_ty), mask=mask)


class _StatelessDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed_buf):
        x = x.contiguous()
        y = torch.empty_like(x)
        mask = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        n = x.numel()
        blk = 8192
        _dropout_fwd[(triton.cdiv(n, blk),)](x.view(-1), y.view(-1), mask.view(-1), seed_buf, n, P=float(p),
                                             BLOCK=blk, num_warps=1)
        ctx.save_for_backward(mask)
        ctx.p = float(p)
        return y

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(grad)
        n = grad.numel()
        blk = 8192
        _dropout_bwd[(triton.cdiv(n, blk),)](grad.view(-1), mask.view(-1), dx.view(-1), n, BLOCK=blk, num_warps=1)
        return dx, None, None


_SEED = None


def seed_buffer(device=None):
    """The process-wide seed tensor; creation is idempotent so callers never thread it through modules."""
    global _SEED
    if _SEED is None:
        _SEED = torch.zeros(1, device=device, dtype=torch.int64)
    return _SEED


def bump():
    """Advance the seed.  Must be called OUTSIDE the captured region, once per replay."""
    seed_buffer().add_(1)


def dropout(x, p, seed_buf=None):
    if not HAS_TRITON or p <= 0.0:
        return x
    return _StatelessDropout.apply(x, p, seed_buffer(x.device) if seed_buf is None else seed_buf)


def install(model, seed_buf):
    """Route **all** dropout through the stateless kernel and return the number of `nn.Dropout` modules seen.

    Patching `nn.Dropout` modules is not enough: the attention paths call `F.dropout(a, p)` directly, and that
    functional call is what still reached `NPUGeneratorImpl::philox_engine_inputs` and aborted the capture with
    "Refactor this op to use NPUGeneratorImpl::philox_npu_state" even after every module was replaced.
    """
    original = torch.nn.functional.dropout
    if getattr(original, '_pymss_stateless', False):
        return 0

    def _stateless_dropout(input, p=0.5, training=True, inplace=False):  # noqa: A002 - torch's signature
        if not training or p <= 0.0:
            return input
        out = dropout(input, p, seed_buf)
        if inplace:
            input.copy_(out)
            return input
        return out

    _stateless_dropout._pymss_stateless = True
    torch.nn.functional.dropout = _stateless_dropout
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0.0)
