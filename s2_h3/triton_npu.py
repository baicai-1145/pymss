"""Triton kernels for the two things the profile says matter on this device (TASK §6, see PROFILE.md).

Target 1 - the time-axis depthwise conv of the MSC block (k=31, groups=C, dilation 1, layout (b*f, C, t)).
    eager per conv: forward 3.7 ms, dX 8.2 ms, **dW 122.9 ms**; 4 convs = 496 ms of the 1169 ms step, because the weight
    gradient becomes `Conv3DBackpropFilterV2` in FP32 on a shape where that kernel costs 33x its own forward.
Target 2 - the gated elementwise chain of the same block (`u * silu(v)`, one of 186 `Mul` launches, 191 ms/step).

Each stage (forward / dX / dW) can be switched to Triton independently, because the admission test is per stage: a
kernel that loses against eager on this device is simply not installed.  fp32 accumulation, same op order as the
PyTorch path.  `triton_admit.py` measures numerics and speed, TRITON.md records the verdict.

Installed only when `S2_TRITON` names the kernel (`dwconv_fwd,dwconv_dx,dwconv_dw,silu_mul` or `all`); `stage.py`'s
56-window evaluation always stays eager, so no reported dB ever depends on this file.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = tl = None


def available() -> bool:
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _dw_fwd(X, W, B, Y, C, T, K, BC: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, HAS_B: tl.constexpr):
        """Y[n, c, t] = sum_k W[c, k] * X[n, c, t + k - K//2] + B[c].

        One 3-D tile (c, k, t) per program: the K taps are a single load of a dense band, so a sample is fetched once
        and used for every tap it contributes to, instead of K shifted passes over the row.
        """
        n, pt, pc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        oc = pc * BC + tl.arange(0, BC)
        ok = tl.arange(0, BK)
        ot = pt * BT + tl.arange(0, BT)
        mc, mk = oc < C, ok < K
        src = ot[None, :] + ok[:, None] - K // 2
        m = mc[:, None, None] & mk[None, :, None] & (src >= 0)[None, :, :] & (src < T)[None, :, :]
        x = tl.load(X + n * C * T + oc[:, None, None] * T + src[None, :, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + oc[:, None] * K + ok[None, :], mask=mc[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(x * w[:, :, None], axis=1)
        if HAS_B:
            acc += tl.load(B + oc, mask=mc, other=0.0)[:, None]
        tl.store(Y + n * C * T + oc[:, None] * T + ot[None, :], acc.to(Y.dtype.element_ty),
                 mask=mc[:, None] & (ot < T)[None, :])

    @triton.jit
    def _dw_dx(DY, W, DX, C, T, K, BC: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        """DX[n, c, t] = sum_k W[c, k] * DY[n, c, t - k + K//2]  (adjoint of the forward)."""
        n, pt, pc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        oc = pc * BC + tl.arange(0, BC)
        ok = tl.arange(0, BK)
        ot = pt * BT + tl.arange(0, BT)
        mc, mk = oc < C, ok < K
        src = ot[None, :] - ok[:, None] + K // 2
        m = mc[:, None, None] & mk[None, :, None] & (src >= 0)[None, :, :] & (src < T)[None, :, :]
        d = tl.load(DY + n * C * T + oc[:, None, None] * T + src[None, :, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + oc[:, None] * K + ok[None, :], mask=mc[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(d * w[:, :, None], axis=1)
        tl.store(DX + n * C * T + oc[:, None] * T + ot[None, :], acc.to(DX.dtype.element_ty),
                 mask=mc[:, None] & (ot < T)[None, :])

    @triton.jit
    def _dw_dw(DY, X, DW, DB, N, C, T, K, NS: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
               HAS_B: tl.constexpr):
        """DW[c, k] = sum_{n, t} DY[n, c, t] * X[n, c, t + k - K//2].  Rows are split over NS programs, atomic add."""
        pc, ps = tl.program_id(0), tl.program_id(1)
        oc = pc * BC + tl.arange(0, BC)
        ok = tl.arange(0, BK)
        mc, mk = oc < C, ok < K
        acc = tl.zeros((BC, BK), dtype=tl.float32)
        accb = tl.zeros((BC,), dtype=tl.float32)
        for n in range(ps, N, NS):
            base = n * C * T
            for t0 in range(0, T, BT):
                ot = t0 + tl.arange(0, BT)
                mt = ot < T
                src = ot[None, :] + ok[:, None] - K // 2
                m = mc[:, None, None] & mk[None, :, None] & (src >= 0)[None, :, :] & (src < T)[None, :, :]
                x = tl.load(X + base + oc[:, None, None] * T + src[None, :, :], mask=m, other=0.0).to(tl.float32)
                dy = tl.load(DY + base + oc[:, None] * T + ot[None, :], mask=mc[:, None] & mt[None, :],
                             other=0.0).to(tl.float32)
                acc += tl.sum(x * dy[:, None, :], axis=2)
                if HAS_B:
                    accb += tl.sum(dy, axis=1)
        tl.atomic_add(DW + oc[:, None] * K + ok[None, :], acc, mask=mc[:, None] & mk[None, :])
        if HAS_B:
            tl.atomic_add(DB + oc, accb, mask=mc)

    @triton.jit
    def _silu_mul(A, G, OUT, n, BLOCK: tl.constexpr):
        """out = silu(A) * G."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + offs, (a * tl.sigmoid(a) * g).to(OUT.dtype.element_ty), mask=m)


STAGES = ('dwconv_fwd', 'dwconv_dx', 'dwconv_dw', 'silu_mul')


def _cfg():
    """Tile sizes (thinned on this device: the (BC, BK, BT) tile must fit the 254 KB UB)."""
    return dict(bc=int(os.environ.get('S2_TRITON_BC', 1)), bt=int(os.environ.get('S2_TRITON_BT', 128)),
                bk=int(os.environ.get('S2_TRITON_BK', 32)), ns=int(os.environ.get('S2_TRITON_NS', 4)),
                block=int(os.environ.get('S2_TRITON_BLOCK', 8192)))


def _pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _match_dtype(weight, bias, like):
    """Autocast does not reach into a custom Function, so cast weight/bias to the activation dtype the way it would."""
    if like.dtype in (torch.float16, torch.bfloat16):
        weight = weight.to(like.dtype)
        bias = bias.to(like.dtype) if bias is not None else None
    return weight, bias


class _DWConv(torch.autograd.Function):
    """Depthwise Conv1d (groups=C, dilation 1); each of forward / dX / dW runs in Triton when asked to."""

    @staticmethod
    def forward(ctx, x, weight, bias, pad, stages):
        n, c, t = x.shape
        k = weight.shape[-1]
        cfg = _cfg()
        x = x.contiguous()
        w_dtype = weight.dtype
        weight, bias = _match_dtype(weight, bias, x)
        ctx.save_for_backward(x, weight)
        ctx.cfg, ctx.stages, ctx.has_bias = cfg, stages, bias is not None
        ctx.w_dtype = w_dtype
        if 'dwconv_fwd' not in stages:
            return F.conv1d(x, weight, bias, padding=pad, groups=c)   # same kernel autograd would pick
        out = torch.empty_like(x)
        grid = (n, triton.cdiv(t, cfg['bt']), triton.cdiv(c, cfg['bc']))
        _dw_fwd[grid](x, weight.reshape(c, k), bias if bias is not None else x, out, c, t, k, BC=cfg['bc'],
                      BT=cfg['bt'], BK=_pow2(k), HAS_B=bias is not None, num_warps=1)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        cfg, stages = ctx.cfg, ctx.stages
        n, c, t = x.shape
        k = weight.shape[-1]
        grad_out = grad_out.contiguous()
        pad = k // 2
        weight, _ = _match_dtype(weight, None, x)
        if 'dwconv_dx' in stages:
            dx = torch.empty_like(x)
            grid = (n, triton.cdiv(t, cfg['bt']), triton.cdiv(c, cfg['bc']))
            _dw_dx[grid](grad_out, weight.reshape(c, k), dx, c, t, k, BC=cfg['bc'], BT=cfg['bt'], BK=_pow2(k),
                         num_warps=1)
        else:
            dx = F.conv1d(grad_out, weight.flip(-1), padding=pad, groups=c)
        if 'dwconv_dw' in stages:
            dw = torch.zeros(c, k, device=x.device, dtype=torch.float32)
            db = torch.zeros(c, device=x.device, dtype=torch.float32) if ctx.has_bias else dw
            grid = (triton.cdiv(c, cfg['bc']), cfg['ns'])
            _dw_dw[grid](grad_out, x, dw, db, n, c, t, k, NS=cfg['ns'], BC=cfg['bc'], BT=cfg['bt'], BK=_pow2(k),
                         HAS_B=ctx.has_bias, num_warps=1)
        else:                                                        # 31 shifted masked reductions
            pad_x = F.pad(x, (pad, pad))
            dw = torch.stack([(grad_out * pad_x[:, :, j:j + t]).sum(dim=(0, 2)) for j in range(k)], dim=-1)
            db = grad_out.sum(dim=(0, 2))
        return dx, dw.to(ctx.w_dtype).reshape(weight.shape), (db.to(ctx.w_dtype) if ctx.has_bias else None), None, None


def depthwise_conv1d(x, weight, bias, padding, stages=STAGES):
    """Drop-in for nn.Conv1d(groups=C, dilation=1); `stages` selects which parts run in Triton."""
    return _DWConv.apply(x, weight, bias, padding, stages)


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, g):
        out = torch.empty_like(a)
        n = a.numel()
        block = _cfg()['block']
        _silu_mul[(triton.cdiv(n, block),)](a.contiguous(), g.contiguous(), out, n, BLOCK=block, num_warps=4)
        ctx.save_for_backward(a, g)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, g = ctx.saved_tensors
        s = torch.sigmoid(a)
        return grad_out * g * (s * (1 + a * (1 - s))), grad_out * (a * s)


def silu_mul(a, g):
    """silu(a) * g."""
    return _SiluMul.apply(a, g)


def _conv_forward(self, x):
    if self.dilation[0] != 1 or self.padding[0] != self.kernel_size[0] // 2 or self.groups != self.in_channels:
        return self._s2_eager_forward(x)
    return depthwise_conv1d(x, self.weight, self.bias, self.padding[0], self._s2_stages)


def _msc_forward(self, x):
    """entry.MSC.forward with `u * silu(v)` replaced by the fused kernel; every other op unchanged."""
    b, t, f, d = x.shape
    u, v = self.proj_in(x).chunk(2, dim=-1)
    g = self.gate_drop(silu_mul(v, u))
    hid = g.shape[-1]
    g = g.permute(0, 2, 1, 3).reshape(b * f, t, hid).transpose(1, 2)
    g = sum(c(g) for c in self.convs)
    g = g.transpose(1, 2).reshape(b, f, t, hid).permute(0, 2, 1, 3)
    return self.drop(self.proj_out(g))


def install(model, names):
    """Patch the requested kernels into a built model.  names: any of STAGES, or 'all'."""
    assert HAS_TRITON, 'triton is not importable'
    names = set(STAGES) if 'all' in names else set(names) & set(STAGES)
    assert names, f'nothing to install from {names}'
    done = []
    conv_stages = tuple(s for s in ('dwconv_fwd', 'dwconv_dx', 'dwconv_dw') if s in names)
    if conv_stages:
        count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d) and module.groups == module.in_channels == module.out_channels:
                module._s2_eager_forward = module.forward
                module._s2_stages = conv_stages
                module.forward = types.MethodType(_conv_forward, module)
                count += 1
        assert count, 'no depthwise Conv1d found'
        done.append(f'{conv_stages} on {count} convs')
    if 'silu_mul' in names:
        from entry import MSC
        MSC.forward = _msc_forward
        done.append('silu_mul')
    return done
