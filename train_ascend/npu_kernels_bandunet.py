"""triton-ascend fused kernels for the frequency U-Net (band-split / RoPE / gated attention out).

Verified on Ascend 950PR with triton-ascend 3.6.0.post2 (triton 3.6.0, npu backend),
torch 2.13.0+cpu + torch_npu 2.13.0.

Every kernel is guarded by `triton_ascend_available()` and has a numerically
equivalent eager fallback, so the model path works either way:

    1) band-split fusion   `bandsplit_norm_scale`   fused F.normalize*scale*gamma
    2) RoPE fusion         `fused_rope`             in-place interleaved->halves rotation
    3) gated attention out `fused_gated_out`        attn * sigmoid(gates) without broadcast materialization

`PYMSS_NPU_TRITON=0` disables the fused paths.
"""

from __future__ import annotations

import importlib.metadata as md
import os

import torch

TRITON_ENV = "PYMSS_NPU_TRITON"
_available: bool | None = None
_reason: str = "not probed"


def triton_ascend_available() -> bool:
    """True only when triton is importable *and* driving an NPU backend."""
    global _available, _reason
    if _available is not None:
        return _available
    if os.environ.get(TRITON_ENV, "1") == "0":
        _available, _reason = False, f"{TRITON_ENV}=0"
        return _available
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            _available, _reason = False, "torch_npu present but no NPU device"
            return _available
        import triton

        target = triton.runtime.driver.active.get_current_target()
        backend = str(getattr(target, "backend", "")).lower()
        if "npu" not in backend:
            _available, _reason = False, f"triton backend is {backend!r}, not npu"
            return _available
        _available = True
        _reason = f"triton {md.version('triton')} + triton-ascend {md.version('triton-ascend')} on {backend}"
    except Exception as exc:
        _available, _reason = False, f"{type(exc).__name__}: {exc}"
    return _available


def triton_status() -> dict:
    available = triton_ascend_available()
    info = {"available": available, "reason": _reason, "env": TRITON_ENV}
    for name in ("triton", "triton-ascend"):
        try:
            info[name.replace("-", "_") + "_version"] = md.version(name)
        except Exception:
            info[name.replace("-", "_") + "_version"] = None
    return info


if triton_ascend_available():  # compiled lazily below; guarded at call sites
    import triton
    import triton.language as tl

    @triton.jit
    def _bandsplit_norm_scale_kernel(x_ptr, gamma_ptr, out_ptr, n_rows, n_feat: tl.constexpr,
                                     scale, BLOCK_M: tl.constexpr):
        """Row-wise ``F.normalize(x) * scale * gamma`` in one pass.

        Replaces normalize (norm + div) followed by scale+mul: 3 HBM round trips
        collapse into 1 read + 1 write.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, n_feat)
        rmask = rows < n_rows
        mask = rmask[:, None]
        offs = rows[:, None] * n_feat + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=cols < n_feat, other=0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(x * x, axis=1))
        norm = tl.maximum(norm, 1e-12)
        y = x / norm[:, None] * scale * gamma[None, :]
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rope_kernel(t_ptr, cos_ptr, sin_ptr, out_ptr, n_rows, D: tl.constexpr, HALF: tl.constexpr,
                     BLOCK_M: tl.constexpr):
        """``apply_rotary_emb_fast``: interleaved (even,odd) input -> [even | odd] output halves.

        even_i = t[2i]   * cos_i - t[2i+1] * sin_i
        odd_i  = t[2i+1] * cos_i + t[2i]   * sin_i
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < n_rows
        i = tl.arange(0, HALF)
        base = rows[:, None] * D
        mask = rmask[:, None]
        even_in = tl.load(t_ptr + base + 2 * i[None, :], mask=mask, other=0.0).to(tl.float32)
        odd_in = tl.load(t_ptr + base + 2 * i[None, :] + 1, mask=mask, other=0.0).to(tl.float32)
        # cos/sin are broadcast over the "1" head axis upstream -> stored as (rows_of_seq, HALF)
        cos = tl.load(cos_ptr + i[None, :], mask=i[None, :] < HALF, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + i[None, :], mask=i[None, :] < HALF, other=0.0).to(tl.float32)
        even = even_in * cos - odd_in * sin
        odd = odd_in * cos + even_in * sin
        tl.store(out_ptr + base + i[None, :], even.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + base + HALF + i[None, :], odd.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gated_out_kernel(attn_ptr, gate_ptr, out_ptr, n_rows, K: tl.constexpr, ROWS: tl.constexpr):
        """out[r, :] = attn[r, :] * sigmoid(gate[r]) -- 2-D tile, one gate load per row (no div)."""
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, K)
        rmask = rows < n_rows
        mask = rmask[:, None]
        offs = rows[:, None] * K + cols[None, :]
        a = tl.load(attn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (a * tl.sigmoid(g)[:, None]).to(out_ptr.dtype.element_ty), mask=mask)


# ------------------------------------------------------------------ fused ops
def bandsplit_norm_scale(x: torch.Tensor, gamma: torch.Tensor, block_m: int = 32) -> torch.Tensor:
    """``F.normalize(x, dim=-1) * dim**0.5 * gamma`` with the normalize+scale chain fused.

    x: (..., n_feat) contiguous, gamma: (n_feat,)
    """
    n_feat = x.shape[-1]
    scale = n_feat ** 0.5
    if not triton_ascend_available() or not x.is_contiguous() or n_feat < 16:
        return torch.nn.functional.normalize(x, dim=-1) * scale * gamma
    flat = x.reshape(-1, n_feat)
    out = torch.empty_like(flat)
    n_rows = flat.shape[0]
    grid = (triton.cdiv(n_rows, block_m),)
    _bandsplit_norm_scale_kernel[grid](flat, gamma, out, n_rows, n_feat=n_feat, scale=scale, BLOCK_M=block_m)
    return out.reshape_as(x)


def fused_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_m: int = 32) -> torch.Tensor:
    """Rotate ``t`` (…, D) in place, writing ``[even | odd]`` halves like pymss_core does."""
    d = t.shape[-1]
    half = d // 2
    if not triton_ascend_available() or not t.is_contiguous() or d < 16 or cos.numel() != half:
        even = t[..., ::2] * cos - t[..., 1::2] * sin
        odd = t[..., 1::2] * cos + t[..., ::2] * sin
        return torch.cat((even, odd), dim=-1)
    out = torch.empty_like(t)
    n_rows = t.numel() // d
    grid = (triton.cdiv(n_rows, block_m),)
    _rope_kernel[grid](t, cos, sin, out, n_rows, D=d, HALF=half, BLOCK_M=block_m)
    return out


def fused_gated_out(attn: torch.Tensor, gates: torch.Tensor, rows_per_prog: int = 32) -> torch.Tensor:
    """``attn * sigmoid(gates)[..., None]`` with one gate row read per ROWS tile.

    The naive 1-D version needs an integer divide per element and overflowed the UB
    at BLOCK=16384; a 2-D tile keeps the gate scalar in a register for K elements.
    """
    k = attn.shape[-1]
    if not triton_ascend_available() or not attn.is_contiguous() or k < 16 or gates.numel() != attn.numel() // k:
        return attn.mul_(gates.sigmoid().unsqueeze(-1))
    n_rows = attn.numel() // k
    out = torch.empty_like(attn)
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    _gated_out_kernel[grid](attn, gates.contiguous(), out, n_rows, K=k, ROWS=rows_per_prog)
    return out
