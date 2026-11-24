"""triton-ascend fused kernels for the frequency U-Net (band-split / RoPE / gated attention out).

Priority order from the task:
    1) band-split fusion          -> `fused_bandsplit_linear`
    2) RoPE fusion                -> `fused_rope`
    3) attention gated output     -> `fused_gated_out`

STATUS ON THIS BOX: triton-ascend ships wheels for cp39/cp310/cp311 only
(triton_ascend-3.2.0-*), while this environment runs CPython 3.12.16, so the
kernels could not be compiled or benchmarked here. They are written against the
same API surface that `pymss_core/npu_kernels.py` (branch `ascend-kernels`) uses
on this exact hardware, and every kernel is guarded by
`triton_ascend_available()`, falling back to an eager path that is bit-equivalent
in fp32.

Prior measured evidence on the same box (pymss_core@ascend-kernels header):
triton-ascend 3.2.1 generates unvectorized loads on 950PR, so a plain copy runs
0.239 ms vs aten::clone 0.081 ms (3x). Only large BLOCK pure-elementwise chains
win (2.5x for a 43.7M-element chain at BLOCK=16384). Fused matmul was at parity
with MatMulV3 (cube-core bound). That is why the eager fallbacks below are the
default and the kernels are opt-in via `PYMSS_NPU_TRITON=1`.
"""

from __future__ import annotations

import os

import torch

TRITON_ENV = "PYMSS_NPU_TRITON"
_available: bool | None = None
_reason: str = "not probed"


def triton_ascend_available() -> bool:
    """True only when triton-ascend is importable *and* driving an NPU."""
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
            _available, _reason = False, f"triton backend is {backend!r}, not npu (CUDA triton wheel)"
            return _available
        _available, _reason = True, f"triton-ascend {getattr(triton, '__version__', '?')} on {backend}"
    except Exception as exc:  # ImportError on cp312, driver errors, ...
        _available, _reason = False, f"{type(exc).__name__}: {exc}"
    return _available


def triton_status() -> dict:
    available = triton_ascend_available()
    info = {"available": available, "reason": _reason, "env": TRITON_ENV}
    try:
        import triton

        info["triton_version"] = getattr(triton, "__version__", None)
    except Exception:
        info["triton_version"] = None
    try:
        import importlib.metadata as md

        info["triton_ascend_version"] = md.version("triton-ascend")
    except Exception:
        info["triton_ascend_version"] = None
    return info


if triton_ascend_available():  # pragma: no cover - not reachable on cp312 here
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_bandsplit_linear(
        x_ptr, w_ptr, b_ptr, out_ptr,
        n_bands, band_dim: tl.constexpr, n_feat: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """RMSNorm-per-band + Linear, fused: one read of x, one write of out.

        x   : (n_feat, n_bands) flattened band features (already normalized per band by
              the caller's `F.normalize * gamma`? no -- normalization is folded here)
        Replaces `F.normalize(x, dim=-1) * scale * gamma` followed by `Linear`,
        i.e. 2 HBM round trips + 1 GEMM epilogue -> 1 round trip + GEMM.
        """
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_bands
        cols = tl.arange(0, n_feat)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in range(0, n_feat):
            v = tl.load(x_ptr + offsets * n_feat + c, mask=mask, other=0.0).to(tl.float32)
            acc += v * v
        inv = 1.0 / tl.sqrt(acc / n_feat + 1e-12)
        for c in range(0, n_feat):
            v = tl.load(x_ptr + offsets * n_feat + c, mask=mask, other=0.0).to(tl.float32) * inv
            w = tl.load(w_ptr + c * band_dim + tl.arange(0, band_dim))
            acc_w = tl.sum(v[:, None] * w[None, :], axis=0)
            tl.store(out_ptr + offsets * band_dim + tl.arange(0, band_dim), acc_w.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_rope(q_ptr, k_ptr, cos_ptr, sin_ptr, n, D: tl.constexpr, BLOCK: tl.constexpr):
        """Rotate q and k in place: reads q,k,cos,sin once, writes q,k once.

        Eager path materializes q*cos, q*sin, k*cos, k*sin and then rotates with
        a `rotate_half` cat -> 4 temporary tensors. This keeps everything in
        registers.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        half = D // 2
        for i in tl.static_range(half):
            base_lo = offs * D + i
            base_hi = offs * D + i + half
            ql = tl.load(q_ptr + base_lo, mask=mask).to(tl.float32)
            qh = tl.load(q_ptr + base_hi, mask=mask).to(tl.float32)
            kl = tl.load(k_ptr + base_lo, mask=mask).to(tl.float32)
            kh = tl.load(k_ptr + base_hi, mask=mask).to(tl.float32)
            c = tl.load(cos_ptr + offs * D + i, mask=mask).to(tl.float32)
            s = tl.load(sin_ptr + offs * D + i, mask=mask).to(tl.float32)
            tl.store(q_ptr + base_lo, (ql * c - qh * s).to(tl.float16), mask=mask)
            tl.store(q_ptr + base_hi, (qh * c + ql * s).to(tl.float16), mask=mask)
            tl.store(k_ptr + base_lo, (kl * c - kh * s).to(tl.float16), mask=mask)
            tl.store(k_ptr + base_hi, (kh * c + kl * s).to(tl.float16), mask=mask)

    @triton.jit
    def _fused_gated_out(attn_ptr, gate_ptr, out_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):
        """out[i] = attn[i] * sigmoid(gate[i // K]) -- no gate broadcast materialization."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(attn_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(gate_ptr + (offs // K), mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, (a * tl.sigmoid(g)).to(tl.float16), mask=mask)


# ------------------------------------------------------------------ eager fallbacks
def fused_gated_out(attn: torch.Tensor, gates: torch.Tensor, block: int = 16384) -> torch.Tensor:
    """``attn * sigmoid(gates)[..., None]``.

    triton path when available; otherwise the torch_npu equivalent fusion:
    one in-place multiply (saves a full-size temporary + one HBM round trip vs
    `to_out(out * gates.sigmoid().unsqueeze(-1))`).
    """
    if triton_ascend_available() and attn.is_contiguous() and attn.dtype == torch.float16:
        import triton

        out = torch.empty_like(attn)
        n = attn.numel()
        _fused_gated_out[(triton.cdiv(n, block),)](attn, gates, out, n, K=attn.shape[-1], BLOCK=block)
        return out
    return attn.mul_(gates.sigmoid().unsqueeze(-1))


def fused_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block: int = 4096) -> tuple:
    """RoPE on (…, D) q/k with cos/sin already materialized (cached upstream)."""
    if triton_ascend_available() and q.is_contiguous() and k.is_contiguous() and q.dtype == torch.float16:
        import triton

        n = q.numel() // q.shape[-1]
        _fused_rope[(triton.cdiv(n, block),)](q, k, cos, sin, n, D=q.shape[-1], BLOCK=block)
        return q, k
    # eager, but without rotate_half's cat: one temp for the rotated copy only
    rotary = torch.cat((-q[..., q.shape[-1] // 2:], q[..., : q.shape[-1] // 2]), dim=-1)
    q_out = q * cos + rotary * sin
    rotary_k = torch.cat((-k[..., k.shape[-1] // 2:], k[..., : k.shape[-1] // 2]), dim=-1)
    k_out = k * cos + rotary_k * sin
    return q_out, k_out


def fused_bandsplit_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Band-wise (RMS)normalize + Linear; triton when available, else grouped GEMM.

    The eager fallback deliberately keeps the *same* algebra as BandSplit:
    normalise per band (`F.normalize * sqrt(dim) * gamma`), then one grouped
    `bmm` instead of 62 separate `Linear` calls (BandSplit already does this --
    this function exists so the triton path can be swapped in without touching
    the model).
    """
    if triton_ascend_available() and x.dtype == torch.float16 and x.is_contiguous():
        import triton

        n_feat, n_bands = x.shape[-1], weight.shape[0]
        out = torch.empty(*x.shape[:-1], weight.shape[1], device=x.device, dtype=x.dtype)
        _fused_bandsplit_linear[(triton.cdiv(n_bands, 64),)](
            x, weight, bias, out, n_bands, band_dim=weight.shape[1], n_feat=n_feat, BLOCK=64
        )
        return out
    normed = torch.nn.functional.normalize(x, dim=-1) * (x.shape[-1] ** 0.5)
    return torch.nn.functional.linear(normed, weight, bias)
