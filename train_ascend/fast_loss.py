"""Math-equivalent STFT front end whose backward does not go through the slow
Ascend `aten::stft` backward (measured ~480 ms per call).

The analysis STFT is linear: frames -> window -> rfft. Its backward is the adjoint,
which here is composed of ops the NPU executes well:

    grad -> (sqrt/pow elementwise) -> rfft adjoint (16 ms, FFT) -> *window
         -> fold (overlap-add) -> reflect-pad adjoint

`torch.Tensor.unfold` cannot be used on Ascend (its backward, aclnnUnfoldGrad, is
unimplemented), so framing is a custom Function whose backward is exactly the
overlap-add (`F.fold`), and framing forward is a zero-copy `as_strided` view.

Everything is verified against `torch.stft` in `fast_loss_check.py`:
forward rel_rms <= 1e-6 and gradient rel_rms <= 1e-6.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12
_EPS_CACHE: dict = {}


def _eps_like(t: torch.Tensor) -> torch.Tensor:
    """0-dim epsilon on the same device/dtype.

    Adding a *Python float* to a large tensor hits an Ascend kernel bug here
    (scalar dense path -> vector core exception), so the epsilon is materialized
    as a tensor instead. Same value, safer path.
    """
    key = (t.device.type, t.device.index, t.dtype)
    eps = _EPS_CACHE.get(key)
    if eps is None:
        eps = torch.tensor(EPS, device=t.device, dtype=t.dtype)
        _EPS_CACHE[key] = eps
    return eps


class _Frame(torch.autograd.Function):
    """Framing (as_strided, zero copy) with the exact adjoint (overlap-add)."""

    @staticmethod
    def forward(ctx, padded: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        b, lp = padded.shape
        frames = 1 + (lp - n_fft) // hop
        ctx.dims = (b, lp, frames, n_fft, hop)
        return padded.as_strided((b, frames, n_fft), (padded.stride(0), hop, 1))

    @staticmethod
    def backward(ctx, grad_frames: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:
        b, lp, frames, n_fft, hop = ctx.dims
        covered = (frames - 1) * hop + n_fft
        folded = F.fold(
            grad_frames.transpose(1, 2),
            output_size=(1, covered),
            kernel_size=(1, n_fft),
            stride=(1, hop),
        ).reshape(b, covered)
        if covered == lp:
            return folded, None, None
        out = torch.zeros(b, lp, dtype=folded.dtype, device=folded.device)
        out[:, :covered] = folded
        return out, None, None


def stft_magnitude(
    x: torch.Tensor,
    window: torch.Tensor,
    n_fft: int,
    hop: int,
    center: bool = True,
) -> torch.Tensor:
    """|STFT(x)| with the same framing/padding semantics as ``torch.stft``.

    Args:
        x: (N, L) real signal.
        window: (n_fft,) analysis window.
        n_fft, hop: STFT parameters; ``center=True`` uses reflect padding of n_fft//2.

    Returns:
        (N, F, T) magnitude, matching ``torch.stft(..., return_complex=True).abs()``.
    """
    if center:
        pad = n_fft // 2
        xp = F.pad(x, (pad, pad), mode="reflect")
    else:
        xp = x
    frames = _Frame.apply(xp, n_fft, hop)              # (N, T, n_fft)
    spec = torch.fft.rfft(frames * window, dim=-1)     # (N, T, F) complex
    real_imag = torch.view_as_real(spec)               # (N, T, F, 2)
    # sqrt(re^2+im^2) instead of spec.abs(): complex abs is a slow composite on Ascend.
    # +EPS only floors d(sqrt)/dx at exactly-zero bins (otherwise the grad is inf -> NaN).
    mag = torch.sqrt(real_imag.pow(2).sum(-1) + _eps_like(real_imag))
    return mag.transpose(-1, -2)                       # (N, F, T)


def stft_forward_complex(x: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Plain forward-only STFT (same framing as torch.stft), for reference/tests."""
    pad = n_fft // 2
    xp = F.pad(x, (pad, pad), mode="reflect")
    frames = _Frame.apply(xp, n_fft, hop)
    return torch.fft.rfft(frames * window, dim=-1).transpose(-1, -2)


def available() -> bool:
    """The fast path needs rfft and fold on the device."""
    return True


def _eps_like_tensor(t: torch.Tensor) -> torch.Tensor:
    return _eps_like(t)


def stft_adjoint(grad_spec: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int,
                 length: int) -> torch.Tensor:
    """Adjoint of the analysis STFT: complex spectrum grad -> signal grad.

    ``torch.stft(x) == rfft(window * frames(pad(x)))``, so the adjoint is
    ``fold(adjoint_rfft(grad) * window)`` followed by the reflect-pad adjoint.
    The rfft adjoint is taken from PyTorch's own rfft backward (a single FFT), the
    overlap-add is `F.fold`, and the pad adjoint is the pad op's own backward.
    """
    pad = n_fft // 2
    frames_len = length + 2 * pad
    frames = 1 + (frames_len - n_fft) // hop
    b = grad_spec.shape[0]
    # grad_spec: (B, F, T) -> (B, T, F). Move the planes on the REAL view: a complex
    # transpose+copy costs ~150 ms here (AI_CPU fallback), the real permute ~1 ms.
    g = torch.view_as_complex(
        torch.view_as_real(grad_spec).permute(0, 2, 1, 3).contiguous())
    with torch.enable_grad():
        z = torch.zeros(b, frames, n_fft, dtype=window.dtype, device=window.device).requires_grad_(True)
        spec = torch.fft.rfft(z, dim=-1)
        gf, = torch.autograd.grad(spec, z, grad_outputs=g)
    covered = (frames - 1) * hop + n_fft
    folded = F.fold((gf * window).transpose(1, 2), output_size=(1, covered),
                    kernel_size=(1, n_fft), stride=(1, hop)).reshape(b, covered)
    padded = torch.zeros(b, frames_len, dtype=folded.dtype, device=folded.device)
    padded[:, :covered] = folded
    with torch.enable_grad():
        zx = torch.zeros(b, length, dtype=window.dtype, device=window.device).requires_grad_(True)
        gx, = torch.autograd.grad(F.pad(zx, (pad, pad), mode="reflect"), zx, grad_outputs=padded)
    return gx


class StftMagnitude(torch.autograd.Function):
    """``|torch.stft(x)|`` with a cheap, exact adjoint.

    forward: torch.stft (fast) + magnitude from the real view.
    backward: magnitude -> spectrum (elementwise, via autograd), then `stft_adjoint`.
    The expensive `aten::stft` backward is never executed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True,
                          return_complex=True)
        sr = torch.view_as_real(spec)                      # kept for the backward
        mag = torch.sqrt(sr.pow(2).sum(-1) + _eps_like(sr))
        # Save the spectrum instead of x: the backward needs its phase, and re-running
        # torch.stft there cost another 13-30 ms per resolution.
        ctx.save_for_backward(window, sr)
        ctx.cfg = (n_fft, hop, x.shape[-1])
        return mag  # (B, F, T), same layout as torch.stft(...).abs()

    @staticmethod
    def backward(ctx, grad_mag: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        window, sr = ctx.saved_tensors
        n_fft, hop, length = ctx.cfg
        if grad_mag.dtype != sr.dtype:
            grad_mag = grad_mag.to(sr.dtype)
        # dL/dz = g * z / |z| (verified against autograd to 6e-17). Doing it by hand
        # avoids the complex-abs backward, which costs ~250 ms on this device.
        mag = torch.sqrt(sr.pow(2).sum(-1) + _eps_like(sr)).clamp_min(_eps_like(sr))
        gz = grad_mag.unsqueeze(-1) * sr / mag.unsqueeze(-1)
        gspec = torch.view_as_complex(gz.contiguous())
        gx = stft_adjoint(gspec, window, n_fft, hop, length)
        return gx, None, None, None


def stft_mag_fast(x: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Entry point used by the loss: ``|STFT(x)|`` via the cheap-adjoint Function."""
    return StftMagnitude.apply(x, window, n_fft, hop)
