"""iSTFT with an explicit adjoint (math-equivalent reverse of :func:`stft`).

``torch.istft`` / ``istft_npu``'s own backward on this device decomposes into the
reflection-pad / conv / complex-abs paths and is slow. iSTFT is linear:

    y = crop( fold( irfft(X) * w ) / fold(w^2) )

so its adjoint is

    X_grad = irfft_adjoint( w * unfold( pad(g) / fold(w^2) ) )

where ``irfft_adjoint`` is the adjoint of ``torch.fft.irfft``: an unnormalized rfft
whose DC and Nyquist bins carry n_fft instead of n_fft/2 (the one-sided doubling).

Verified against ``torch.istft``'s autograd: see ``istft_check.py``.
"""

from __future__ import annotations

import torch

_TINY = 1e-11
_EPS_CACHE: dict = {}


def _tiny_like(t: torch.Tensor) -> torch.Tensor:
    """0-dim epsilon on t's device/dtype.

    ``clamp_min(1e-11)`` with a Python float on a large tensor faults the vector
    core on this device (LocalScalarDenseNpu / MTE invalid GM), so the bound is
    materialized as a tensor.
    """
    key = (t.device.type, t.device.index, t.dtype)
    eps = _EPS_CACHE.get(key)
    if eps is None:
        eps = torch.tensor(_TINY, device=t.device, dtype=t.dtype)
        _EPS_CACHE[key] = eps
    return eps


def _pad_window(window: torch.Tensor, n_fft: int) -> torch.Tensor:
    win = window.to(dtype=torch.float32)
    if win.numel() == n_fft:
        return win
    out = win.new_zeros(n_fft)
    start = (n_fft - win.numel()) // 2
    out[start:start + win.numel()] = win
    return out


def _envelope(window: torch.Tensor, frames: int, hop: int, n_fft: int,
              batch: int) -> torch.Tensor:
    """fold(w^2): the same window-square envelope istft divides by."""
    idx = (torch.arange(frames, device=window.device)[:, None] * hop
           + torch.arange(n_fft, device=window.device)[None, :]).reshape(-1)
    wsq = (window * window)[None, :].expand(frames, n_fft).reshape(-1)
    out_len = (frames - 1) * hop + n_fft
    env = window.new_zeros(batch, out_len)
    env.index_add_(1, idx, wsq[None, :].expand(batch, -1))
    return env


def _irfft_adjoint(frames_windowed: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Adjoint of ``torch.fft.irfft``: (B, T, n_fft) real -> (B, T, n_fft//2+1) complex.

    Per-bin factor measured against ``torch.autograd.grad(irfft(z), z, g)``:
    1/n_fft on the one-sided DC and Nyquist bins, 2/n_fft on every interior bin.
    """
    freq = torch.fft.rfft(frames_windowed, dim=-1)
    scale = torch.full((freq.shape[-1],), 2.0 / n_fft, dtype=freq.real.dtype, device=freq.device)
    scale[0] = 1.0 / n_fft
    if freq.shape[-1] > 1:
        scale[-1] = 1.0 / n_fft
    return freq * scale


class Istft(torch.autograd.Function):
    """``istft`` forward, exact adjoint backward (no autograd through istft)."""

    @staticmethod
    def forward(ctx, spec: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int,
                length: int, center: bool = True) -> torch.Tensor:
        from pymss_core.npu_fft import istft_npu

        win = _pad_window(window, n_fft)
        wav = istft_npu(spec, n_fft=n_fft, hop_length=hop, window=win, length=length,
                        center=center, return_complex=False)
        frames = spec.shape[-1]
        ctx.save_for_backward(win)
        ctx.cfg = (n_fft, hop, length, frames, center, spec.shape, wav.shape)
        return wav

    @staticmethod
    def backward(ctx, grad_wav: torch.Tensor):
        win, = ctx.saved_tensors
        n_fft, hop, length, frames, center, spec_shape, wav_shape = ctx.cfg
        lead = wav_shape[:-1]
        out_len = (frames - 1) * hop + n_fft
        b = grad_wav.reshape(-1, wav_shape[-1]).shape[0]
        g = grad_wav.reshape(b, -1).float()

        env_full = _envelope(win, frames, hop, n_fft, b)
        env = env_full.clamp_min(_tiny_like(env_full))
        start = n_fft // 2 if center else 0
        padded = g.new_zeros(b, out_len)
        take = min(g.shape[-1], max(out_len - start, 0))
        padded[:, start:start + take] = g[:, :take]
        scaled = padded / env

        # adjoint of fold == strided windows
        # Materialize the strided windows first: multiplying a strided view by the
        # window in place trips a vector-core fault on this device (MTE invalid GM).
        frames_view = scaled.as_strided((b, frames, n_fft), (out_len, hop, 1)).contiguous()
        spec_grad = _irfft_adjoint(frames_view * win, n_fft)
        # keep the F<->T move on the real planes: complex transposes fall back to AI_CPU
        spec_grad = torch.view_as_complex(
            torch.view_as_real(spec_grad).transpose(-3, -2).contiguous()).reshape(spec_shape)
        return spec_grad, None, None, None, None, None


def istft_adjoint(spec: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int,
                  length: int, center: bool = True) -> torch.Tensor:
    """``istft`` whose backward is the exact adjoint (equal to the STFT)."""
    return Istft.apply(spec, window, n_fft, hop, length, center)


def split_channels_real(spec_c: torch.Tensor, channels: int) -> torch.Tensor:
    """(B, F*C, T) complex -> (B*C, F, T) complex, moving planes on the real view.

    Doing the same reshape/permute on a complex tensor costs 73 ms per call here
    (78x slower than the real-plane version) because complex permutes fall back to
    the AI_CPU.
    """
    b, fc, t = spec_c.shape
    f = fc // channels
    real = torch.view_as_real(spec_c).reshape(b, f, channels, t, 2)
    moved = real.permute(0, 2, 1, 3, 4).contiguous()
    return torch.view_as_complex(moved.reshape(b * channels, f, t, 2))


class IstftChannels(torch.autograd.Function):
    """Per-channel iSTFT with an exact adjoint, all plane moves on real views."""

    @staticmethod
    def forward(ctx, spec_c: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int,
                length: int, channels: int) -> torch.Tensor:
        from pymss_core.npu_fft import istft_npu

        win = _pad_window(window, n_fft)
        flat = split_channels_real(spec_c, channels)
        wav = istft_npu(flat, n_fft=n_fft, hop_length=hop, window=win, length=length,
                        center=True, return_complex=False)
        ctx.save_for_backward(win)
        ctx.cfg = (n_fft, hop, length, spec_c.shape, channels, wav.shape)
        return wav.reshape(spec_c.shape[0], channels, -1)

    @staticmethod
    def backward(ctx, grad_wav: torch.Tensor):
        win, = ctx.saved_tensors
        n_fft, hop, length, spec_shape, channels, wav_shape = ctx.cfg
        bc, l = grad_wav.shape[0] * grad_wav.shape[1], grad_wav.shape[-1]
        g = grad_wav.reshape(bc, l).float()
        frames = spec_shape[-1]
        out_len = (frames - 1) * hop + n_fft
        start = n_fft // 2

        env_full = _envelope(win, frames, hop, n_fft, bc)
        env = env_full.clamp_min(_tiny_like(env_full))
        padded = g.new_zeros(bc, out_len)
        take = min(l, max(out_len - start, 0))
        padded[:, start:start + take] = g[:, :take]
        scaled = (padded / env).contiguous()
        frames_view = scaled.as_strided((bc, frames, n_fft), (out_len, hop, 1)).contiguous()
        grad_flat = _irfft_adjoint(frames_view * win, n_fft)          # (B*C, T, F)
        grad_flat = torch.view_as_complex(
            torch.view_as_real(grad_flat).transpose(-3, -2).contiguous())  # (B*C, F, T)

        # invert the channel split on real planes: (B*C, F, T) -> (B, F*C, T)
        b = spec_shape[0]
        f = spec_shape[1] // channels
        back = torch.view_as_real(grad_flat).reshape(b, channels, f, spec_shape[-1], 2)
        back = back.permute(0, 2, 1, 3, 4).contiguous().reshape(*spec_shape, 2)
        return torch.view_as_complex(back), None, None, None, None, None


def istft_channels(spec_c: torch.Tensor, window: torch.Tensor, n_fft: int, hop: int,
                   length: int, channels: int) -> torch.Tensor:
    """(B, F*C, T) complex -> (B, C, L) waveform, exact adjoint, real-plane split."""
    return IstftChannels.apply(spec_c, window, n_fft, hop, length, channels)
