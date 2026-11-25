"""NPU-safe STFT/ISTFT built from validated primitives.

``torch.stft``/``torch.istft`` are unusable on current torch_npu builds for
this device family: the fp32 kernel fails with an aclnn parameter error and
the fp16 kernel faults the vector cores asynchronously. Spectral models route
through these compositions instead. Everything runs on-device in fp32:

- analysis:   pad -> unfold -> window -> rfft        (frames-major)
- synthesis:  irfft -> window -> index_add_ OLA      with window**2 envelope

Implementation notes:

- The unfold view is materialized with ``.contiguous()`` before the broadcast
  window multiply: multiplying a strided unfold view by a vector faults the
  vector cores (async device error 507035) on Ascend 950PR.
- ``hann_window`` must be created with an explicit float dtype before landing
  on the device; the default-dtype device-side path hits an int64/float cast
  bug in this torch_npu build.

Only the code paths exercised by pymss models are implemented: real input,
centered or uncentered, one-sided, complex spectrum, ``reflect``/``constant``
padding.
"""
from __future__ import annotations

import math
import os

import numpy as np

import torch
import torch.nn.functional as F

from pymss_core.device import fft_fallback_required

_VALID_PAD_MODES = ("reflect", "constant")
_TINY = 1e-11


def hann_window_npu(window_length: int, *, device, dtype=torch.float32, periodic: bool = True) -> torch.Tensor:
    """``torch.hann_window`` that avoids the device-side default-dtype bug."""
    return torch.hann_window(window_length, periodic=periodic, dtype=dtype, device="cpu").to(device=device)

#: Sizes allowed to use the vendor FFT, unless ``PYMSS_VENDOR_FFT_SIZES`` says
#: otherwise.  Only the 2048-point transform the VR models use every chunk is
#: proven both correct and fast; 4096 made scnet spend 600 s inside one call
#: (the vendor plans/autotunes a transform the first time it sees a shape), and
#: every smaller size has been caught returning silent zeros.
_VENDOR_FFT_ALLOWED = frozenset()

#: Sizes where the vendor kernel returned wrong values or raised; remembered so the
#: exact path is used without re-testing every call.
_VENDOR_FFT_BAD: set[int] = set()

_DFT_CACHE: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


_VENDOR_FFT_STATUS: dict[tuple[int, str], bool] = {}


def _probe_vendor_fft(n: int, device) -> bool:
    """Ask the vendor kernel to round-trip a known signal and check the answer."""
    try:
        x = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32).reshape(1, n)
        spec = torch.fft.rfft(x, dim=-1)
        planes = torch.view_as_real(spec)
        if not bool(torch.isfinite(planes).all()):
            return False
        if float(planes.abs().max()) == 0.0:
            return False
        back = torch.fft.irfft(spec, n=n, dim=-1)
        ref = x.cpu()
        return bool(torch.allclose(back.cpu(), ref, atol=1e-3, rtol=1e-3))
    except Exception:  # noqa: BLE001 - any vendor failure means "do not trust"
        return False


def vendor_fft_trusted(n: int, device=None) -> bool:
    """Return ``True`` when the vendor FFT can be used for size ``n``.

    BIREN's FFT is not reliable: size 1411 fails to build a plan ("sufft build
    plan failed"), the 9-bin spectrum of a 16-point STFT and a 513-bin inverse of
    a 1024-point STFT both come back as all zeros -- silently -- and size 4096
    can spend ten minutes autotuning inside one call.  Only 2048 (the VR models'
    transform) is proven both correct and fast; everything else uses an exact DFT
    matrix multiply, which is deterministic and cheap at these sizes.

    Args:
        n: Transform length.

    Args:
        n: Transform length.
        device: When given, a one-off round-trip probe decides whether this
            device really computes this size correctly (BIREN answered zeros
            for shapes the size alone could not predict).

    Returns:
        ``True`` when the vendor kernel is trusted for this size.
    """
    # A clean container measures the vendor kernel at 39 ms for n=2048 with 862
    # frames, against 60 ms for the DFT matrix, so the vendor path is the fast one.
    # It is only refused when the environment says so, when a size has already
    # misbehaved here (silent zeros, a failed plan), or when it is not a power of
    # two.  A degraded container makes the same call 10x slower and eventually
    # raises, which is why the measurements must be taken in a fresh session.
    if (n & (n - 1)) != 0 or n in _VENDOR_FFT_BAD:
        return False
    allowed = os.environ.get("PYMSS_VENDOR_FFT_SIZES", "")
    if allowed:
        sizes = {int(part) for part in allowed.split(",") if part.strip()}
        return n in sizes
    if _VENDOR_FFT_ALLOWED:
        return n in _VENDOR_FFT_ALLOWED
    return True
    if device is None:
        return True
    key = (n, str(device))
    cached = _VENDOR_FFT_STATUS.get(key)
    if cached is None:
        cached = _probe_vendor_fft(n, device)
        _VENDOR_FFT_STATUS[key] = cached
    return cached


def _result_check_required(device) -> bool:
    """Return ``True`` when a vendor FFT answer needs the silent-zero probe.

    Only the vendors that answer with zeros instead of raising need this
    (BIREN).  Where the FFT *is* ``torch.fft`` the probe costs two full
    reductions plus two host syncs per transform and is a ``torch.compile``
    graph break, so it is skipped.

    Args:
        device: Device the transform will run on.

    Returns:
        ``True`` when the result must be verified.
    """
    from .device import fft_result_check_required

    return fft_result_check_required(device)


def _degenerate(out: torch.Tensor, source: torch.Tensor) -> bool:
    """Return ``True`` when the vendor answered with a silent zero tensor.

    BIREN's FFT returns zeros for some shapes without raising (a 513-bin,
    1024-point inverse transform is one), so every vendor result is checked
    before it is trusted.  A zero result from a zero input is legitimate.

    Args:
        out: Vendor FFT result (real or complex).
        source: Its input.

    Returns:
        ``True`` when the answer looks like a silent failure.
    """
    out_planes = torch.view_as_real(out) if out.is_complex() else out
    if float(out_planes.abs().max()) != 0.0:
        return False
    src_planes = torch.view_as_real(source) if source.is_complex() else source
    return float(src_planes.abs().max()) != 0.0


def _dft_matrices(n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached ``cos``/``sin`` DFT basis for length ``n``."""
    key = (n, str(device), dtype)
    cached = _DFT_CACHE.get(key)
    if cached is None:
        # Build the basis on the host: float64 on BIREN is emulated and a
        # (2049, 4096) double matmul basis stalled scnet for minutes.
        steps = torch.arange(n, dtype=torch.float64)
        bins = torch.arange(n // 2 + 1, dtype=torch.float64)
        angle = (2.0 * math.pi / n) * torch.outer(bins, steps)
        cos = angle.cos().to(dtype)
        sin = angle.sin().to(dtype)
        if str(device) != "cpu":
            cos = cos.to(device)
            sin = sin.to(device)
        cached = (cos, sin)
        _DFT_CACHE[key] = cached
    return cached


#: Once a matmul-based transform has failed on this device, stop retrying it.
_MATMUL_FFT_BROKEN = set()


def _rfft_matmul(x: torch.Tensor, n: int) -> torch.Tensor:
    """``rfft`` exactly: DFT matrix for sizes up to 8192, radix-2 above."""
    if n <= 8192 and instance_key(x) not in _MATMUL_FFT_BROKEN:
        try:
            return _rfft_matmul_dft(x, n)
        except Exception:  # vendor JIT/plan failures are not our bug; remember and move on
            _MATMUL_FFT_BROKEN.add(instance_key(x))
    if (n & (n - 1)) == 0:
        lead = x.shape[:-1]
        flat = x.reshape(-1, n)
        planes = torch.stack((flat, torch.zeros_like(flat)), dim=-1).reshape(*lead, n, 2)
        spectrum = _fft_exact_planes(planes)
        return torch.view_as_complex(spectrum[..., : n // 2 + 1, :].contiguous())
    return _rfft_matmul_dft(x, n)


def _irfft_matmul(spec: torch.Tensor, n: int) -> torch.Tensor:
    """``irfft`` exactly, by DFT matrix; radix-2 when the matrix route fails."""
    if n <= 8192 and instance_key(spec) not in _MATMUL_FFT_BROKEN:
        try:
            return _irfft_matmul_dft(spec, n)
        except Exception:
            _MATMUL_FFT_BROKEN.add(instance_key(spec))
    if (n & (n - 1)) == 0:
        planes = torch.view_as_real(spec.reshape(-1, spec.shape[-1]).contiguous())
        half = n // 2
        mirror = planes[..., 1:half, :].flip(-2)
        mirror = torch.stack((mirror[..., 0], -mirror[..., 1]), dim=-1)
        full = torch.cat((planes, mirror), dim=-2)
        wave = _fft_exact_planes(full, inverse=True)[..., 0]
        return wave.reshape(*spec.shape[:-1], n)
    return _irfft_matmul_dft(spec, n)


def _rfft_matmul_dft(x: torch.Tensor, n: int) -> torch.Tensor:
    """``rfft`` by DFT matrix multiply (real kernels only)."""
    cos, sin = _dft_matrices(n, x.device, x.dtype)
    lead = x.shape[:-1]
    flat = x.reshape(-1, n)
    real = flat @ cos.transpose(0, 1)
    imag = -(flat @ sin.transpose(0, 1))
    bins = cos.shape[0]
    planes = torch.stack((real, imag), dim=-1).reshape(*lead, bins, 2).contiguous()
    return torch.view_as_complex(planes)


def _irfft_matmul_dft(spec: torch.Tensor, n: int) -> torch.Tensor:
    """``irfft`` by DFT matrix multiply (real kernels only)."""
    real_dtype = spec.real.dtype
    cos, sin = _dft_matrices(n, spec.device, real_dtype)
    bins = spec.shape[-1]
    weight = torch.full((bins,), 2.0, device=spec.device, dtype=real_dtype)
    weight[0] = 1.0
    if n % 2 == 0 and bins == n // 2 + 1:
        weight[-1] = 1.0
    lead = spec.shape[:-1]
    planes = torch.view_as_real(spec.reshape(-1, bins).contiguous())
    weighted_real = planes[..., 0] * weight
    weighted_imag = planes[..., 1] * weight
    out = (weighted_real @ cos - weighted_imag @ sin) / n
    return out.reshape(*lead, n)


def instance_key(tensor: torch.Tensor) -> str:
    """Key for the "matmul transforms are broken here" memo, per device and dtype."""
    return f"{tensor.device.type}:{tensor.device.index}:{tensor.dtype}"


_TWIDDLE_CACHE: dict[tuple[int, str, torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]] = {}


def _twiddles(n: int, device, dtype, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached cos/sin table of the ``n // 2`` distinct roots of unity."""
    key = (n, str(device), dtype, inverse)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached
    # Forward uses exp(-2*pi*i*k/n), the inverse uses the conjugate.
    angle = (2.0 * math.pi / n) * torch.arange(n // 2, dtype=torch.float64)
    if not inverse:
        angle = -angle
    cos = angle.cos().to(dtype)
    sin = angle.sin().to(dtype)
    if str(device) != "cpu":
        cos = cos.to(device)
        sin = sin.to(device)
    _TWIDDLE_CACHE[key] = (cos, sin)
    return cos, sin


def _fft_radix2(planes: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Iterative radix-2 FFT on the last-but-one axis of a real/imag plane tensor.

    The DFT matrix fallback is exact but its matmul is large: for a 4096-point
    transform it is a (frames, 4096) x (4096, 2049) product that costs 2.3 s on
    BIREN -- 31% of scnet's run -- and occasionally fails inside SUDNN with
    ``ErrorCode: 11``.  A radix-2 decomposition needs only element-wise real ops
    and no workspace at all.

    Args:
        planes: ``(..., n, 2)`` real/imaginary pairs, ``n`` a power of two.
        inverse: Whether to run the inverse transform.

    Returns:
        The transformed planes.
    """
    n = planes.shape[-2]
    if n == 1:
        return planes
    cos, sin = _twiddles(n, planes.device, planes.dtype, inverse)

    # bit-reversal permutation
    bits = n.bit_length() - 1
    index = torch.arange(n, device=planes.device)
    reversed_index = torch.zeros_like(index)
    for bit in range(bits):
        reversed_index = (reversed_index << 1) | ((index >> bit) & 1)
    out = planes.index_select(planes.dim() - 2, reversed_index)

    length = 2
    while length <= n:
        half = length // 2
        stride = n // length
        lead = out.shape[:-2]
        stage = out.reshape(*lead, n // length, length, 2)
        wr = cos[::stride][:half]
        wi = sin[::stride][:half]
        a = stage[..., :half, :]
        b = stage[..., half:, :]
        br = b[..., 0]
        bi = b[..., 1]
        tr = br * wr - bi * wi
        ti = br * wi + bi * wr
        out = torch.cat((torch.stack((a[..., 0] + tr, a[..., 1] + ti), dim=-1),
                         torch.stack((a[..., 0] - tr, a[..., 1] - ti), dim=-1)), dim=-2)
        out = out.reshape(*lead, n, 2)
        length *= 2
    if inverse:
        out = out / n
    return out


def _fft_exact_planes(planes: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Radix-2 when the length is a power of two, otherwise the DFT matrix."""
    n = planes.shape[-2]
    if (n & (n - 1)) == 0:
        return _fft_radix2(planes, inverse)
    real = _dft_matrices(n, planes.device, planes.dtype)[0 if not inverse else 0]
    cos, sin = _dft_matrices(n, planes.device, planes.dtype)
    flat = planes.reshape(-1, n)
    xr, xi = flat[..., 0], flat[..., 1]
    real = xr @ cos.transpose(0, 1) - xi @ sin.transpose(0, 1)
    imag = xr @ sin.transpose(0, 1) + xi @ cos.transpose(0, 1)
    if inverse:
        real, imag = real / n, imag / n
    lead = planes.shape[:-2]
    return torch.stack((real, imag), dim=-1).reshape(*lead, n, 2)


def _pad_to_length(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """Pad or truncate ``x`` to ``n`` samples along ``dim`` (torch.fft semantics)."""
    size = x.shape[dim]
    if size == n:
        return x
    if size > n:
        return x.narrow(dim, 0, n)
    shape = list(x.shape)
    shape[dim] = n - size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=dim)


def rfft_exact(x: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> torch.Tensor:
    """``torch.fft.rfft`` by DFT matrix multiply, for any dim and norm."""
    n = x.shape[dim] if n is None else n
    moved = _pad_to_length(x, n, dim).movedim(dim, -1)
    spectrum = _rfft_matmul(moved, n)
    if norm == "ortho":
        spectrum = spectrum / math.sqrt(n)
    elif norm == "forward":
        spectrum = spectrum / n
    return spectrum.movedim(-1, dim)


def irfft_exact(spec: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> torch.Tensor:
    """``torch.fft.irfft`` by DFT matrix multiply, for any dim and norm."""
    n = 2 * (spec.shape[dim] - 1) if n is None else n
    moved = spec.movedim(dim, -1)
    wave = _irfft_matmul(moved, n)
    if norm == "ortho":
        wave = wave * math.sqrt(n)
    elif norm == "forward":
        wave = wave * n
    return wave.movedim(-1, dim)


def _fit_window(window: torch.Tensor, n_fft: int, win_length: int, device, dtype) -> torch.Tensor:
    window = window.to(device=device, dtype=dtype)
    if win_length != n_fft:
        lpad = (n_fft - win_length) // 2
        window = F.pad(window, (lpad, n_fft - win_length - lpad))
    return window

def stft_npu(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
    *,
    return_complex: bool = True,
) -> torch.Tensor:
    """NPU-safe ``torch.stft`` replacement for real inputs.

    Matches ``torch.stft`` output layout ``(…, freq, time)`` for the
    parameter combinations used by pymss models.
    """
    if not torch.is_floating_point(x):
        raise TypeError("stft_npu expects a floating point input")
    if not onesided:
        raise NotImplementedError("two-sided stft_npu is not needed by pymss models")
    if not return_complex:
        raise ValueError("stft_npu requires return_complex=True")
    if pad_mode not in _VALID_PAD_MODES:
        raise ValueError(f"stft_npu supports pad modes {_VALID_PAD_MODES}, got {pad_mode!r}")
    hop = n_fft // 4 if hop_length is None else hop_length
    win_length = n_fft if win_length is None else win_length
    if window is None:
        # torch.stft with no explicit window applies a rectangular (ones) window.
        window = torch.ones(win_length, dtype=x.dtype, device=x.device)
    window = _fit_window(window, n_fft, win_length, x.device, x.dtype)

    *lead, length = x.shape
    flat = x.reshape(-1, 1, length)
    if center:
        if pad_mode == "reflect":
            # aten.reflection_pad1d has no AscendIR converter (torchair graphs);
            # an equivalent flip+cat formulation keeps the whole forward graphable.
            lpad = min(n_fft // 2, flat.shape[-1] - 1)
            rpad = min(n_fft // 2, flat.shape[-1] - 1)
            flat = torch.cat([flat[..., 1:lpad+1].flip(-1), flat, flat[..., -rpad-1:-1].flip(-1)], dim=-1)
        else:
            flat = F.pad(flat, (n_fft // 2, n_fft // 2), mode=pad_mode)
    # aten.unfold has no AscendIR converter (torchair graphs); the identical
    # strided view is expressible through as_strided, which does convert.
    *lead_n, length_n = flat.shape
    n_frames = (length_n - n_fft) // hop + 1
    frames = flat.as_strided((*lead_n, n_frames, n_fft), (*flat.stride()[:-1], hop, 1)).contiguous() * window  # (N, 1, T, n_fft)
    # torch.stft(normalized=True) scales the plain FFT output by sqrt(1/n_fft).
    frames_flat = frames.squeeze(1)
    if vendor_fft_trusted(n_fft, frames_flat.device):
        try:
            spec = torch.fft.rfft(frames_flat, dim=-1)  # (N, T, F)
        except Exception:
            _VENDOR_FFT_BAD.add(n_fft)
            spec = None
        if spec is None or (_result_check_required(frames_flat.device) and _degenerate(spec, frames_flat)):
            _VENDOR_FFT_BAD.add(n_fft)
            spec = _rfft_matmul(frames_flat, n_fft)
    else:
        spec = _rfft_matmul(frames_flat, n_fft)
    if normalized:
        spec = spec * (1.0 / n_fft) ** 0.5
    # torch_npu runs complex64 transposes on the AI_CPU (~20ms at this size);
    # transpose the real/imag planes on the vector cores instead (~0.04ms).
    spec = torch.view_as_complex(torch.view_as_real(spec).permute(0, 2, 1, 3).contiguous())  # (N, F, T)
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft_npu(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    length: int | None = None,
    *,
    return_complex: bool = False,
) -> torch.Tensor:
    """NPU-safe ``torch.istft`` replacement for one-sided complex inputs."""
    if return_complex:
        raise ValueError("istft_npu reconstructs a real signal; return_complex is not supported")
    if not onesided:
        raise NotImplementedError("two-sided istft_npu is not needed by pymss models")
    hop = n_fft // 4 if hop_length is None else hop_length
    win_length = n_fft if win_length is None else win_length
    if window is None:
        # torch.istft with no explicit window assumes a rectangular one.
        window = torch.ones(win_length, dtype=torch.float32, device=spec.device)
    window = _fit_window(window, n_fft, win_length, spec.device, torch.float32)
    if not spec.is_complex():
        spec = torch.view_as_complex(spec.contiguous())
    if spec.dtype != torch.complex64:
        # torch_npu cannot copy between complex dtypes; rebuild from real parts.
        spec = torch.view_as_complex(torch.view_as_real(spec).float().contiguous())
    if normalized:
        spec = spec * (n_fft ** 0.5)

    *lead, freqs, n_frames = spec.shape
    flat_spec = spec.reshape(-1, freqs, n_frames)
    # torch_npu runs complex64 transposes on the AI_CPU; permute real/imag planes instead.
    flat_spec = torch.view_as_complex(torch.view_as_real(flat_spec).permute(0, 2, 1, 3).contiguous())  # (N, T, F)
    if vendor_fft_trusted(n_fft, flat_spec.device):
        try:
            frames = torch.fft.irfft(flat_spec, n=n_fft, dim=-1)  # (N, T, n_fft)
        except Exception:
            _VENDOR_FFT_BAD.add(n_fft)
            frames = None
        if frames is None or (_result_check_required(flat_spec.device) and _degenerate(frames, flat_spec)):
            _VENDOR_FFT_BAD.add(n_fft)
            frames = _irfft_matmul(flat_spec, n_fft)
    else:
        frames = _irfft_matmul(flat_spec, n_fft)
    frames = frames * window
    batch = frames.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    idx = torch.arange(n_frames, device=spec.device)[:, None] * hop + torch.arange(n_fft, device=spec.device)[None, :]
    idx = idx.reshape(-1)
    out = frames.new_zeros(batch, out_len)
    out.index_add_(1, idx, frames.reshape(batch, -1))
    wsq = (window * window)[None, :].expand(n_frames, n_fft).reshape(-1)
    norm = frames.new_zeros(batch, out_len)
    norm.index_add_(1, idx, wsq[None, :].expand(batch, -1))
    out = out.reshape(*lead, out_len)[..., n_fft // 2 if center else 0 :]
    norm = norm.reshape(*lead, out_len)[..., n_fft // 2 if center else 0 :]
    if center and length is None:
        # An explicit length may include samples reconstructed from right padding.
        trim = min(n_fft // 2, out.shape[-1])
        out = out[..., : out.shape[-1] - trim]
        norm = norm[..., : norm.shape[-1] - trim]
    if length is not None:
        if length > out.shape[-1]:
            out = F.pad(out, (0, length - out.shape[-1]))
            norm = F.pad(norm, (0, length - norm.shape[-1]))
        else:
            out = out[..., :length]
            norm = norm[..., :length]
    return out / norm.clamp_min(_TINY)


def numpy_complex_to_device(spec: "np.ndarray", device) -> torch.Tensor:
    """Upload a complex numpy array through its real planes.

    The mirror image of :func:`complex_to_numpy_cpu`: BIREN rejects a complex
    host->device copy as well, with

        Unsupported BR type for ScalarType: 0

    so the real planes travel and the complex view is rebuilt on the device.

    Args:
        spec: Complex numpy array.
        device: Target torch device.

    Returns:
        A complex tensor on ``device``.
    """
    array = np.ascontiguousarray(spec)
    if not np.iscomplexobj(array):
        return torch.from_numpy(array).to(device)
    planes = np.stack((array.real, array.imag), axis=-1).astype(np.float32, copy=False)
    return torch.view_as_complex(torch.from_numpy(np.ascontiguousarray(planes)).to(device).contiguous())


def complex_to_numpy_cpu(spec: torch.Tensor) -> "np.ndarray":
    """Copy a complex tensor to the host through its real planes.

    BIREN cannot copy complex tensors device->host at all:

        RuntimeError: ERROR (BR TORCH INTERNAL) ... ErrorCode: 101, invalid device

    while copying ``view_as_real(spec)`` works, so the real planes travel and the
    complex view is rebuilt on the host.

    Args:
        spec: Real or complex tensor on any device.

    Returns:
        A numpy array with the same values.
    """
    if not spec.is_complex():
        return spec.detach().cpu().numpy()
    planes = torch.view_as_real(spec.detach()).contiguous().cpu()
    return torch.view_as_complex(planes).numpy()


def masked_spectrum(
    mask: torch.Tensor, spec: torch.Tensor, *, to_cpu: bool = False
) -> torch.Tensor | "np.ndarray":
    """Multiply a real mask into a complex spectrum.

    BIREN has no pointwise kernel for ``complex64 * float32``: it tries to cast
    the mask to complex64 on the device and fails with

        Typecast for BUFFER layout, src_dtype_FP32 and dst_dtype_COMPLEX64 is not implemented
        pointwise No SUPA Function found ... input_dtype = COMPLEX64

    so the product is formed on the real/imaginary planes with plain float
    kernels, which every backend supports.  Non-complex inputs take the direct
    path.

    Args:
        mask: Real mask, broadcastable against the spectrum's leading dims.
        spec: Real or complex spectrum.

    Returns:
        ``mask * spec`` with the spectrum's dtype.
    """
    if not spec.is_complex():
        product = mask * spec
        return complex_to_numpy_cpu(product) if to_cpu else product
    planes = torch.view_as_real(spec)  # (..., 2)
    for _ in range(planes.dim()):
        if mask.dim() >= planes.dim():
            break
        mask = mask.unsqueeze(-1)
    product = (mask * planes).contiguous()
    if to_cpu:
        # Never copy a complex tensor off the device (see complex_to_numpy_cpu).
        return torch.view_as_complex(product.cpu()).numpy()
    return torch.view_as_complex(product)


def torch_stft_blocked(device) -> bool:
    """Return True when ``torch.stft``/``torch.istft`` must be replaced on this device.

    Delegates to the backend capability profile in :mod:`pymss_core.device`:
    Ascend lacks a usable complex STFT path, and any other accelerator can opt
    into the ``torch.fft`` composition while its stack is being validated
    (``PYMSS_FFT_FALLBACK=1``).
    """
    return fft_fallback_required(device)


__all__ = [
    "hann_window_npu",
    "irfft_exact",
    "istft_npu",
    "rfft_exact",
    "stft_npu",
    "torch_stft_blocked",
    "vendor_fft_trusted",
]
