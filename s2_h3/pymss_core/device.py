"""Accelerator backend profiles.

pymss runs on several PyTorch accelerators, and they do not agree on how they
present themselves to ``torch``:

* **NVIDIA / AMD ROCm**  -- ``device.type == "cuda"``.
* **Iluvatar CoreX / MetaX MACA / Hygon DCU** -- vendor PyTorch builds that
  *impersonate* CUDA: ``device.type`` is ``"cuda"`` and ``torch.cuda.is_available()``
  is ``True``.  ``torch.cuda.get_device_name()`` still reports the real vendor.
* **Huawei Ascend** (``torch_npu``) -- private device type ``"npu"``.
* **Enflame GCU** (``torch_gcu``) -- private device type ``"gcu"``.
* **Cambricon MLU** (``torch_mlu``) -- private device type ``"mlu"``.
* **Biren** (``torch_br``) -- private device type ``"supa"`` on newer stacks,
  CUDA-impersonating on older ones.
* **Google TPU** (``torch_tpu``) -- private device type ``"tpu"`` (torch_tpu
  renames ``PrivateUse1``), and ``torch.tpu.set_device`` is rejected as an
  internal API; device selection goes through ``torch.accelerator``.

Hardcoding ``device.type == "npu"``-style checks ties the code to a single
vendor, so this module centralises detection and exposes capability flags
instead.  Every flag can be forced from the environment while a new vendor is
being brought up::

    PYMSS_DEVICE_BACKEND=biren   # force vendor detection
    PYMSS_FFT_FALLBACK=1         # force the complex-safe STFT/ISTFT path
    PYMSS_RNN_FALLBACK=1         # force the CPU fallback for LSTM/GRU/RNN
    PYMSS_AMP=0                  # disable autocast
    PYMSS_MATMUL_PRECISION=highest  # override the fp32 matmul precision (TPU)

Capability defaults are deliberately conservative only where a vendor is known
to be incomplete (Ascend).  Vendors that are still being validated default to
the native path and can be switched to the fallback path per-run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

NVIDIA = "nvidia"
ROCM = "rocm"
HYGON = "hygon"
ILUVATAR = "iluvatar"
METAX = "metax"
ENFLAME = "enflame"
BIREN = "biren"
ASCEND = "ascend"
MLU = "mlu"
MPS = "mps"
XPU = "xpu"
TPU = "tpu"
CPU = "cpu"

#: Vendors a user may pass as ``--device``.  ``auto`` is handled by callers.
DEVICE_ALIASES: tuple[str, ...] = (
    "cpu",
    "cuda",
    "rocm",
    "mps",
    "mlx",
    "npu",
    "xpu",
    "gcu",
    "mlu",
    "supa",
    "biren",
    "iluvatar",
    "metax",
    "enflame",
    "hygon",
    "tpu",
)

#: ``--device`` alias -> vendor key.
ALIAS_TO_VENDOR: dict[str, str] = {
    "cuda": NVIDIA,
    "rocm": ROCM,
    "mps": MPS,
    "mlx": MPS,
    "npu": ASCEND,
    "xpu": XPU,
    "gcu": ENFLAME,
    "mlu": MLU,
    "supa": BIREN,
    "biren": BIREN,
    "iluvatar": ILUVATAR,
    "metax": METAX,
    "enflame": ENFLAME,
    "hygon": HYGON,
    "tpu": TPU,
    "cpu": CPU,
}


@dataclass(frozen=True)
class DeviceProfile:
    """Capabilities of the accelerator pymss is currently running on.

    Attributes:
        vendor: Canonical vendor key (see the module level constants).
        device_type: torch device type string to build devices with.
        plugin: Import name of the vendor PyTorch extension, if one is required
            before the backend can be used (``torch_npu``, ``torch_gcu``,
            ``torch_mlu``, ``torch_br``).
        complex_fft: ``torch.stft``/``torch.istft`` handle complex tensors on the
            device.  When ``False`` the ``torch.fft`` based fallback is used.
        rnn: ``nn.LSTM``/``nn.GRU``/``nn.RNN`` run natively on the device.  When
            ``False`` recurrent layers bounce through CPU.
        rnn_blocked: individual cells (``"lstm"``/``"gru"``/``"rnn"``) that must
            bounce through CPU although the rest of the family is native.
        dilated_conv: dilated convolutions can be planned by the vendor kernel.
        dilated_conv_allowed: symmetric dilations the kernel still accepts when
            ``dilated_conv`` is ``False`` (BIREN's SUDNN: ``(1,1)/(2,2)/(4,4)``
            per its own error text, but only ``(1,1)``/``(2,2)`` survived our
            measurements, so only those stay native).
        amp: ``torch.amp.autocast`` is usable for the device type.
        amp_dtype: dtype autocast should use; bfloat16 where the vendor has no
            float16 matmul (BIREN br10x).
        channels_last: ``channels_last`` memory format is beneficial/supported.
        multi_device: more than one device index is usable.
        group_norm_contiguous: ``F.group_norm`` rejects a non-contiguous input
            (TPU's GroupNorm reaches for ``view()`` internally); copy first.
        fold_exact: ``F.fold``/``col2im`` accumulate in reduced precision on the
            device; replace them with the exact ``scatter_add_`` composition.
        fft_result_check: every FFT result must be verified for a silent zero
            answer (BIREN returns zeros for some shapes without raising).
            Devices whose FFT is plain ``torch.fft`` skip the check, which costs
            two full reductions plus two host syncs per STFT and breaks
            ``torch.compile`` (``float(tensor)`` is a graph break).
        matmul_precision: value to install with
            ``torch.set_float32_matmul_precision`` when the backend is prepared.
    """

    vendor: str
    device_type: str
    plugin: str | None = None
    complex_fft: bool = True
    rnn: bool = True
    rnn_blocked: frozenset[str] = frozenset()
    dilated_conv: bool = True
    complex_mul: bool = True
    complex_reshape: bool = True
    window_on_device: bool = True
    glu_shim: bool = False
    group_norm_shim: bool = False
    interpolate_shim: bool = False
    grouped_conv1d: bool = False
    complex_cat: bool = False
    fft_shim: bool = False
    fft_dtype_shim: bool = False
    as_strided_shim: bool = False
    unfold_shim: bool = False
    std_shim: bool = False
    cpu_shim: bool = False
    sdpa_shim: bool = False
    rope_shim: bool = False
    scatter_add: bool = True
    dilated_conv_allowed: frozenset[tuple[int, int]] = frozenset()
    amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    channels_last: bool = True
    multi_device: bool = True
    group_norm_contiguous: bool = False
    fold_exact: bool = False
    fft_result_check: bool = True
    matmul_precision: str | None = None

    @property
    def is_accelerator(self) -> bool:
        return self.device_type not in ("cpu", "mps")

    @property
    def needs_fft_fallback(self) -> bool:
        return not self.complex_fft

    @property
    def needs_rnn_fallback(self) -> bool:
        return not self.rnn

    @property
    def blocked_rnn_ops(self) -> frozenset[str]:
        """Recurrent cells that must bounce through CPU even when ``rnn`` is set."""
        return frozenset() if not self.rnn else self.rnn_blocked

    @property
    def supports_amp(self) -> bool:
        return self.amp

    @property
    def supports_data_parallel(self) -> bool:
        return self.multi_device and self.device_type == "cuda"


#: Baseline profiles.  ``complex_fft``/``rnn`` are ``False`` only where the
#: vendor stack is known to be missing the operator.
_PROFILES: dict[str, DeviceProfile] = {
    NVIDIA: DeviceProfile(NVIDIA, "cuda"),
    ROCM: DeviceProfile(ROCM, "cuda"),
    HYGON: DeviceProfile(HYGON, "cuda", fft_dtype_shim=True),
    ILUVATAR: DeviceProfile(ILUVATAR, "cuda", fft_dtype_shim=True),
    METAX: DeviceProfile(METAX, "cuda", fft_dtype_shim=True),
    ENFLAME: DeviceProfile(ENFLAME, "xla", plugin="torch_gcu", complex_fft=False, as_strided_shim=True,
        complex_mul=False, complex_reshape=False, complex_cat=True, unfold_shim=True),
    # Measured on BIREN106M + torch 2.7.0+cpu / torch-br 1.7.0.20700 (SUPA 1.9.0),
    # 31.75 GiB, device type ``supa``:
    # * ``stft(return_complex=True)`` raises "does not support return_complex=True
    #   now"; the rfft-based composition matches torch to 2.3e-05 (spec) / 9.5e-07
    #   (waveform), so ``complex_fft=False``.
    # * ``hann_window(..., device="supa")`` dies with a SUPA kernel launch error;
    #   windows must be built on CPU, which ``hann_window_npu`` already does.
    # * The fused ``_VF.lstm`` is silently WRONG (max |err| 0.18-0.47 against the
    #   CPU reference in every layer/bidirectional/batch_first combination) while
    #   a manual unrolled cell built from matmul/sigmoid/tanh matches to 2e-06.
    #   GRU additionally raises for a missing ``aten::_thnn_fused_gru_cell``.
    #   Recurrent cells therefore always bounce through CPU on this stack.
    # * float16 is unusable: br10x matmul takes bf16/fp32/s8/u8 only and
    #   ``torch.autocast("supa", dtype=torch.float16)`` raises "Current SUPA
    #   Device does not support float16. Please switch dtype to bfloat16."
    # * bfloat16 is the working reduced precision (the vendor documents bf16 and
    #   fp32 for SDPA, and br10x matmul takes bf16/fp32/s8/u8).  One operator has
    #   no bf16 kernel -- bilinear upsample, which every separator model ends
    #   with -- so the interpolate shim runs that one op in fp32 on the device
    #   instead of disabling autocast for the whole model.
    # Dilations: only the SUDNN plan path aborts.  With the eager conv kernels
    # that pymss selects for BIREN (BRTB_ENABLE_FORCE_EAGER_CONV2D, see
    # _VENDOR_ENV_DEFAULTS) every asymmetric combination measured passes:
    #   pad=(0,1) dil=(1,2) max_err 5.8e-05 | pad=(1,0) dil=(1,2) 6.4e-05
    #   pad=(2,1) dil=(1,3) 6.3e-05       | pad=(1,2) dil=(2,1) 5.9e-05
    # so this profile keeps the limited list as a safety net for users who turn
    # the eager path off, and dilated_conv_blocked() short-circuits when it is on.
    BIREN: DeviceProfile(
        BIREN, "supa", plugin="torch_br", complex_fft=False, rnn=False,
        amp=True, amp_dtype=torch.bfloat16, dilated_conv=False,
        dilated_conv_allowed=frozenset({(1, 1), (2, 2)}),
        complex_mul=False, complex_reshape=False, window_on_device=False, glu_shim=True,
        group_norm_shim=True, interpolate_shim=True, grouped_conv1d=True, complex_cat=True, fft_shim=True, std_shim=True, cpu_shim=True, sdpa_shim=True,
        rope_shim=True,
        scatter_add=False,
    ),
    MLU: DeviceProfile(MLU, "mlu", plugin="torch_mlu"),
    # Ascend's torch_npu builds ship npu_lstm/npu_gru without inference support
    # and lack a complex STFT path, so both fallbacks are part of the profile.
    ASCEND: DeviceProfile(ASCEND, "npu", plugin="torch_npu", complex_fft=False, rnn=False, multi_device=False),
    MPS: DeviceProfile(MPS, "mps", complex_fft=False, rnn=True, amp=True, channels_last=False, multi_device=False),
    XPU: DeviceProfile(XPU, "xpu", amp=False, channels_last=False, multi_device=False),
    # Measured on Colab TPU v5e1 + torch 2.14.1+cpu / torch_tpu 0.2.0 (Python 3.13),
    # 16 GiB HBM, device type ``tpu``:
    # * ``torch.stft`` is exact (rel 3.1e-07) but ``torch.istft`` is not: it
    #   returns 1.5e-03 rel / 2.8e-03 max error and takes 217 ms where pymss's
    #   ``istft_npu`` overlap-add composition does it in 6.8 ms at 1.5e-07.
    #   ``complex_fft=False`` therefore routes both STFT and ISTFT through the
    #   ``torch.fft`` path, which is 2.1x faster end to end AND 80x more
    #   accurate (mask-core end-to-end rel 3.8e-05 instead of 3.2e-03).
    # * ``F.group_norm``/``nn.GroupNorm`` die on a non-contiguous input with
    #   "view(): cannot create a view of shape [...]", which is what SCNet and
    #   htdemucs feed them.  A ``.contiguous()`` copy before the kernel makes
    #   them exact (rel 5.1e-08).
    # * ``torch.tpu.set_device`` is internal to TorchTPU and raises; device
    #   selection has to go through ``torch.accelerator``.
    # * Recurrent cells are native and accurate (LSTM 6.0e-05, GRU 3.0e-05) and
    #   take 1.2 ms, so no CPU fallback.
    # * Everything else pymss uses has a native kernel and matches torch: complex
    #   mul/reshape/cat/view_as_complex, grouped + dilated conv, conv_transpose1d,
    #   replicate/reflect pad, bilinear upsample, fold/unfold, as_strided, std,
    #   scatter_add, index_add, GLU.
    # * fp32 matmul and convolution run in bfloat16 by default (rel 2.4e-03);
    #   ``high`` gives 1.4e-05 for 27.9/50.2 TFLOPS at 2048/4096 (slightly faster
    #   than the default at 4096), ``highest`` gives 2.4e-07 for 18.9/27.5
    #   TFLOPS.  ``high`` is free, so it is the profile default.
    # * bfloat16 is the native reduced precision; float16 works but is slower
    #   than fp32 (0.46 ms vs 0.44 ms for one 4096 GEMM) so it is never used.
    # * ``F.fold``/``col2im`` accumulate in bfloat16 (2.6e-03 over seven chunk
    #   windows) and take 99 ms where the exact ``scatter_add_`` composition
    #   takes 4.2 ms.  ``pymss.utils`` folds every chunk window, so the whole
    #   track inherited the 2.6e-03 even though one forward was 3e-05 off.
    # * ``torch.fft.rfft``/``irfft`` are the vendor kernels themselves and are
    #   exact (3.1e-07), so the per-call "did the vendor return silent zeros"
    #   probe is dead weight: it costs two full reductions and two host syncs per
    #   STFT and is a ``torch.compile`` graph break.
    TPU: DeviceProfile(
        TPU, "tpu", plugin="torch_tpu", complex_fft=False, rnn=True,
        amp=True, amp_dtype=torch.bfloat16, channels_last=False, multi_device=False,
        group_norm_contiguous=True, matmul_precision="high",
        fold_exact=True, fft_result_check=False,
    ),
    CPU: DeviceProfile(CPU, "cpu", amp=False, channels_last=False, multi_device=False),
}

#: Substrings looked for in ``torch.cuda.get_device_name()`` (lowercased).
_VENDOR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (HYGON, ("hygon", "dtk", "dcu", "k100", "z100", "bw1000", "shuxian")),
    (ILUVATAR, ("iluvatar", "corex", "bi-v", "mr-v", "bi150", "bi-v150")),
    (METAX, ("metax", "maca", "mxc", "c500", "n260", "xiyun")),
    (ENFLAME, ("enflame", "gcu", "s60", "t20", "cloudblazer")),
    (BIREN, ("biren", "supa", "br100", "br104", "bili")),
    (MLU, ("cambricon", "mlu")),
    (ASCEND, ("ascend", "npu")),
)

#: Devices that always keep their own profile, even under a forced vendor.
_HOST_DEVICE_TYPES = ("cpu", "mps")

#: Private device types that imply a vendor without any probing.
_DEVICE_TYPE_VENDOR: dict[str, str] = {
    "npu": ASCEND,
    "xla": ENFLAME,  # torch_gcu is torch_xla lineage: gcu_device() -> xla:0
    "gcu": ENFLAME,
    "mlu": MLU,
    "supa": BIREN,
    "mps": MPS,
    "xpu": XPU,
    "tpu": TPU,
    "cpu": CPU,
}

_cache: dict[str, DeviceProfile] = {}


def _env_flag(name: str) -> bool | None:
    """Return a tri-state boolean from ``name`` (``1/0``, ``true/false``, ...)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _device_type_of(device) -> str:
    """Normalise ``device`` (str, ``torch.device``, tensor, ``None``) to a type."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.Tensor):
        return device.device.type
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError):
        # Backends that are not registered in this build (``"npu:0"`` on a CUDA
        # host) cannot be parsed by torch.device; fall back to the type prefix so
        # the profile is still resolvable in tests and on mixed hosts.
        prefix = str(device).split(":", 1)[0].strip().lower()
        return prefix or "cpu"


def _cuda_device_name() -> str:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return str(torch.cuda.get_device_name(0)).lower()
    except Exception:  # a half-initialised vendor backend must not break detection
        return ""
    return ""


def detect_vendor(device_type: str) -> str:
    """Best-effort vendor detection for a torch device type.

    ``PYMSS_DEVICE_BACKEND`` wins over every probe for accelerators.  Host
    devices (``cpu``) keep their own profile so a forced accelerator vendor never
    reclassifies host tensors -- otherwise every CPU tensor would inherit the
    vendor's fallbacks.

    Args:
        device_type: ``torch.device(...).type`` of the target device.

    Returns:
        Vendor key; defaults to :data:`NVIDIA` for ``"cuda"`` and :data:`CPU`
        for anything unknown.
    """
    forced = os.environ.get("PYMSS_DEVICE_BACKEND", "").strip().lower()
    if forced in _PROFILES and device_type not in _HOST_DEVICE_TYPES:
        return forced

    if device_type in _DEVICE_TYPE_VENDOR:
        return _DEVICE_TYPE_VENDOR[device_type]
    if device_type != "cuda":
        return CPU

    name = _cuda_device_name()
    for vendor, keywords in _VENDOR_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            return vendor
    if getattr(torch.version, "hip", None):
        return ROCM
    return NVIDIA


def _with_overrides(profile: DeviceProfile) -> DeviceProfile:
    """Apply ``PYMSS_FFT_FALLBACK`` / ``PYMSS_RNN_FALLBACK`` / ``PYMSS_AMP`` / ``PYMSS_ROPE_BACKEND``."""
    updates: dict[str, bool] = {}
    fft_fallback = _env_flag("PYMSS_FFT_FALLBACK")
    if fft_fallback is not None:
        updates["complex_fft"] = not fft_fallback
    rnn_fallback = _env_flag("PYMSS_RNN_FALLBACK")
    if rnn_fallback is not None:
        updates["rnn"] = not rnn_fallback
    amp = _env_flag("PYMSS_AMP")
    if amp is not None:
        updates["amp"] = amp
    rope = os.environ.get("PYMSS_ROPE_BACKEND", "").strip().lower()
    if rope in ("manual", "vendor"):
        updates["rope_shim"] = rope == "vendor"
    precision = os.environ.get("PYMSS_MATMUL_PRECISION", "").strip().lower()
    if precision in ("highest", "high", "medium"):
        updates["matmul_precision"] = precision
    return replace(profile, **updates) if updates else profile


def device_profile(device=None) -> DeviceProfile:
    """Return the (cached) capability profile for ``device``.

    Args:
        device: Optional device spec -- ``None``, a device string, a
            ``torch.device`` or a tensor.

    Returns:
        The matching :class:`DeviceProfile`.
    """
    device_type = _device_type_of(device)
    cached = _cache.get(device_type)
    if cached is not None:
        return cached
    profile = _with_overrides(_PROFILES[detect_vendor(device_type)])
    _cache[device_type] = profile
    return profile


def clear_device_cache() -> None:
    """Drop the detection cache (tests, or after loading a vendor plugin)."""
    _cache.clear()


def profile_for_alias(alias: str | None) -> DeviceProfile | None:
    """Return the profile implied by a ``--device`` alias, or ``None``."""
    if not alias:
        return None
    vendor = ALIAS_TO_VENDOR.get(str(alias).strip().lower())
    return _PROFILES[vendor] if vendor else None


def requires_plugin(alias: str) -> str | None:
    """Return the vendor extension that must be imported for ``alias``."""
    profile = profile_for_alias(alias)
    return profile.plugin if profile else None


#: Defaults applied to ``os.environ`` *before* the vendor extension is imported.
#:
#: BIREN's SUDNN spends ~220 s choosing a kernel for a new convolution shape on
#: a 1025x860 spectrogram, while the steady-state kernel takes 10-15 ms.  pymss
#: feeds it a new shape for every audio chunk, so the search dominates.
#:
#: Measured on BIREN106M / torch-br 1.7.0.20700, one 3x3 fp32 conv on
#: (2,32,1025,860), timeout 75 s per candidate:
#:
#: ==============================  ===========  ==========
#: setting                         first call   steady
#: ==============================  ===========  ==========
#: default (SUDNN search)          >75 000 ms   14.9 ms
#: ``BRTB_DISABLE_HEURISTIC_SEARCH`` >75 000 ms    -
#: ``BRTB_ENABLE_BUILD_CACHE``     >75 000 ms    -
#: ``BRTB_ENABLE_OP_FRONTEND_CACHE`` >75 000 ms   -
#: ``BRTB_ENABLE_FAST_MAKE_KERNEL_KEY`` >75 000 ms -
#: ``BRTB_ENABLE_FORCE_EAGER_CONV2D`` 1 790 ms   21.0 ms
#: ==============================  ===========  ==========
#:
#: The eager path is ~40% slower per call in steady state but removes a per-shape
#: search that costs minutes, so it is the default.  Values set by the user are
#: never overwritten; ``PYMSS_VENDOR_ENV=0`` disables the whole mechanism.
_VENDOR_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    BIREN: {
        "BRTB_ENABLE_FORCE_EAGER_CONV2D": "1",
        "BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE": "1",
    },
}


def vendor_env_defaults(vendor: str) -> dict[str, str]:
    """Return the environment defaults pymss applies for ``vendor``."""
    return dict(_VENDOR_ENV_DEFAULTS.get(vendor, {}))


def apply_vendor_env(device_or_vendor) -> dict[str, str]:
    """Apply :data:`_VENDOR_ENV_DEFAULTS` for a device or vendor key.

    Must run before the vendor extension is imported, because the backend reads
    these switches when it builds its kernels.

    Args:
        device_or_vendor: Vendor key (``"biren"``) or any device spec.

    Returns:
        The variables that were set, for logging.
    """
    vendor = device_or_vendor if device_or_vendor in _PROFILES else detect_vendor(_device_type_of(device_or_vendor))
    if _env_flag("PYMSS_VENDOR_ENV") is False:
        return {}
    applied: dict[str, str] = {}
    for name, value in vendor_env_defaults(vendor).items():
        if os.environ.get(name):
            continue
        os.environ[name] = value
        applied[name] = value
    return applied


def set_current_vendor_device(device_type: str, index: int | None = None) -> int | None:
    """Make ``index`` the current device for a vendor backend.

    A device spec without an index (``torch.device("supa")``, which is what
    pymss builds) resolves to the *current* device.  For PrivateUse1 backends
    nothing sets it implicitly, so an op that needs it fails with a bare

        ERROR (SUPA): ErrorCode: 101, invalid device

    ``torch.cuda.set_device`` has the same role on NVIDIA.  The index defaults
    to ``PYMSS_DEVICE_INDEX`` and then 0.

    Args:
        device_type: Device type string, e.g. ``"supa"``.
        index: Device index to select.

    Returns:
        The index that was set, or ``None`` when the backend has no ``set_device``.
    """
    if device_type in ("cpu", "meta"):
        return None  # torch.cpu.set_device exists but has nothing to select
    if device_type == TPU:
        # torch_tpu deliberately raises from ``torch.tpu.set_device`` ("internal
        # to TorchTPU"); ``torch.accelerator`` is the public entry point.
        chosen = index if index is not None else int(os.environ.get("PYMSS_DEVICE_INDEX", "0") or 0)
        try:
            torch.accelerator.set_device_index(chosen)
        except Exception:  # noqa: BLE001 - an older torch has no accelerator API
            return None
        return chosen
    module = getattr(torch, device_type, None)
    setter = getattr(module, "set_device", None) if module is not None else None
    if setter is None:
        return None
    chosen = index if index is not None else int(os.environ.get("PYMSS_DEVICE_INDEX", "0") or 0)
    try:
        setter(chosen)
    except Exception:  # noqa: BLE001 - a vendor may reject the index; keep going
        return None
    return chosen


def prepare_device_alias(alias: str) -> str:
    """Import the vendor extension for ``alias`` and return the device type.

    Every domestic vendor needs its extension imported before the backend is
    registered on ``torch`` (``torch_npu``, ``torch_gcu``, ``torch_mlu``,
    ``torch_br``).  Imports are best-effort: an ImportError is reported to the
    caller so it can raise a helpful error message.

    Args:
        alias: A ``--device`` value such as ``"biren"`` or ``"npu"``.

    Returns:
        The torch device type string to build devices with.
    """
    profile = profile_for_alias(alias)
    if profile is None:
        return str(alias)
    if profile.plugin:
        # The backend reads its tuning switches when the extension loads.
        apply_vendor_env(profile.vendor)
        __import__(profile.plugin)
    clear_device_cache()
    device_type = profile.device_type
    if device_type == "supa" and not hasattr(torch, "supa"):
        # Older BIRENSUPA releases expose SUPA devices as CUDA.
        device_type = "cuda"
    # PrivateUse1 backends do not set a current device on their own.
    set_current_vendor_device(device_type)
    # Kernel gaps that every model would otherwise have to work around.
    from pymss_core.vendor_shims import install_vendor_shims

    profile_now = device_profile(device_type)
    install_vendor_shims(
        device_type,
        complex_mul=profile_now.complex_mul,
        complex_reshape=profile_now.complex_reshape,
        window_on_device=profile_now.window_on_device,
        glu=profile_now.glu_shim,
        group_norm=profile_now.group_norm_shim,
        interpolate=profile_now.interpolate_shim,
        conv1d=profile_now.grouped_conv1d,
        complex_cat=profile_now.complex_cat,
        fft=profile_now.fft_shim,
        fft_dtype=profile_now.fft_dtype_shim,
        std=profile_now.std_shim,
        as_strided_shim=profile_now.as_strided_shim,
        unfold_shim=profile_now.unfold_shim,
        cpu=profile_now.cpu_shim,
        sdpa=profile_now.sdpa_shim,
        rope=profile_now.rope_shim,
        group_norm_contiguous=profile_now.group_norm_contiguous,
        fold_exact=profile_now.fold_exact,
    )
    # Recurrent cells: the fused _VF.lstm is both wrong and catastrophically slow
    # on BIREN (605 s for one (18, 57, 128) call), and the profile already says
    # so, so preparing the device installs the fallback instead of leaving it to
    # whichever caller remembers.
    if profile_now.needs_rnn_fallback or profile_now.rnn_blocked:
        from pymss_core.npu_rnn import install_rnn_cpu_fallback

        install_rnn_cpu_fallback()
    # A global knob, so it is set once here rather than at every matmul site.
    # TPU executes fp32 matmul/conv in bfloat16 unless asked otherwise, which is
    # 2.4e-03 relative error over a whole model.
    if profile_now.matmul_precision:
        torch.set_float32_matmul_precision(profile_now.matmul_precision)
    return device_type


def scatter_add_blocked(device=None) -> bool:
    """Return ``True`` when ``scatter_add_`` has no kernel and must be emulated."""
    return not device_profile(device).scatter_add


def fft_result_check_required(device=None) -> bool:
    """Return ``True`` when FFT results need a per-call silent-zero probe."""
    return device_profile(device).fft_result_check


def fft_fallback_required(device=None) -> bool:
    """Return ``True`` when ``torch.stft``/``torch.istft`` must be replaced."""
    return device_profile(device).needs_fft_fallback


def rnn_fallback_required(device=None) -> bool:
    """Return ``True`` when recurrent layers must bounce through CPU."""
    return device_profile(device).needs_rnn_fallback


def rnn_op_blocked(device, op: str) -> bool:
    """Return ``True`` when the recurrent cell ``op`` must bounce through CPU.

    ``op`` is the lowercase cell name (``"lstm"``, ``"gru"``, ``"rnn"``).  A
    device level ``rnn=False`` blocks every cell; otherwise only the cells listed
    in :attr:`DeviceProfile.rnn_blocked` are affected.
    """
    profile = device_profile(device)
    if not profile.rnn:
        return True
    return str(op).lower() in profile.rnn_blocked


def dilated_conv_supported(device=None) -> bool:
    """Return ``True`` when the vendor can plan dilated convolutions."""
    return device_profile(device).dilated_conv


def _normalise_dilation(dilation) -> tuple[int, int]:
    """Return ``(h, w)`` for an int / 1-tuple / 2-tuple dilation."""
    if isinstance(dilation, int):
        return (dilation, dilation)
    values = tuple(int(value) for value in dilation)
    if not values:
        return (1, 1)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


def eager_conv_enabled() -> bool:
    """Return ``True`` when BIREN convs run on the vendor eager kernels.

    Those kernels accept every dilation and skip the per-shape SUDNN plan
    search; :data:`_VENDOR_ENV_DEFAULTS` enables them automatically.
    """
    return os.environ.get("BRTB_ENABLE_FORCE_EAGER_CONV2D", "") not in ("", "0", "false", "False")


def dilated_conv_blocked(device, dilation) -> bool:
    """Return ``True`` when a conv with ``dilation`` must bounce through CPU.

    A profile that supports every dilation returns ``False``; a restricted
    profile only keeps the dilations it lists (symmetric, ``h == w``).
    """
    profile = device_profile(device)
    if profile.vendor == BIREN and eager_conv_enabled():
        return False  # eager conv kernels handle every dilation (measured)
    if profile.dilated_conv:
        return False
    h, w = _normalise_dilation(dilation)
    if h != w:
        return True
    return (h, w) not in profile.dilated_conv_allowed


def autocast_dtype(device=None) -> torch.dtype:
    """Return the autocast dtype the backend supports (float16 or bfloat16)."""
    profile = device_profile(device)
    return profile.amp_dtype if profile.amp else torch.float32


def amp_supported(device=None) -> bool:
    """Return ``True`` when autocast is usable on ``device``."""
    return device_profile(device).supports_amp


__all__ = (
    "ALIAS_TO_VENDOR",
    "ASCEND",
    "BIREN",
    "CPU",
    "DEVICE_ALIASES",
    "DeviceProfile",
    "ENFLAME",
    "HYGON",
    "ILUVATAR",
    "METAX",
    "MLU",
    "MPS",
    "NVIDIA",
    "ROCM",
    "TPU",
    "XPU",
    "amp_supported",
    "apply_vendor_env",
    "autocast_dtype",
    "dilated_conv_blocked",
    "dilated_conv_supported",
    "scatter_add_blocked",
    "fft_result_check_required",
    "eager_conv_enabled",
    "clear_device_cache",
    "detect_vendor",
    "device_profile",
    "fft_fallback_required",
    "prepare_device_alias",
    "profile_for_alias",
    "requires_plugin",
    "rnn_fallback_required",
    "rnn_op_blocked",
    "set_current_vendor_device",
    "vendor_env_defaults",
)
