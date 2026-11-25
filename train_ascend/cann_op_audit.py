"""Read-only audit of the CANN operator surface behind our STFT / complex / fold path.

Nothing here touches training. Three stages:

  map      - aten op -> is there an NPU implementation -> which aclnn/acl symbol backs it
  kernels  - how many device kernels each aten op actually launches on npu:0
  stft     - four STFT implementations, fwd and fwd+bwd, fp32 and bf16, with numeric checks
  npuapi   - what torch.ops.npu / torch_npu expose for FFT, STFT and complex

    python cann_op_audit.py --stage map
    python cann_op_audit.py --stage kernels --out results/cann_op_kernels.json
    python cann_op_audit.py --stage stft --out results/stft_impl_bench.json
    python cann_op_audit.py --stage npuapi
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

TARGETS = [
    "_fft_r2c", "_fft_c2c", "_fft_c2r", "_fft_r2c_backward", "_fft_c2c_backward",
    "stft", "istft", "unfold", "col2im", "im2col", "fold", "unfold_backward",
    "view_as_complex", "view_as_real", "_conj", "complex_mul", "l_shift", "reflection_pad1d",
    "real", "imag", "complex", "angle", "conv1d", "avg_pool2d", "col2im",
]

SITE = Path("/usr/local/python3.12.16/lib/python3.12/site-packages/torch_npu")
YAML = SITE / "csrc/aten/npu_native_functions_by_codegen.yaml"
NATIVE_YAML = SITE / "csrc/aten/npu_native_functions.yaml"
CANN = Path("/usr/local/Ascend/ascend-toolkit/latest")


# --------------------------------------------------------------------------- map
def stage_map(_args) -> int:
    txt = YAML.read_text()
    entries, lines = {}, txt.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^- func: (\S+)", line)
        if not m:
            continue
        name, flags = m.group(1).split("(")[0], []
        for j in range(i + 1, min(i + 6, len(lines))):
            if lines[j].startswith("- func:"):
                break
            m2 = re.match(r"^  (impl_ns|dispatch|device_check|autograd):\s*(\S+)", lines[j])
            if m2:
                flags.append(f"{m2.group(1)}={m2.group(2)}")
        entries.setdefault(name, " ".join(flags))

    native = set()
    for line in NATIVE_YAML.read_text().splitlines():
        m = re.match(r"^\s*- func: (\S+)", line) or re.match(r"^\s*- (\w+)", line)
        if m:
            native.add(m.group(1).split("(")[0])

    # which CANN symbols exist? acl* (legacy) vs aclnn* (new)
    libs = [CANN / "x86_64-linux/lib64/libopapi_math.so", CANN / "x86_64-linux/lib64/libopapi.so"]
    symbols = set()
    for lib in libs:
        if not lib.exists():
            continue
        out = subprocess.run(["nm", "-D", "--defined-only", str(lib)], capture_output=True, text=True).stdout
        symbols |= set(re.findall(r"\b(acl(?:nn)?[A-Za-z]*(?:Fft|Stft|Rfft)[A-Za-z]*)\b", out))
    so = subprocess.run(["strings", "-a", str(SITE / "lib/libtorch_npu.so")],
                        capture_output=True, text=True).stdout
    so_hits = sorted(set(re.findall(r"aclnn[A-Za-z]*(?:Fft|Stft|Complex|Col2Im|Im2col|UnfoldGrad)[A-Za-z]*", so)))

    print(f"{'aten op':26s} {'NPU impl':9s} {'registered via':22s} flags")
    rows = []
    for t in TARGETS:
        hit = next((k for k in entries if k == t or k.startswith(t + ".")), None)
        impl = "yes" if hit else ("view/composite" if t in native else "NO")
        via = "op_api->aclnn" if hit and "op_api" in entries[hit] else (
            "aclnn (acl_op)" if hit else "-")
        print(f"{t:26s} {impl:9s} {via:22s} {entries.get(hit, '') if hit else ''}")
        rows.append({"aten": t, "npu_impl": impl, "via": via, "flags": entries.get(hit, "")})
    print(f"\nCANN acl/aclnn FFT-ish exported symbols: {sorted(symbols)}")
    print(f"libtorch_npu.so aclnn strings: {so_hits}")
    print(f"950 kernel configs present: "
          f"{sorted(p.name for p in (CANN / 'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_math').glob('*') if re.search('stft|rfft|complex', p.name, re.I))}")
    if _args.out:
        _args.out.write_text(json.dumps({"aten": rows, "cann_symbols": sorted(symbols),
                                         "so_strings": so_hits}, indent=2))
    return 0


# ----------------------------------------------------------------------- kernels
def stage_kernels(args) -> int:
    import torch
    import torch_npu

    dev = torch.device("npu:0")
    from torch.profiler import ProfilerActivity, profile

    def count_kernels(fn, n_iter=3):
        fn()
        torch.npu.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            for _ in range(n_iter):
                fn()
            torch.npu.synchronize()
        n_npu = n_aicpu = 0
        names = {}
        for evt in prof.events():
            dev_type = str(getattr(evt, "device_type", ""))
            if "NPU" not in dev_type and "PrivateUse1" not in dev_type:
                continue
            key = evt.key.split("(")[0]
            names[key] = names.get(key, 0) + 1
            if "aicpu" in key.lower() or "AICPU" in key:
                n_aicpu += 1
            else:
                n_npu += 1
        total = n_npu + n_aicpu
        return round(total / n_iter, 2), round(n_aicpu / n_iter, 2), dict(
            sorted(names.items(), key=lambda kv: -kv[1])[:5])

    fx = torch.randn(24, 2, 352800, device=dev)
    win = torch.hann_window(2048).to(dev)   # hann_window on NPU hits aclnnInplaceMuls 161002
    cf = torch.randn(24, 2, 1025, 690, device=dev).to(torch.complex64)  # no complex randn on NPU
    rf = torch.view_as_real(cf)
    cases = {
        "aten::_fft_r2c": lambda: torch.fft.rfft(fx, n=2048, dim=-1),
        "aten::stft(return_complex=True)": lambda: torch.stft(
            fx.reshape(-1, 352800), 2048, 512, window=win, return_complex=True),
        "aten::stft(return_complex=False)": lambda: torch.stft(
            fx.reshape(-1, 352800), 2048, 512, window=win, return_complex=False),
        "aten::_fft_c2c": lambda: torch.fft.fft(cf, dim=-1),
        "aten::_fft_c2r": lambda: torch.fft.irfft(cf, n=2048, dim=-1),
        "aten::istft": lambda: torch.istft(cf, 2048, 512, window=win, length=352800),
        "aten::unfold": lambda: fx.unfold(-1, 2048, 512),
        "aten::as_strided": lambda: fx.as_strided((24, 2, 690, 2048), (705600, 352800, 512, 1)),
        "aten::view_as_real": lambda: torch.view_as_real(cf),
        "aten::view_as_complex": lambda: torch.view_as_complex(rf),
        "aten::col2im": lambda: torch.nn.functional.fold(
            torch.randn(24, 2 * 2048, 690, device=dev), (1, 352800), (1, 2048), stride=(1, 512)),
        "aten::fold(F)": lambda: torch.nn.functional.fold(
            torch.randn(24, 2048, 11818, device=dev), (1, 352800), (1, 2048), stride=(1, 512)),
        "aten::reflection_pad1d": lambda: torch.nn.functional.pad(fx, (512, 512), mode="reflect"),
        "complex_mul(addcmul)": lambda: torch.view_as_complex(rf) * torch.view_as_complex(rf),
        "aten::conv1d(depthwise k7)": lambda: torch.conv1d(
            torch.randn(24, 192, 690, device=dev), torch.randn(192, 1, 7, device=dev), groups=192),
    }
    rows = {}
    for name, fn in cases.items():
        try:
            per_call, aicpu, top = count_kernels(fn)
            rows[name] = {"device_kernels_per_call": per_call, "aicpu_kernels_per_call": aicpu,
                          "top_kernels": top,
                          "verdict": "native (1-3)" if per_call <= 3 else
                                     ("fused-ish (4-10)" if per_call <= 10 else "decomposed (many)")}
            print(f"{name:34s} {per_call:7.2f} kernels/call  aicpu={aicpu:5.2f}  {rows[name]['verdict']}")
        except Exception as exc:  # noqa: BLE001
            rows[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"{name:34s} FAILED {type(exc).__name__}: {str(exc)[:120]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


# -------------------------------------------------------------------------- stft
def stage_stft(args) -> int:
    import torch
    import torch_npu
    import torch.nn.functional as F

    sys_path = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path))
    from fast_loss import stft_mag_fast

    dev = torch.device("npu:0")
    B, C, L, N, HOP = 24, 2, 352800, 2048, 512
    win = torch.hann_window(N).to(dev, torch.float32)

    def flat(x):
        """The loss calls these on a 2-D (batch*channels, length) signal: torch.stft only
        accepts 1-D/2-D input."""
        return x.reshape(-1, L) if x.dim() == 3 else x

    def ref_stft(x):                      # native path: aten::stft -> aclStft
        return torch.stft(flat(x), N, HOP, window=win, return_complex=True)

    def torch_mag(x):
        return torch.stft(flat(x), N, HOP, window=win, return_complex=True).abs()

    def pure_torch_mag(x):
        """rfft + as_strided framing + broadcast window multiply (no aten::unfold).

        center=True pads n_fft//2 on both sides, which is what makes the frame count 690
        rather than 686, so the as_strided view is taken on the padded signal.
        """
        # unfold is a pure view here (0 device kernels, measured), so this framing is free;
        # as_strided on the padded tensor faults the vector core on this stack (error 95),
        # which is why the view goes through unfold + reshape instead.
        xp = F.pad(flat(x), (N // 2, N // 2), mode="reflect")
        frames = xp.unfold(-1, N, HOP).reshape(xp.shape[0], 690, N)
        return torch.fft.rfft(frames * win, dim=-1).abs()

    def bench(fn, it=5, warm=2):
        for _ in range(warm):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(it):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return round(statistics.median(ts), 2)

    def bench_bwd(make_loss, it=5, warm=2):
        """fwd + backward of a scalar loss derived from the spectrum/magnitude."""
        def run(dtype):
            x = torch.randn(B, C, L, device=dev, dtype=dtype, requires_grad=True)
            def go():
                loss = make_loss(x)
                loss.float().mean().backward()
            return go
        go = run(torch.float32)
        return bench(go, it, warm)

    def rel(a, b):
        a, b = a.float(), b.float()
        return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12))

    out = {"config": {"batch": B, "channels": C, "length": L, "n_fft": N, "hop": HOP,
                      "frames": 690, "note": "NPU shared with the running training job: "
                                             "kernel counts are exact, times are inflated but comparable"}}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        x = torch.randn(B, C, L, device=dev, dtype=dtype)
        res = {}
        # a) native torch.stft magnitude
        res["a_torch_stft_mag"] = {"fwd_ms": bench(lambda: torch_mag(x))}
        # b) my fast_loss magnitude (rfft + explicit adjoint)
        res["b_fast_loss_mag"] = {"fwd_ms": bench(lambda: stft_mag_fast(flat(x), win, N, HOP))}
        # c) pure torch rfft + as_strided + broadcast window
        res["c_pure_torch_mag"] = {"fwd_ms": bench(lambda: pure_torch_mag(x))}
        # d) native complex stft (what aten::stft gives without .abs())
        res["d_native_complex"] = {"fwd_ms": bench(lambda: ref_stft(x))}

        # numeric agreement of the magnitude variants against the native one
        with torch.no_grad():
            ref = torch_mag(x.float() if dtype == torch.bfloat16 else x)
            res["b_fast_loss_mag"]["rel_rms_vs_a"] = rel(stft_mag_fast(flat(x), win, N, HOP), ref)
            res["c_pure_torch_mag"]["rel_rms_vs_a"] = rel(pure_torch_mag(x), ref)

        # fwd+bwd timings (loss = mean magnitude, so the adjoint path is exercised)
        for key, mk in (("a_torch_stft_mag", torch_mag),
                        ("b_fast_loss_mag", lambda z: stft_mag_fast(flat(z), win, N, HOP)),
                        ("c_pure_torch_mag", pure_torch_mag)):
            def make_loss(z, mk=mk):
                return mk(z)
            def go(mk=mk, dt=dtype):
                xx = torch.randn(B, C, L, device=dev, dtype=dt, requires_grad=True)
                mk(xx).float().mean().backward()
            res[key]["fwd_bwd_ms"] = bench(go)
        # gradient agreement, fp32 only (bf16 grads are not meaningful at 1e-6)
        if dtype == torch.float32:
            xref = x.detach().clone().requires_grad_(True)
            torch_mag(xref).mean().backward()
            for key, mk in (("b_fast_loss_mag", lambda z: stft_mag_fast(flat(z), win, N, HOP)),
                            ("c_pure_torch_mag", pure_torch_mag)):
                xk = x.detach().clone().requires_grad_(True)
                mk(xk).mean().backward()
                res[key]["grad_rel_rms_vs_a"] = rel(xk.grad, xref.grad)
        out[tag] = res
        print(f"--- {tag}: " + json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                                           for k, v in res.items()}, indent=1)[:1200])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------------ npuapi
def stage_npuapi(_args) -> int:
    import torch
    import torch_npu

    names = [n for n in dir(torch.ops.npu) if not n.startswith("_")]
    fftish = [n for n in names if any(k in n.lower() for k in ("fft", "stft", "complex", "conj", "real", "imag", "fold", "im2col", "col2im"))]
    py = [n for n in dir(torch_npu) if any(k in n.lower() for k in ("fft", "stft", "complex"))]
    npuns = [n for n in dir(torch_npu.npu) if any(k in n.lower() for k in ("fft", "stft", "complex"))]
    print(f"torch.ops.npu total={len(names)}; fft-ish={fftish}")
    print(f"torch_npu.* : {py}")
    print(f"torch_npu.npu.* : {npuns}")
    cache = getattr(torch_npu.npu, "_fft_plan_cache", None)
    if cache is not None:
        print(f"_fft_plan_cache: {cache} attrs={[a for a in dir(cache) if not a.startswith('__')][:12]}")
        try:
            print(f"  size={cache.size()} max_size={getattr(cache, 'max_size', None)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  probe failed: {exc}")
    print("direct aclnn bypass for stft/fft available in python:", bool(
        [n for n in fftish if n in ("stft", "fft", "rfft", "aclnn_stft")]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("map", "kernels", "stft", "npuapi"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    return {"map": stage_map, "kernels": stage_kernels, "stft": stage_stft,
            "npuapi": stage_npuapi}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
