"""Peak calibration for this 950PR: GEMM TFLOPS, HBM bandwidth, elementwise roofs,
small-op launch overhead, and where the current model sits against them.

Everything lands in train_ascend/results/peak_calib.json.

    python calibrate_peak.py --out results/peak_calib.json

Method notes
- every timed loop is closed by `torch.npu.synchronize()`; medians over 30 reps
  after 10 warmups (fewer reps for the very large shapes where one call is >100 ms)
- GFLOPs for GEMM = 2*M*K*N / t; bandwidth = bytes moved / t (read+write where the
  op writes), so the reported GB/s is "effective", counting every byte touched
- launch overhead is separated from execution by timing a 1-element op 1000x
  (pure dispatch/launch) and the real activation shape 1000x (launch+execution)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

NOMINAL_HBM_TB_S = 1.6  # vendor figure for this part; user-supplied
NOMINAL_BF16_TFLOPS = 425.0  # vendor table, 28-core tier (486 for the 32-core tier; this part reports 28 AI cores)
NOMINAL_TF32_TFLOPS = 212.0  # same table, 28-core tier (fp32 has no cube path; TF32 does)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}

# (M, K, N) -- 6 square + 7 rectangular, all >= 512, 4096 and 8192 included
GEMM_SHAPES = [
    (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
    (8192, 8192, 8192), (16384, 16384, 16384),
    (4096, 4096, 1024), (1024, 4096, 4096), (8192, 8192, 2048), (2048, 8192, 8192),
    (16384, 4096, 4096), (4096, 16384, 4096), (16384, 8192, 8192),
]

ACT_SHAPE = (690, 62, 192)  # the shape that dominates the U-Net activations


def sync():
    torch.npu.synchronize()


def timed(fn, warmup: int = 10, reps: int = 30) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(times), 4),
            "min_ms": round(min(times), 4),
            "mean_ms": round(statistics.mean(times), 4), "reps": reps}


def timed_amortized(fn, inner: int, warmup: int = 2, reps: int = 10) -> float:
    """Time `inner` calls inside one sync window -> per-call ms incl. dispatch."""
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    sync()
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000 / inner)
    return round(statistics.median(times), 5)


def gemm_section(device) -> dict:
    out = {}
    for name, dtype in DTYPES.items():
        rows = []
        peak = None
        n_big = 0
        for (m, k, n) in GEMM_SHAPES:
            # keep memory sane: fp32 16384^3 would be 1 GiB per matrix, fine, but
            # 30 reps of a 400 ms call is 12 s -- drop reps for the biggest shapes.
            flops = 2 * m * k * n
            big = flops > 5e11
            reps = 5 if big else 30
            warmup = 2 if big else 10
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            try:
                t = timed(lambda: a @ b, warmup=warmup, reps=reps)
            except Exception as exc:
                rows.append({"M": m, "K": k, "N": n, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
                del a, b
                torch.npu.empty_cache()
                continue
            tflops = flops / (t["median_ms"] * 1e-3) / 1e12
            row = {"M": m, "K": k, "N": n, "shape": f"({m},{k})@({k},{n})",
                   "gflop": round(flops / 1e9, 1), "median_ms": t["median_ms"],
                   "tflops": round(tflops, 2), "reps": t["reps"]}
            if peak is None or tflops > peak["tflops"]:
                peak = row
            rows.append(row)
            n_big += 1
            del a, b
            torch.npu.empty_cache()
        out[name] = {"rows": rows, "peak": peak,
                     "peak_tflops": peak["tflops"] if peak else None,
                     "peak_shape": peak["shape"] if peak else None,
                                  "nominal_tflops": NOMINAL_BF16_TFLOPS,
                     "share_of_nominal_pct": None}
    return out


def bandwidth_section(device) -> dict:
    out = {"nominal_hbm_tb_s": NOMINAL_HBM_TB_S, "rows": []}
    for nbytes in (1 << 30, 2 << 30):
        elems = nbytes // 4
        src = torch.empty(elems, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)
        src.normal_()
        sync()

        t_copy = timed(lambda: dst.copy_(src), warmup=5, reps=10)
        gb = 2 * elems * 4 / 1e9  # read + write
        out["rows"].append({"op": "d2d copy_ (copy 1-pass)", "size_gib": round(elems * 4 / 2**30, 2),
                            "bytes_moved_gb": round(gb, 3), "median_ms": t_copy["median_ms"],
                            "effective_gb_s": round(gb / (t_copy["median_ms"] * 1e-3), 1)})

        x = torch.zeros(elems, device=device, dtype=torch.float32)

        t_add = timed(lambda: x.add_(1.0), warmup=5, reps=10)
        out["rows"].append({"op": "add_(1) read+write", "size_gib": round(elems * 4 / 2**30, 2),
                            "bytes_moved_gb": round(gb, 3), "median_ms": t_add["median_ms"],
                            "effective_gb_s": round(gb / (t_add["median_ms"] * 1e-3), 1)})

        y = torch.empty_like(x)

        t_mul = timed(lambda: torch.mul(x, 2.0, out=y), warmup=5, reps=10)
        out["rows"].append({"op": "mul out= (read+write)", "size_gib": round(elems * 4 / 2**30, 2),
                            "bytes_moved_gb": round(gb, 3), "median_ms": t_mul["median_ms"],
                            "effective_gb_s": round(gb / (t_mul["median_ms"] * 1e-3), 1)})

        del src, dst, x, y
        torch.npu.empty_cache()
    best = max(r["effective_gb_s"] for r in out["rows"])
    out["best_effective_gb_s"] = best
    out["best_tb_s"] = round(best / 1000, 3)
    out["share_of_nominal_pct"] = round(100 * (best / 1000) / NOMINAL_HBM_TB_S, 1)
    return out


def elementwise_section(device) -> dict:
    numel = 1
    for d in ACT_SHAPE:
        numel *= d
    out = {"shape": list(ACT_SHAPE), "numel": numel, "rows": []}
    for dtype_name in ("bf16", "fp32"):
        dtype = DTYPES[dtype_name]
        x = torch.randn(*ACT_SHAPE, device=device, dtype=dtype)
        b = 2 * numel * x.element_size() / 1e9  # out-of-place: read + write
        ops = {
            "gelu(tanh)": lambda: F.gelu(x),
            "silu": lambda: F.silu(x),
            "sigmoid": lambda: torch.sigmoid(x),
            "rmsnorm(dim=-1)": lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6),
            "add_(1) in-place": lambda: x.add_(1.0),
        }
        for label, fn in ops.items():
            t = timed(fn, warmup=10, reps=30)
            bytes_gb = b if "in-place" not in label else 2 * numel * x.element_size() / 1e9
            out["rows"].append({
                "op": label, "dtype": dtype_name,
                "median_ms": t["median_ms"],
                "effective_gb_s": round(bytes_gb / (t["median_ms"] * 1e-3), 1),
                "launch_share_pct": None,  # filled after launch calibration
            })
        del x
        torch.npu.empty_cache()
    return out


def launch_section(device) -> dict:
    out = {}
    # pure launch/dispatch: a 1-element op still goes through the whole host path
    tiny = torch.zeros(1, device=device)
    per_tiny = timed_amortized(lambda: tiny.add_(1.0), inner=1000)
    out["tiny_op_per_call_us"] = round(per_tiny * 1000, 2)

    # real activation shape, same op: launch + execution
    for dtype_name in ("bf16", "fp32"):
        dtype = DTYPES[dtype_name]
        x = torch.randn(*ACT_SHAPE, device=device, dtype=dtype)
        y = torch.empty_like(x)
        per_act = timed_amortized(lambda: torch.add(x, 1.0, out=y), inner=1000)
        out.setdefault("activation_op_per_call_us", {})[dtype_name] = round(per_act * 1000, 2)
        # single-call latency (includes the sync, i.e. what a 1-op-at-a-time caller sees)
        one = timed(lambda: torch.add(x, 1.0, out=y), warmup=10, reps=30)
        out.setdefault("activation_op_synced_ms", {})[dtype_name] = one["median_ms"]
        del x, y
        torch.npu.empty_cache()
    return out


LAUNCH_US = None


def band_split_section(device, model_path=None) -> dict:
    """The 62-band Linear group: 62 individual mm calls vs one packed bmm."""
    from freq_unet import FreqUNetConfig, build_model

    cfg = FreqUNetConfig(n_fft=2048)
    model = build_model(cfg).to(device).eval()
    feats = []
    for seq in model.band_split.to_features:
        lin = [m for m in seq if isinstance(m, torch.nn.Linear)][0]
        feats.append(lin.in_features)
    out = {"n_bands": len(feats), "in_features_per_band": feats,
           "total_in_features": sum(feats), "dim": model.band_split.to_features[0][-1].out_features}

    rows = []
    for dtype_name in ("bf16", "fp32"):
        dtype = DTYPES[dtype_name]
        t_ = 690
        xs, ws = [], []
        for nf in feats:
            xs.append(torch.randn(t_, nf, device=device, dtype=dtype))
            ws.append(torch.randn(nf, out["dim"], device=device, dtype=dtype))
        flops = 2 * t_ * out["dim"] * sum(feats)

        def loop():
            for x, w in zip(xs, ws):
                x @ w

        t_loop = timed(loop, warmup=5, reps=20)
        per_loop = timed_amortized(loop, inner=20)
        per_call_ms = per_loop / len(feats)

        # packed: pad every band to max width, one bmm
        maxf = max(feats)
        xp = torch.zeros(len(feats), t_, maxf, device=device, dtype=dtype)
        wp = torch.zeros(len(feats), maxf, out["dim"], device=device, dtype=dtype)
        for i, (nf) in enumerate(feats):
            xp[i, :, :nf] = xs[i]
            wp[i, :nf] = ws[i]

        def packed():
            torch.bmm(xp, wp)

        t_packed = timed(packed, warmup=5, reps=20)
        rows.append({
            "dtype": dtype_name,
            "flops": round(flops / 1e9, 2),
            "loop_62_gemms_ms": t_loop["median_ms"],
            "loop_ms_per_62_gemms": round(per_loop, 4),
            "loop_per_single_gemm_us": round(per_call_ms * 1000, 2),
            "launch_share_of_single_gemm_pct": round(100 * LAUNCH_US / (per_call_ms * 1000), 1) if LAUNCH_US else None,
            "packed_bmm_ms": t_packed["median_ms"],
            "speedup_packed": round(t_loop["median_ms"] / t_packed["median_ms"], 3),
            "ach_tflops_loop": round(flops / (t_loop["median_ms"] * 1e-3) / 1e12, 3),
            "ach_tflops_packed": round(flops / (t_packed["median_ms"] * 1e-3) / 1e12, 3),
            "padding_waste_ratio": round(maxf * len(feats) / sum(feats), 3),
        })
        del xs, ws, xp, wp
        torch.npu.empty_cache()
    out["rows"] = rows

    # the model's own two paths (grouped bmm vs per-band Linear), activation cost included.
    # BandSplit takes the flattened feature layout produced by FreqUNet.forward: (B, T, F*C).
    f_bins = 2048 // 2 + 1
    x = torch.randn(1, 690, f_bins * 4, device=device) * 0.1  # f_bins x 2 channels x 2 (re,im)
    model.band_split.use_grouped_forward = True
    t_grouped = timed(lambda: model.band_split(x), warmup=5, reps=20)
    model.band_split.use_grouped_forward = False
    t_perband = timed(lambda: model.band_split(x), warmup=5, reps=20)
    out["model_band_split_grouped_ms"] = t_grouped["median_ms"]
    out["model_band_split_perband_ms"] = t_perband["median_ms"]
    return out


def model_section(device, launch_us: float) -> dict:
    from freq_unet import FreqUNetConfig, build_model

    cfg = FreqUNetConfig(n_fft=2048)
    model = build_model(cfg).to(device).eval()
    f = 2048 // 2 + 1
    spec = torch.randn(1, f * cfg.audio_channels, 690, 2, device=device) * 0.1

    flops = None
    op_count = None
    kernel_count = None
    aten_ops = None
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], record_shapes=False) as prof:
            with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
                model(spec)
        kernel_count = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)
        aten_ops = {}
        for e in prof.key_averages():
            if e.key.startswith("aten::"):
                aten_ops[e.key] = e.count
        op_count = sum(aten_ops.values())
    except Exception as exc:
        print(f"profiler op-count unavailable: {type(exc).__name__}: {exc}", flush=True)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter:
            with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
                model(spec)
        flops = int(counter.get_total_flops())
        flat = {}
        for mod_name, inner in counter.get_flop_counts().items():
            for op, fl in inner.items():
                flat[f"{mod_name}.{op}"] = flat.get(f"{mod_name}.{op}", 0) + int(fl)
        by_mod = flat
        top = sorted(by_mod.items(), key=lambda kv: -kv[1])[:8]
    except Exception as exc:
        by_mod, top = {}, []
        print(f"flop counter unavailable: {type(exc).__name__}: {exc}", flush=True)

    times = []
    for _ in range(30):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            model(spec)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = statistics.median(times)

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "input_frames": 690,
        "forward_median_ms": round(fwd_ms, 3),
        "forward_flops": flops,
        "forward_gflop": round(flops / 1e9, 2) if flops else None,
        "forward_tflops": round(flops / (fwd_ms * 1e-3) / 1e12, 2) if flops else None,
        "flop_counter_ops": op_count,
        "device_kernels_per_forward": kernel_count,
        "top_aten_ops": sorted(aten_ops.items(), key=lambda kv: -kv[1])[:10] if aten_ops else None,
        "flop_counter_top": top,
        "reference_claim": {"gflop": 116.0, "ms": 8.45, "tflops": 13.7,
                            "source": "user-supplied figure, not reproducible on this box "
                                      "(our measured forward is different, see forward_median_ms)"},
        "launch_us_used": launch_us,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/peak_calib.json"))
    parser.add_argument("--skip-big", action="store_true")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    props = {
        "name": torch.npu.get_device_name(0),
        "count": torch.npu.device_count(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "hbm_total_mb": round(torch.npu.mem_get_info()[1] / 2**20),
    }
    payload = {"device": props,
               "nominal": {"hbm_tb_s": NOMINAL_HBM_TB_S,
                           "bf16_tflops_28core": NOMINAL_BF16_TFLOPS,
                           "bf16_tflops_32core": 486.0,
                           "tf32_tflops_28core": NOMINAL_TF32_TFLOPS,
                           "source": "HiSilicon 950 series spec table (Cube+Vector totals)"}}

    print("[1/5] gemm", flush=True)
    payload["gemm"] = gemm_section(device)
    print("[2/5] bandwidth", flush=True)
    payload["bandwidth"] = bandwidth_section(device)
    print("[3/5] elementwise", flush=True)
    payload["elementwise"] = elementwise_section(device)
    print("[4/5] launch overhead", flush=True)
    global LAUNCH_US
    payload["launch"] = launch_section(device)
    LAUNCH_US = payload["launch"]["tiny_op_per_call_us"]
    print("[5/5] band split + model", flush=True)
    payload["band_split"] = band_split_section(device)
    payload["model"] = model_section(device, payload["launch"]["tiny_op_per_call_us"])

    # ---- derived conclusions
    peak_bf16 = payload["gemm"]["bf16"]["peak_tflops"]
    peak_fp32 = payload["gemm"]["fp32"]["peak_tflops"]
    bw = payload["bandwidth"]["best_effective_gb_s"]
    launch_us = payload["launch"]["tiny_op_per_call_us"]
    m = payload["model"]
    ref_tflops = m["reference_claim"]["tflops"]

    for name in DTYPES:
        g = payload["gemm"][name]
        if g.get("peak_tflops") and name != "fp32":
            g["share_of_nominal_pct"] = round(100 * g["peak_tflops"] / NOMINAL_BF16_TFLOPS, 1)
    payload["gemm"]["fp32"]["nominal_note"] = (
        f"no native fp32 cube path; measured includes the fp32 fallback. TF32 nominal is "
        f"{NOMINAL_TF32_TFLOPS} TFLOPS for the 28-core tier, fp32 measured is "
        f"{payload['gemm']['fp32']['peak_tflops']} TFLOPS = "
        f"{round(100*payload['gemm']['fp32']['peak_tflops']/NOMINAL_TF32_TFLOPS,1)}% of the TF32 number")

    bs = payload["band_split"]["rows"][0]
    launch_share = 100 * 62 * launch_us / 1000 / bs["loop_62_gemms_ms"]

    for row in payload["elementwise"]["rows"]:
        if row["dtype"] == "bf16":
            # launch share of a single elementwise op on this shape
            per = payload["launch"]["activation_op_per_call_us"]["bf16"]
            row["launch_share_pct"] = round(100 * launch_us / per, 1) if per else None
            row["bw_roof_pct"] = round(100 * row["effective_gb_s"] / bw, 1) if bw else None

    ideal_ms = m["forward_flops"] / (peak_bf16 * 1e12) * 1000 if (peak_bf16 and m["forward_flops"]) else None
    launch_bound_ms = (m["flop_counter_ops"] * launch_us / 1000) if m["flop_counter_ops"] else None
    kernels = m.get("device_kernels_per_forward")
    launch_bound_kernel_ms = (kernels * launch_us / 1000) if kernels else None

    payload["conclusions"] = {
        "measured_peak_bf16_tflops": peak_bf16,
        "measured_peak_bf16_shape": payload["gemm"]["bf16"]["peak_shape"],
        "measured_peak_fp32_tflops": peak_fp32,
        "measured_peak_fp32_shape": payload["gemm"]["fp32"]["peak_shape"],
        "measured_peak_fp16_tflops": payload["gemm"]["fp16"]["peak_tflops"],
        "d2d_bandwidth_gb_s": bw,
        "d2d_bandwidth_share_of_nominal_pct": payload["bandwidth"]["share_of_nominal_pct"],
        "launch_overhead_us": launch_us,
        "current_forward_ms": m["forward_median_ms"],
        "current_tflops_measured": m["forward_tflops"],
        "current_tflops_reference_claim": ref_tflops,
        "utilization_pct_measured": round(100 * m["forward_tflops"] / peak_bf16, 2) if (m["forward_tflops"] and peak_bf16) else None,
        "utilization_pct_reference_claim": round(100 * ref_tflops / peak_bf16, 2) if peak_bf16 else None,
        "compute_ideal_ms": round(ideal_ms, 3) if ideal_ms else None,
        "launch_bound_ms_all_aten_ops": round(launch_bound_ms, 3) if launch_bound_ms else None,
        "launch_bound_ms_device_kernels": round(launch_bound_kernel_ms, 3) if launch_bound_kernel_ms else None,
        "launch_bound_device_kernels_x_speedup_ceiling": round(m["forward_median_ms"] / launch_bound_kernel_ms, 2)
        if launch_bound_kernel_ms else None,
        "speedup_ceiling_compute_x": round(m["forward_median_ms"] / ideal_ms, 2) if ideal_ms else None,
        "speedup_ceiling_launch_x": round(m["forward_median_ms"] / launch_bound_ms, 2) if launch_bound_ms else None,
        "device_kernels_per_forward": kernels,
        "band_split_launch_share_pct": round(launch_share, 1),
        "measured_peak_bf16_share_of_nominal_pct": round(100 * peak_bf16 / NOMINAL_BF16_TFLOPS, 1) if peak_bf16 else None,
        "notes": [
            "nominal bf16 = 425 TFLOPS for the 28-core tier of 950PR (486 for the 32-core tier); "
            "utilization is quoted against the measured peak, and the measured/nominal ratio separately",
            "all GB/s numbers count read+write bytes actually touched",
        ],
    }

    # 10x target analysis, launch-bound vs compute-bound
    target_ms = m["forward_median_ms"] / 10
    payload["ten_x"] = {
        "target_ms": round(target_ms, 3),
        "launch_bound_max_ops": int(target_ms * 1000 / launch_us) if launch_us else None,
        "current_ops": m["flop_counter_ops"],
        "ops_reduction_needed_x": round(m["flop_counter_ops"] / (target_ms * 1000 / launch_us), 1)
        if (launch_us and m["flop_counter_ops"]) else None,
        "compute_bound_flops_budget_gflop": round(target_ms * 1e-3 * peak_bf16 * 1e3, 1) if peak_bf16 else None,
        "flops_reduction_needed_x": round(m["forward_flops"] / (target_ms * 1e-3 * peak_bf16 * 1e12), 2)
        if (peak_bf16 and m["forward_flops"]) else None,
        "params_needed_if_only_shrinking": int(m["params"] * (target_ms * 1000 / launch_us) / m["device_kernels_per_forward"])
        if (launch_us and m.get("device_kernels_per_forward")) else None,
        "verdict": ("compute is not the limit at 10x: 187 GFLOP in 1.38 ms is only 36% of peak; "
                    "the limit is the 1910 device kernels x 4.47 us = 8.5 ms launch floor"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["conclusions"], indent=2), flush=True)
    print(json.dumps(payload["ten_x"], indent=2), flush=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
