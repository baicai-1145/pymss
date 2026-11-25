"""NPU graph capture, round 3: the official flow (dedicated stream, empty_cache before
capture_begin), plus the real kernel census of a training step.

Stages (each writes into --json so one flaky stage cannot lose the others):

  infer    whole 8 s forward captured as one graph, replay vs eager, kernel counts
  kernels  one real bs24 bf16 training step under the profiler: device kernels, aten
           dispatches, top-20 by operator, framework-vs-compute split, launch floor
  segs     per-module forward graphs with static buffers + make_graphed_callables (fwd+bwd)
  fixedbf16 whole model on a fixed bf16 dtype (no autocast in/out) and its kernel delta

    python graph_capture3.py --stage infer
    python graph_capture3.py --stage kernels
    python graph_capture3.py --stage segs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LAUNCH_US = 4.47          # measured launch overhead, results/peak_calibration*.json

FW_PREFIXES = ("aten::to", "aten::copy_", "aten::_to_copy", "aten::transpose", "aten::reshape",
               "aten::view", "aten::stack", "aten::empty", "aten::empty_strided", "aten::clone",
               "aten::contiguous", "aten::permute", "aten::expand", "aten::select", "aten::slice",
               "aten::as_strided", "aten::detach", "aten::zeros", "aten::ones", "aten::fill_",
               "aten::cat", "aten::split", "aten::unbind", "aten::squeeze", "aten::unsqueeze",
               "aten::_unsafe_view", "aten::lift_fresh", "aten::lift", "aten::resolve_conj",
               "aten::resolve_neg", "aten::is_contiguous", "aten::set_", "aten::numel",
               "aten::item", "aten::_local_scalar_dense")


FW_KERNEL_PREFIXES = ("Cast_", "Slice_", "Copy_", "Transpose_", "Reshape", "View", "Expand",
                      "BroadcastTo", "ZerosLike", "OnesLike", "Fill", "ScalarToTensor",
                      "Contiguous", "Stack", "Unsqueeze", "Squeeze", "Identity", "Memcpy")


def is_framework_kernel(name: str) -> bool:
    """Device kernels that only move/convert data rather than compute."""
    return name.startswith(FW_KERNEL_PREFIXES)


def is_framework(name: str) -> bool:
    return name.startswith(FW_PREFIXES) or name.startswith("aten::lift_fresh_copy")


# --------------------------------------------------------------------------- helpers
def official_capture(build_fn, warm: int = 3, error_mode: str | None = None, note: str = ""):
    """Capture exactly as docs/zh/developer_notes/npugraph.md describes.

    warmup runs *on the dedicated stream*, then synchronize + empty_cache, then
    capture_begin/end on that same stream, then the default stream waits on it.
    build_fn must allocate its outputs during capture and return them, so that replay
    writes back into the addresses reserved by the graph.
    """
    import torch
    import torch_npu

    s = torch.npu.Stream()
    out = {}
    with torch.npu.stream(s):
        for _ in range(warm):
            out["warm"] = build_fn()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        g = torch.npu.NPUGraph()
        try:
            if error_mode:
                g.capture_begin(capture_error_mode=error_mode)
            else:
                g.capture_begin()
            out["captured"] = build_fn()
            g.capture_end()
        except Exception as exc:  # noqa: BLE001
            return None, None, f"{type(exc).__name__}: {str(exc)[:300]}", note
    torch.npu.current_stream().wait_stream(s)
    return g, out.get("captured"), None, note


def bench(fn, it=20, warm=5):
    import torch

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
    return {"median_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3)}


def kernel_count(fn, n=3):
    """Device kernel count and CPU aten dispatch count per call."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.npu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(n):
            fn()
        torch.npu.synchronize()
    dev = {}
    for evt in prof.events():
        if "NPU" not in str(getattr(evt, "device_type", "")) and \
           "PrivateUse1" not in str(getattr(evt, "device_type", "")):
            continue
        key = evt.key.split("(")[0]
        dev[key] = dev.get(key, 0) + 1
    aten = {}
    for evt in prof.events():
        if "CPU" not in str(getattr(evt, "device_type", "")):
            continue
        key = evt.key.split("(")[0]
        if key.startswith("aten::"):
            aten[key] = aten.get(key, 0) + 1
    return (round(sum(dev.values()) / n, 1), round(sum(aten.values()) / n, 1),
            {k: round(v / n, 1) for k, v in sorted(dev.items(), key=lambda kv: -kv[1])[:8]})


def rel_rms(a, b):
    a = a.float()
    b = b.float()
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12))


def model_and_input(arch: str, batch: int, frames: int):
    import torch

    from arch_factory import build

    model = build(arch).to(torch.device("npu:0")).eval()
    x = torch.randn(batch, 2050, frames, 2, device=torch.device("npu:0"))
    return model, x


# --------------------------------------------------------------------------- stages
def stage_infer(args) -> dict:
    import torch
    import torch_npu  # noqa: F401

    model, x = model_and_input(args.arch, args.batch, args.frames)
    dev = x.device
    res = {"arch": args.arch, "batch": args.batch, "frames": args.frames, "attempts": []}

    def run():
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            return model(x)

    with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
        eager_out = run()
        torch.npu.synchronize()
        res["eager"] = bench(run)
        ek, ed, etop = kernel_count(run)
        res["eager"].update({"device_kernels": ek, "aten_dispatches": ed, "top": etop})

    for mode in (None, "relaxed", "global"):
        g, out, err, _ = official_capture(run, error_mode=mode)
        attempt = {"error_mode": mode, "ok": err is None}
        if err:
            attempt["error"] = err
            res["attempts"].append(attempt)
            print(f"  capture error_mode={mode}: FAILED {err[:120]}", flush=True)
            continue
        g.replay()
        torch.npu.synchronize()
        attempt["parity_rel_rms"] = rel_rms(out, eager_out)
        attempt["replay"] = bench(g.replay)
        rk, rd, rtop = kernel_count(g.replay, n=5)
        attempt.update({"replay_device_kernels": rk, "replay_aten_dispatches": rd, "top": rtop})
        res["attempts"].append(attempt)
        print(f"  capture error_mode={mode}: OK replay {attempt['replay']['median_ms']} ms "
              f"(eager {res['eager']['median_ms']}), parity {attempt['parity_rel_rms']:.2e}, "
              f"kernels {ek} -> {rk}", flush=True)
        break
    ok = [a for a in res["attempts"] if a["ok"]]
    res["ok"] = bool(ok)
    if ok:
        res["speedup"] = round(res["eager"]["median_ms"] / ok[0]["replay"]["median_ms"], 3)
    return res


def _training_step_factory(args):
    """One real bs24 bf16 step, built from the same pieces train.py uses."""
    import torch

    from train import (MusdbChunkDataset, _TGT_WAV, complex_l1, leakage_loss,
                       multi_res_stft_loss_fast, scan_musdb, si_sdr, waveform)
    from arch_factory import build

    dev = torch.device("npu:0")
    ds = MusdbChunkDataset(scan_musdb(Path("/workspace/musdb18hq"), splits=("train",)),
                           args.chunk_size, args.n_fft, args.hop)
    model = build(args.arch).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01)
    window = torch.hann_window(args.n_fft).to(dev, torch.float32)
    batch = [ds[i] for i in range(args.batch)]
    if _TGT_WAV:
        target = torch.stack([b[0] for b in batch]).to(dev)
        mixture = torch.stack([b[1] for b in batch]).to(dev)
        tgt_wav_batch = torch.stack([b[2] for b in batch]).to(dev)
    else:
        target = torch.stack([b[0] for b in batch]).to(dev)
        mixture = torch.stack([b[1] for b in batch]).to(dev)
        tgt_wav_batch = None

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            mask = model(mixture)
            mix_c = torch.view_as_complex(mixture.contiguous())
            tgt_c = torch.view_as_complex(target.contiguous())
            pred_c = mix_c * torch.view_as_complex(mask.float().contiguous())
            loss_spec = complex_l1(pred_c, tgt_c)
            pred_wav = waveform(pred_c, 2, args.n_fft, args.hop, args.chunk_size, window)
            tgt_wav = tgt_wav_batch if tgt_wav_batch is not None else waveform(
                tgt_c, 2, args.n_fft, args.hop, args.chunk_size, window)
            loss = (loss_spec + 0.5 * multi_res_stft_loss_fast(pred_wav, tgt_wav)
                    - 0.1 * si_sdr(pred_wav, tgt_wav).mean() / 20.0
                    + 0.1 * leakage_loss(pred_wav, tgt_wav))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    return model, step


def stage_kernels(args) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    model, step = _training_step_factory(args)
    for _ in range(3):
        step()
    torch.npu.synchronize()
    t = bench(step, it=6, warm=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        step()
        torch.npu.synchronize()
    dev, aten = {}, {}
    for evt in prof.events():
        dt = str(getattr(evt, "device_type", ""))
        key = evt.key.split("(")[0]
        if "NPU" in dt or "PrivateUse1" in dt:
            dev[key] = dev.get(key, 0) + 1
        elif "CPU" in dt and key.startswith("aten::"):
            aten[key] = aten.get(key, 0) + 1
    n_dev, n_aten = sum(dev.values()), sum(aten.values())
    fw_dev = sum(v for k, v in dev.items() if is_framework_kernel(k))
    cast_dev = sum(v for k, v in dev.items() if k.startswith("Cast_"))
    top = sorted(dev.items(), key=lambda kv: -kv[1])[:20]
    aten_top = sorted(aten.items(), key=lambda kv: -kv[1])[:20]
    floor_ms = n_dev * LAUNCH_US / 1e3
    return {
        "step_ms": t["median_ms"],
        "device_kernels": n_dev, "aten_dispatches": n_aten,
        "device_kernels_per_aten_dispatch": round(n_dev / max(n_aten, 1), 2),
        "top20_device_kernels": [{"kernel": k, "count": v, "share": round(100 * v / n_dev, 1),
                                  "kind": "framework" if is_framework_kernel(k) else "compute"}
                                 for k, v in top],
        "top20_aten": [{"op": k, "count": v, "kind": "framework" if is_framework(k) else "compute"}
                       for k, v in aten_top],
        "framework_device_kernels": fw_dev,
        "cast_device_kernels": cast_dev,
        "cast_share_pct": round(100 * cast_dev / max(n_dev, 1), 1),
        "framework_share_pct": round(100 * fw_dev / max(n_dev, 1), 1),
        "launch_floor_ms": round(floor_ms, 1),
        "launch_floor_pct_of_step": round(100 * floor_ms / t["median_ms"], 1),
    }


def stage_segs(args) -> dict:
    """Per-module forward graphs with static buffers, then make_graphed_callables for fwd+bwd."""
    import torch
    import torch_npu

    from arch_factory import build

    dev = torch.device("npu:0")
    model = build(args.arch).to(dev).eval()
    x = torch.randn(args.batch, 2050, args.frames, 2, device=dev)
    res = {"arch": args.arch, "segments": [], "graphed_callables": {}}

    # shapes each segment sees, taken from a traced eager pass
    shapes = {}

    def hook(name):
        def fn(mod, inp, out):
            i = inp[0] if isinstance(inp, tuple) else inp
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(i) and torch.is_tensor(o):
                shapes[name] = (tuple(i.shape), tuple(o.shape))
        return fn

    # BandUp takes (x, skip), so it cannot be captured as a single-input module; the decoder
    # blocks dec16/dec32/dec62 are captured instead, and the up-stage is timed eagerly.
    names = ["band_split", "enc62", "down62", "enc32", "down32", "enc16", "down16",
             "dec16", "dec32", "dec62"]
    handles = [getattr(model, n).register_forward_hook(hook(n)) for n in names if hasattr(model, n)]
    with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
        model(x)
    for h in handles:
        h.remove()

    for name, (ishape, oshape) in shapes.items():
        mod = getattr(model, name)
        static_in = torch.randn(*ishape, device=dev, dtype=torch.bfloat16)
        cached = {}
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            # one eager copy of the module's own weights in bf16 so capture sees no autocast
            def run(mod=mod, static_in=static_in, cached=cached):
                cached["out"] = mod(static_in)
                return cached["out"]

            with torch.npu.amp.autocast(enabled=False):
                mod_bf16 = mod.to(torch.bfloat16)
                ref = run().detach().clone()
                g, out, err, _ = official_capture(lambda: mod_bf16(static_in))
            if err:
                res["segments"].append({"name": name, "ok": False, "error": err[:200],
                                        "in": list(ishape), "out": list(oshape)})
                mod.to(torch.float32)
                print(f"  seg {name:12s} FAILED {err[:100]}", flush=True)
                continue
            with torch.npu.amp.autocast(enabled=False):
                mod_bf16(static_in)       # eager reference in the same dtype
            eager = mod_bf16(static_in).detach()
            t_eager = bench(lambda: mod_bf16(static_in))
            g.replay()
            torch.npu.synchronize()
            parity = rel_rms(out, eager)
            t_replay = bench(g.replay)
            res["segments"].append({"name": name, "ok": True, "in": list(ishape), "out": list(oshape),
                                    "eager_ms": t_eager["median_ms"], "replay_ms": t_replay["median_ms"],
                                    "speedup": round(t_eager["median_ms"] / t_replay["median_ms"], 3),
                                    "parity_rel_rms": parity})
            mod.to(torch.float32)
            print(f"  seg {name:12s} eager {t_eager['median_ms']:7.2f} -> replay "
                  f"{t_replay['median_ms']:7.2f} ms  parity {parity:.2e}", flush=True)

    # fwd+bwd via the official helper
    try:
        m2 = build(args.arch).to(dev).train()
        sample = (x,)
        gm = torch_npu.npu.make_graphed_callables(m2, sample)
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            o_eager = m2(x)
            o_graph = gm(x)
        parity = rel_rms(o_graph, o_eager)
        t_eager = bench(lambda: m2(x))
        t_graph = bench(lambda: gm(x))
        res["graphed_callables"] = {"ok": True, "parity_rel_rms": parity,
                                    "eager_ms": t_eager["median_ms"],
                                    "graph_ms": t_graph["median_ms"],
                                    "forward_speedup": round(t_eager["median_ms"] / t_graph["median_ms"], 3)}
        print(f"  make_graphed_callables fwd: {t_eager['median_ms']:.2f} -> "
              f"{t_graph['median_ms']:.2f} ms parity {parity:.2e}", flush=True)
    except Exception as exc:  # noqa: BLE001
        res["graphed_callables"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"  make_graphed_callables FAILED: {str(exc)[:160]}", flush=True)
    return res


def stage_fixedbf16(args) -> dict:
    """Whole model in a fixed bf16 dtype: no autocast churn, one cast at the boundary."""
    import torch
    import torch_npu  # noqa: F401

    from arch_factory import build

    dev = torch.device("npu:0")
    x = torch.randn(args.batch, 2050, args.frames, 2, device=dev)
    res = {}
    m32 = build(args.arch).to(dev).eval()
    with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
        ref = m32(x)
        t_auto = bench(lambda: m32(x))
        k_auto, d_auto, _ = kernel_count(lambda: m32(x))
    m16 = m32.to(torch.bfloat16)          # same weights, fixed dtype - no autocast churn
    with torch.no_grad():
        xb = x.to(torch.bfloat16)
        out16 = m16(xb)
        t_fix = bench(lambda: m16(xb))
        k_fix, d_fix, _ = kernel_count(lambda: m16(xb))
    res["autocast"] = {"median_ms": t_auto["median_ms"], "device_kernels": k_auto, "aten": d_auto}
    res["fixed_bf16"] = {"median_ms": t_fix["median_ms"], "device_kernels": k_fix, "aten": d_fix,
                         "parity_vs_autocast_rel_rms": rel_rms(out16, ref)}
    res["delta_ms"] = round(t_auto["median_ms"] - t_fix["median_ms"], 3)
    res["delta_kernels"] = round(k_auto - k_fix, 1)
    print(f"  autocast {t_auto['median_ms']:.2f} ms / {k_auto} kernels -> fixed bf16 "
          f"{t_fix['median_ms']:.2f} ms / {k_fix} kernels (parity "
          f"{res['fixed_bf16']['parity_vs_autocast_rel_rms']:.2e})", flush=True)
    return res


STAGES = {"infer": stage_infer, "kernels": stage_kernels, "segs": stage_segs,
          "fixedbf16": stage_fixedbf16}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--arch", default="v1", choices=("v1", "v3"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--json", type=Path, default=Path("results/graph_capture3.json"))
    args = parser.parse_args(argv)
    if args.stage == "kernels":
        args.batch = 24
    print(f"[{args.stage}] arch={args.arch} batch={args.batch} frames={args.frames}", flush=True)
    out = STAGES[args.stage](args)
    payload = {}
    if args.json.exists():
        try:
            payload = json.loads(args.json.read_text())
        except Exception:  # noqa: BLE001
            payload = {}
    payload[args.stage] = out
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(out, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
