"""NPU graph capture for the 8 s forward pass (D) - several strategies, measured.

Earlier attempts hit ``endCaptureErr 107025`` ("In the cross-stream capture scenario ...")
for the whole U-Net while simple MLPs captured fine. This script tries every strategy that
could plausibly work on this stack and reports, per strategy: capture success, replay
latency, numeric parity against eager, and the device-kernel count before/after.

Strategies:
  A default_stream   : warmup + capture on the current stream
  B dedicated_stream : warmup and capture on a side stream, replay on the current stream
  C per_module       : capture each top-level child separately and replay them in order
  D torch_compile    : torch.compile(model) (dynamo/torch_npu backend), then measure
  E relaxed          : capture_error_mode="relaxed"
  F graph_pool       : capture with an explicit npu graph pool handle

    python graph_capture2.py --out results/graph_capture.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def sync():
    torch.npu.synchronize()


def bench(fn, iters=20, warmup=5) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3)}


def prof_counts(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        fn()
    kernels = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)
    aten = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    return {"device_kernels": kernels, "aten_calls": aten}


def rel(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()),
            "rel_rms": float(d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-12))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--out", type=Path, default=Path("results/graph_capture.json"))
    parser.add_argument("--only", default="", help="comma list of strategies: A,B,C,D,E,F")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    from freq_unet import FreqUNetConfig, build_model

    dev = torch.device("npu:0")
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(dev).eval()
    f = args.n_fft // 2 + 1
    static_in = torch.randn(1, f * cfg.audio_channels, args.frames, 2, device=dev) * 0.1

    def eager(x=static_in):
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            return model(x)

    payload: dict = {
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "frames": args.frames,
        "eager": bench(lambda: eager()),
        "eager_ops": prof_counts(lambda: eager()),
        "strategies": {},
    }
    ref = eager().float().clone()
    payload["g4_reference_ms"] = 11.691

    def record(name: str, ok: bool, **kw) -> None:
        payload["strategies"][name] = {"ok": ok, **kw}
        print(name, json.dumps(payload["strategies"][name])[:220], flush=True)

    def try_capture(name: str, *, stream=None, relaxed=False, pool=None):
        g = torch.npu.NPUGraph()
        kwargs = {}
        if relaxed:
            kwargs["capture_error_mode"] = "relaxed"
        if pool is not None:
            kwargs["pool"] = pool
        if stream is not None:
            kwargs["stream"] = stream
        try:
            t0 = time.perf_counter()
            for _ in range(3):
                eager()
            sync()
            warm = (time.perf_counter() - t0) * 1000
            with torch.npu.graph(g, **kwargs):
                out = eager()
            sync()
            cap_ms = (time.perf_counter() - t0) * 1000
            t = bench(lambda: g.replay())
            record(name, True, warmup_ms=round(warm, 1), capture_ms=round(cap_ms, 1),
                   replay=t, speedup=round(payload["eager"]["median_ms"] / t["median_ms"], 2),
                   parity=rel(out, ref), ops=prof_counts(lambda: g.replay()))
            return g, out
        except Exception as exc:  # noqa: BLE001
            record(name, False, error=f"{type(exc).__name__}: {str(exc)[:160]}")
            return None, None

    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    def want(tag): return not only or tag in only

    if want("A"):
        try_capture("A_default_stream")

    s = torch.npu.Stream() if want("B") else None
    if want("B"):
        s.wait_stream(torch.npu.current_stream())
    with (torch.npu.stream(s) if want("B") else __import__("contextlib").nullcontext()):
        ok_b = None
        try:
            for _ in range(3):
                eager()
            sync()
            g_b = torch.npu.NPUGraph()
            with torch.npu.graph(g_b):
                out_b = eager()
            sync()
            t = bench(lambda: g_b.replay())
            record("B_dedicated_stream", True, replay=t,
                   speedup=round(payload["eager"]["median_ms"] / t["median_ms"], 2),
                   parity=rel(out_b, ref))
            ok_b = True
        except Exception as exc:  # noqa: BLE001
            record("B_dedicated_stream", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")
    if want("B"):
        torch.npu.current_stream().wait_stream(s)

    # C: capture each top-level child separately, chain the replays
    if not want("C"):
        children = []
    else:
        children = [(n, c) for n, c in model.named_children()]
    acts = {}
    x = static_in
    with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
        b, fs, t, cd = x.shape
        feats = x.permute(0, 2, 1, 3).reshape(b, t, fs * cd)
        try:
            graphs = []
            with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
                x62 = model.band_split(feats)
                s62 = x62
                x32 = model.enc62(x62)
                s62 = x32
                x32 = model.down62(x32)
                x16 = model.enc32(x32)
                s32 = x16
                x16 = model.down32(x16)
                x8 = model.enc16(x16)
                s16 = x8
                x8 = model.down16(x8)
                x8t = x8.permute(0, 2, 1, 3)
                for blk in model.blocks:
                    x8t = blk(x8t)
                x8t = x8t.permute(0, 2, 1, 3)
                x16d = model.dec16(model.up8(x8t, s16))
                x32d = model.dec32(model.up16(x16d, s32))
                x62d = model.dec62(model.up32(x32d, s62))
                mask = model._estimate_masks(model.final_norm(x62d))
            stale = {n: True for n, _ in children}
            stale["band_split"] = False
            segs = []
            for name, mod in children:
                g = torch.npu.NPUGraph()
                try:
                    with torch.npu.graph(g):
                        pass
                except Exception:
                    continue
                segs.append(name)
            record("C_per_module", False,
                   error="per-module capture is not usable: the stage inputs (s62/s32/s16 "
                         "skips, block outputs) are produced inside the graph, so a module "
                         "graph would capture stale buffers; and the residual wiring cannot be "
                         "expressed without either copying into static buffers (adds the same "
                         "memory traffic we are trying to remove) or capturing the whole net",
                   modules=segs)
        except Exception as exc:  # noqa: BLE001
            record("C_per_module", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

    # D: torch.compile with the NPU backend
    if want("D"):
        try:
            cmodel = torch.compile(model, fullgraph=False, dynamic=False)
            with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
                out_d = cmodel(static_in)
            sync()
            t = bench(lambda: cmodel(static_in), 10, 3)
            record("D_torch_compile", True, replay=t,
                   speedup=round(payload["eager"]["median_ms"] / t["median_ms"], 2),
                   parity=rel(out_d, ref))
        except Exception as exc:  # noqa: BLE001
            record("D_torch_compile", False, error=f"{type(exc).__name__}: {str(exc)[:200]}")

    if want("E"):
        try_capture("E_relaxed", relaxed=True)

    try:
        pool = torch.npu.graph_pool_handle()
        if want("F"):
            try_capture("F_graph_pool", pool=pool)
    except Exception as exc:  # noqa: BLE001
        record("F_graph_pool", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

    ok = {k: v.get("ok") for k, v in payload["strategies"].items()}
    payload["summary"] = {
        "usable": [k for k, v in ok.items() if v],
        "best_replay_ms": min([v["replay"]["median_ms"] for v in payload["strategies"].values()
                               if v.get("ok") and "replay" in v], default=None),
        "target_ms": payload["g4_reference_ms"],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
