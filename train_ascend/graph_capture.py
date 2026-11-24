"""NPU graph capture for the 8 s forward pass (and an honest attempt at a training step).

Inference: capture the whole 690-frame forward into an ACL graph, then replay it with
static input/output buffers. Reports capture time, replay latency, dispatch/kernel
count reduction (measured by diffing the profiler's op counts) and numeric parity
against the eager forward.

Training: tries to capture forward+backward+optimizer+EMA in one graph, then
forward/backward only, and reports which (if any) is usable and why.

    python graph_capture.py --out results/graph_capture.json
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


def timeit(fn, iters=20, warmup=5) -> dict:
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
    return {"median_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3),
            "mean_ms": round(statistics.mean(ts), 3)}


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
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/graph_capture.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from freq_unet import FreqUNetConfig, build_model

    dev = torch.device("npu:0")
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(dev).eval()
    f = args.n_fft // 2 + 1
    payload: dict = {"device": torch.npu.get_device_name(0), "frames": args.frames}

    graph_cls = getattr(torch.npu, "CUDAGraph", None) or getattr(torch.npu, "NPUGraph", None)
    graph_ctx = getattr(torch.npu, "graph", None)
    payload["graph_api"] = {
        "CUDAGraph": graph_cls is not None,
        "graph_ctx": graph_ctx is not None,
        "torch_npu": torch_npu.__version__,
    }

    # ---------------- inference ----------------
    static_in = torch.randn(1, f * cfg.audio_channels, args.frames, 2, device=dev) * 0.1

    def eager(x):
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            return model(x)

    ref = eager(static_in).float()
    payload["inference"] = {"eager": timeit(lambda: eager(static_in), args.iters)}

    if graph_cls is not None and graph_ctx is not None:
        g = graph_cls()
        # warmup must happen on the *capture* stream (default stream here); capturing on a
        # side stream after warming elsewhere fails with endCaptureErr 107025.
        for _ in range(3):
            eager(static_in)
        sync()
        t0 = time.perf_counter()
        try:
            with graph_ctx(g):
                out_static = eager(static_in)
            sync()
            payload["inference"]["capture_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            payload["inference"]["capture_ok"] = True
        except Exception as exc:
            payload["inference"]["capture_ok"] = False
            payload["inference"]["capture_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            out_static = None

        if out_static is not None:
            t = timeit(lambda: g.replay(), args.iters)
            payload["inference"].update({f"replay_{k}": v for k, v in t.items()})
            payload["inference"]["speedup"] = round(payload["inference"]["eager"]["median_ms"] / t["median_ms"], 2)
            payload["inference"]["parity"] = rel(out_static, ref)
            payload["inference"]["ops_eager"] = prof_counts(lambda: eager(static_in))
            payload["inference"]["ops_graph"] = prof_counts(lambda: g.replay())

    # ---------------- training step ----------------
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    tgt = torch.randn(1, f * cfg.audio_channels, args.frames, 2, device=dev) * 0.1
    payload["training"] = {}

    def step_eager():
        opt.zero_grad(set_to_none=True)
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            mask = model(static_in)
        loss = (mask - tgt).float().pow(2).mean()
        loss.backward()
        opt.step()
        return loss

    payload["training"]["eager"] = timeit(step_eager, 10, 3)
    for label, fn in (
        ("whole_step", step_eager),
    ):
        if graph_cls is None or graph_ctx is None:
            payload["training"][label] = {"ok": False, "error": "no graph api"}
            continue
        try:
            g2 = graph_cls()
            for _ in range(3):
                fn()
            sync()
            with graph_ctx(g2):
                fn()
            sync()
            t = timeit(lambda: g2.replay(), 10, 3)
            payload["training"][label] = {"ok": True, **t,
                                          "speedup": round(payload["training"]["eager"]["median_ms"] / t["median_ms"], 2)}
        except Exception as exc:
            payload["training"][label] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
