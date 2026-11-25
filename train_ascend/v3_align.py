"""Layer-by-layer NPU/CPU alignment for freq_unet_v3 (fp32, allow_hf32 off) + bf16 table.

Same protocol as align_check.py used for v1: identical weights on both devices, one forward
pass over the same input, hooks on every named submodule, report max_abs / rel_rms per layer.

    python v3_align.py --frames 64 --out results/v3_align.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from freq_unet_v3 import FreqUNetV3Config, build_model


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).pow(2).mean().sqrt()
    den = b.pow(2).mean().sqrt().clamp_min(1e-12)
    return float(num / den)


def capture(model, stft, dtype=None) -> dict:
    """Run one forward and collect every 4-D activation, keyed by module name."""
    out = {}

    def hook(name):
        def fn(_m, _i, o):
            if torch.is_tensor(o) and o.dim() == 4:
                out[name] = o.detach().float().cpu()
        return fn

    handles = [m.register_forward_hook(hook(name)) for name, m in model.named_modules() if name]
    try:
        ctx = torch.autocast("npu", dtype=dtype) if dtype is not None else torch.no_grad()
        with ctx:
            mask = model(stft)
    finally:
        for h in handles:
            h.remove()
    out["__mask__"] = mask.detach().float().cpu()
    return out


def compare(a: dict, b: dict) -> dict:
    rows = []
    for k in b:
        if k not in a:
            continue
        ref = b[k]
        got = a[k]
        if ref.shape != got.shape:
            rows.append({"layer": k, "shape_mismatch": [list(got.shape), list(ref.shape)]})
            continue
        rows.append({"layer": k, "max_abs": float((got - ref).abs().max()), "rel_rms": rel_rms(got, ref)})
    worst = max(rows, key=lambda r: r.get("max_abs", float("inf")))
    return {"layers": len(rows), "worst": worst,
            "worst_max_abs": worst["max_abs"], "worst_rel": worst["rel_rms"], "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("results/v3_align.json"))
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    cfg = FreqUNetV3Config()
    torch.manual_seed(0)
    cpu = build_model(cfg).eval()
    npu = build_model(cfg).eval()
    npu.load_state_dict(cpu.state_dict())
    npu = npu.npu()
    torch.npu.conv.allow_hf32 = False  # fp32 reference must not silently go hf32

    n_freq = cfg.n_fft // 2 + 1
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, n_freq * cfg.audio_channels, args.frames, 2, generator=g)

    report = {"arch": "v3", "frames": args.frames, "params": int(sum(p.numel() for p in npu.parameters()))}
    if not args.skip_cpu:
        with torch.no_grad():
            cpu_out = capture(cpu, x)
        npu_out = capture(npu, x.to("npu:0"))
        report["fp32_npu_vs_cpu"] = compare(npu_out, cpu_out)
        print(f"fp32: layers {report['fp32_npu_vs_cpu']['layers']}, "
              f"worst max_abs {report['fp32_npu_vs_cpu']['worst_max_abs']:.3e} "
              f"rel_rms {report['fp32_npu_vs_cpu']['worst_rel']:.3e} "
              f"({report['fp32_npu_vs_cpu']['worst']['layer']})", flush=True)

    npu_fp32 = capture(npu, x.to("npu:0")) if args.skip_cpu else npu_out
    npu_bf16 = capture(npu, x.to("npu:0"), dtype=torch.bfloat16)
    report["bf16_npu_vs_fp32_npu"] = compare(npu_bf16, npu_fp32)
    print(f"bf16: worst max_abs {report['bf16_npu_vs_fp32_npu']['worst_max_abs']:.3e} "
          f"rel_rms {report['bf16_npu_vs_fp32_npu']['worst_rel']:.3e} "
          f"({report['bf16_npu_vs_fp32_npu']['worst']['layer']})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
