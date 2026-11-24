"""Phase E: layer-by-layer numerical alignment NPU(fp32/bf16) vs CPU fp32 reference.

The CUDA G4 reference runs the same `pymss_core` code path. No CUDA device
exists on this box, so CPU fp32 is used as the transportable reference and the
results are labelled NPU-vs-CPU.

Both runs use the *same* op recipe: BandSplit/MaskEstimator grouped-forward is
disabled so CPU and NPU execute identical matmuls (grouped bmm vs per-band
linear differ in reduction order and would pollute the comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402


def set_npu_precision(strict_fp32: bool) -> dict:
    """torch_npu runs conv in HF32 by default (TF32-like); fp32 equivalence needs it off."""
    state = {}
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch.npu, "conv"):
            state["conv.allow_hf32_before"] = bool(torch.npu.conv.allow_hf32)
            torch.npu.conv.allow_hf32 = not strict_fp32
            state["conv.allow_hf32_after"] = bool(torch.npu.conv.allow_hf32)
        if hasattr(torch.npu, "matmul"):
            state["matmul.allow_hf32_before"] = bool(torch.npu.matmul.allow_hf32)
            torch.npu.matmul.allow_hf32 = False
            state["matmul.allow_hf32_after"] = bool(torch.npu.matmul.allow_hf32)
    except Exception as exc:
        state["error"] = repr(exc)
    return state


def force_flat_paths(model) -> None:
    for name in ("band_split", "_estimate_masks"):
        mod = getattr(model, name, None)
        if mod is not None and hasattr(mod, "use_grouped_forward"):
            mod.use_grouped_forward = False


def capture_run(model, spec, device, precision, framework="torch") -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    caps: list[tuple[str, torch.Tensor]] = []
    handles = []
    for name, child in model.named_modules():
        if not name or not isinstance(child, torch.nn.Module):
            continue

        def hook(_mod, _inp, out, _name=name):
            if isinstance(out, torch.Tensor):
                caps.append((_name, out.detach().float().cpu().clone()))

        handles.append(child.register_forward_hook(hook))

    model.to(device).eval()
    x = spec.to(device)
    try:
        with torch.no_grad():
            if precision == "bf16":
                with torch.npu.amp.autocast(dtype=torch.bfloat16):
                    out = model(x)
            else:
                out = model(x)
    finally:
        for handle in handles:
            handle.remove()
    if device.type == "npu":
        torch.npu.synchronize()
    return out.detach().float().cpu(), caps


def compare(ref: list, cand: list) -> list[dict]:
    ref_map = {name: t for name, t in ref}
    cand_map = {name: t for name, t in cand}
    rows = []
    for name, ta in ref_map.items():
        tb = cand_map.get(name)
        if tb is None:
            continue
        if ta.shape != tb.shape:
            rows.append({"layer": name, "shape": f"{tuple(ta.shape)} vs {tuple(tb.shape)}"})
            continue
        diff = (ta - tb).abs()
        rms_ref = ta.pow(2).mean().sqrt()
        rows.append({
            "layer": name,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rel_rms": float(diff.pow(2).mean().sqrt() / rms_ref.clamp_min(1e-12)),
            "abs_mean_ref": float(ta.abs().mean()),
            "numel": int(ta.numel()),
        })
    return rows


def summarize(rows: list[dict], key: str = "rel_rms") -> dict:
    valid = [r for r in rows if key in r]
    worst = sorted(valid, key=lambda r: r[key], reverse=True)[:8]
    return {
        "layers": len(valid),
        "worst_max_abs": max((r["max_abs"] for r in valid), default=None),
        "worst_rel_rms": max((r["rel_rms"] for r in valid), default=None),
        "median_rel_rms": sorted(r["rel_rms"] for r in valid)[len(valid) // 2] if valid else None,
        "worst_layers": worst,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-hf32", action="store_true", help="keep torch_npu's default HF32 conv path")
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/align_report.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    precision_state = set_npu_precision(strict_fp32=not args.allow_hf32)
    print("npu precision:", json.dumps(precision_state), flush=True)

    torch.manual_seed(args.seed)
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg)
    force_flat_paths(model)
    n_params = sum(p.numel() for p in model.parameters())
    f = args.n_fft // 2 + 1
    spec = torch.randn(1, f * cfg.audio_channels, args.frames, 2) * 0.1
    print(f"params={n_params} freq_bins={f} channels={cfg.audio_channels} frames={args.frames}", flush=True)

    out_cpu, cap_cpu = capture_run(model, spec, torch.device("cpu"), "32")
    out_npu, cap_npu = capture_run(model, spec, torch.device("npu:0"), "32")
    rows32 = compare(cap_cpu, cap_npu)
    payload = {
        "params": n_params,
        "frames": args.frames,
        "npu_precision": precision_state,
        "fp32_npu_vs_cpu": {
            "output_max_abs": float((out_npu - out_cpu).abs().max()),
            "output_rel_rms": float((out_npu - out_cpu).pow(2).mean().sqrt() / out_cpu.pow(2).mean().sqrt().clamp_min(1e-12)),
            "layers": summarize(rows32),
            "rows": rows32,
        },
    }
    print(json.dumps(payload["fp32_npu_vs_cpu"]["layers"], indent=2), flush=True)

    if args.bf16:
        out_bf16, cap_bf16 = capture_run(model, spec, torch.device("npu:0"), "bf16")
        rows_bf = compare(cap_cpu, cap_bf16)
        payload["bf16_npu_vs_fp32_cpu"] = {
            "output_max_abs": float((out_bf16 - out_cpu).abs().max()),
            "output_rel_rms": float((out_bf16 - out_cpu).pow(2).mean().sqrt() / out_cpu.pow(2).mean().sqrt().clamp_min(1e-12)),
            "layers": summarize(rows_bf),
            "rows": rows_bf,
        }
        print(json.dumps(payload["bf16_npu_vs_fp32_cpu"]["layers"], indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
