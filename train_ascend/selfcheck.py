"""Phase C self-check: one full 8 s forward + backward on CPU and on NPU.

Checks output waveform shape (1, 2, 352800), finiteness of outputs, and that
every trainable parameter receives a finite, non-zero-less gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model, param_count  # noqa: E402


def run(device: torch.device, frames: int, n_fft: int, hop: int, length: int, channels: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = FreqUNetConfig(n_fft=n_fft)
    model = build_model(cfg).to(device).train()
    n_params = param_count(model)
    f = n_fft // 2 + 1
    spec = torch.randn(1, f * channels, frames, 2, device=device) * 0.1

    t0 = time.time()
    mask = model(spec)
    mix_c = torch.view_as_complex(spec.contiguous())
    pred_c = mix_c * torch.view_as_complex(mask.float().contiguous())
    from pymss_core.npu_fft import istft_npu

    window = torch.hann_window(n_fft, device=device, dtype=torch.float32)
    flat = pred_c.reshape(1, f, channels, pred_c.shape[-1]).permute(0, 2, 1, 3).reshape(channels, f, pred_c.shape[-1])
    wav = istft_npu(flat, n_fft=n_fft, hop_length=hop, window=window, length=length, return_complex=False)
    if device.type == "npu":
        torch.npu.synchronize()
    fwd_s = time.time() - t0

    loss = pred_c.abs().mean()
    t1 = time.time()
    loss.backward()
    if device.type == "npu":
        torch.npu.synchronize()
    bwd_s = time.time() - t1

    no_grad, nonfinite, zero = [], [], 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
        elif p.grad.abs().max() == 0:
            zero += 1

    params = list(model.parameters())
    return {
        "device": str(device),
        "params": n_params,
        "mask_shape": list(mask.shape),
        "mask_finite": bool(torch.isfinite(mask).all()),
        "waveform_shape": list(wav.shape),
        "waveform_finite": bool(torch.isfinite(wav).all()),
        "waveform_absmax": float(wav.abs().max()),
        "loss": float(loss),
        "params_without_grad": no_grad[:10],
        "params_without_grad_count": len(no_grad),
        "params_nonfinite_grad": nonfinite[:10],
        "params_nonfinite_grad_count": len(nonfinite),
        "params_zero_grad_count": zero,
        "param_groups": len(params),
        "forward_s": round(fwd_s, 3),
        "backward_s": round(bwd_s, 3),
        "peak_mem_mb": round(torch.npu.max_memory_allocated() / 2**20, 1) if device.type == "npu" else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--length", type=int, default=352800)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/selfcheck.json"))
    args = parser.parse_args(argv)

    frames = args.length // args.hop + 1
    results = {"frames": frames}
    results["cpu"] = run(torch.device("cpu"), frames, args.n_fft, args.hop, args.length, args.channels)
    print(json.dumps(results["cpu"], indent=2), flush=True)

    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            results["npu"] = run(torch.device("npu:0"), frames, args.n_fft, args.hop, args.length, args.channels)
            print(json.dumps(results["npu"], indent=2), flush=True)
    except Exception as exc:  # pragma: no cover
        results["npu_error"] = repr(exc)
        print("NPU self-check failed:", repr(exc), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
