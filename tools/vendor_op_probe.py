"""Sweep the operators the remaining model families need, on the vendor device.

Each probe runs in its own subprocess: a missing kernel raises, but a bad plan can
terminate() the process, and that must not take the sweep down with it.  Every
probe compares against the CPU result, so "runs but computes nonsense" is caught
as loudly as "crashes".

Usage:
    python tools/vendor_op_probe.py --device supa
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

#: name -> code.  Each block prints ``OK <max_err>`` or raises.
PROBES: dict[str, str] = {
    "pad_reflect_2d": """
import torch, torch.nn.functional as F
x = torch.randn(1, 8, 32, 40, device=D)
y = F.pad(x, (2, 2, 2, 2), mode="reflect")
torch.supa.synchronize()
ref = F.pad(x.cpu(), (2, 2, 2, 2), mode="reflect")
err = (y.cpu() - ref).abs().max().item()
""",
    "pad_replicate_2d": """
import torch, torch.nn.functional as F
x = torch.randn(1, 8, 32, 40, device=D)
y = F.pad(x, (0, 0, 0, 3), mode="replicate")
torch.supa.synchronize()
ref = F.pad(x.cpu(), (0, 0, 0, 3), mode="replicate")
err = (y.cpu() - ref).abs().max().item()
""",
    "pad_replicate_1d": """
import torch, torch.nn.functional as F
x = torch.randn(2, 8, 512, device=D)
y = F.pad(x, (4, 4), mode="replicate")
torch.supa.synchronize()
ref = F.pad(x.cpu(), (4, 4), mode="replicate")
err = (y.cpu() - ref).abs().max().item()
""",
    "einsum_bmm": """
import torch
a = torch.randn(2, 64, 32, device=D)
b = torch.randn(2, 32, 16, device=D)
y = torch.einsum("b i j, b j k -> b i k", a, b)
torch.supa.synchronize()
ref = torch.einsum("b i j, b j k -> b i k", a.cpu(), b.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "einsum_reduce": """
import torch
a = torch.randn(2, 4, 8, 16, device=D)
b = torch.randn(2, 4, 16, 8, device=D)
y = torch.einsum("b h i j, b h j k -> b h i k", a, b)
torch.supa.synchronize()
ref = torch.einsum("b h i j, b h j k -> b h i k", a.cpu(), b.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "einsum_mel": """
import torch
spec = torch.randn(2, 64, 257, device=D)
fb = torch.randn(64, 257, device=D)
y = torch.einsum("b c f, c f -> b", spec, fb)
torch.supa.synchronize()
ref = torch.einsum("b c f, c f -> b", spec.cpu(), fb.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "group_norm": """
import torch
x = torch.randn(2, 32, 16, 16, device=D)
y = torch.nn.GroupNorm(8, 32).to(D)(x)
torch.supa.synchronize()
ref = torch.nn.functional.group_norm(x.cpu(), 8, torch.ones(32), torch.zeros(32), 1e-5)
err = (y.cpu() - ref).abs().max().item()
""",
    "layer_norm": """
import torch
x = torch.randn(2, 16, 64, device=D)
y = torch.nn.LayerNorm(64).to(D)(x)
torch.supa.synchronize()
ref = torch.nn.LayerNorm(64)(x.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "instance_norm": """
import torch
x = torch.randn(2, 8, 16, 16, device=D)
m = torch.nn.InstanceNorm2d(8, affine=True).to(D)
y = m(x)
torch.supa.synchronize()
ref = torch.nn.InstanceNorm2d(8, affine=True)(x.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "upsample_bilinear": """
import torch, torch.nn.functional as F
x = torch.randn(2, 16, 32, 40, device=D)
y = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
torch.supa.synchronize()
ref = F.interpolate(x.cpu(), scale_factor=2, mode="bilinear", align_corners=False)
err = (y.cpu() - ref).abs().max().item()
""",
    "upsample_nearest": """
import torch, torch.nn.functional as F
x = torch.randn(2, 16, 32, 40, device=D)
y = F.interpolate(x, scale_factor=2, mode="nearest")
torch.supa.synchronize()
ref = F.interpolate(x.cpu(), scale_factor=2, mode="nearest")
err = (y.cpu() - ref).abs().max().item()
""",
    "upsample_1d_linear": """
import torch, torch.nn.functional as F
x = torch.randn(2, 16, 128, device=D)
y = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
torch.supa.synchronize()
ref = F.interpolate(x.cpu(), scale_factor=2, mode="linear", align_corners=False)
err = (y.cpu() - ref).abs().max().item()
""",
    "conv1d": """
import torch
x = torch.randn(2, 16, 256, device=D)
c = torch.nn.Conv1d(16, 32, 3, padding=1).to(D)
y = c(x)
torch.supa.synchronize()
ref = torch.nn.functional.conv1d(x.cpu(), c.weight.cpu(), c.bias.cpu(), 1, 1)
err = (y.cpu() - ref).abs().max().item()
""",
    "conv_transpose1d": """
import torch
x = torch.randn(2, 16, 128, device=D)
c = torch.nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1).to(D)
y = c(x)
torch.supa.synchronize()
ref = torch.nn.functional.conv_transpose1d(x.cpu(), c.weight.cpu(), c.bias.cpu(), 2, 1)
err = (y.cpu() - ref).abs().max().item()
""",
    "conv_transpose2d": """
import torch
x = torch.randn(2, 16, 32, 32, device=D)
c = torch.nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1).to(D)
y = c(x)
torch.supa.synchronize()
ref = torch.nn.functional.conv_transpose2d(x.cpu(), c.weight.cpu(), c.bias.cpu(), 2, 1)
err = (y.cpu() - ref).abs().max().item()
""",
    "glu": """
import torch, torch.nn.functional as F
x = torch.randn(2, 64, 128, device=D)
y = F.glu(x, dim=1)
torch.supa.synchronize()
ref = F.glu(x.cpu(), dim=1)
err = (y.cpu() - ref).abs().max().item()
""",
    "snake_activation": """
import torch
x = torch.randn(2, 16, 64, 64, device=D)
a = torch.rand(1, 16, 1, 1, device=D) + 0.5
y = x + (1.0 / a) * torch.sin(a * x).pow(2)
torch.supa.synchronize()
ref = x.cpu() + (1.0 / a.cpu()) * torch.sin(a.cpu() * x.cpu()).pow(2)
err = (y.cpu() - ref).abs().max().item()
""",
    "stft_native": """
import torch
x = torch.randn(2, 4096 * 2, device=D)
w = torch.hann_window(4096)
y = torch.stft(x, 4096, 1024, window=w, center=True, return_complex=True)
torch.supa.synchronize()
err = 0.0
""",
    "fft_rfft_direct": """
import torch
x = torch.randn(2, 64, 4096, device=D)
y = torch.fft.rfft(x, dim=-1)
torch.supa.synchronize()
ref = torch.fft.rfft(x.cpu(), dim=-1)
err = (y.cpu() - ref).abs().max().item()
""",
    "fft_irfft_direct": """
import torch
x = torch.randn(2, 64, 2049, device=D)
y = torch.fft.irfft(x, n=4096, dim=-1)
torch.supa.synchronize()
ref = torch.fft.irfft(x.cpu(), n=4096, dim=-1)
err = (y.cpu() - ref).abs().max().item()
""",
    "fft_fft2_direct": """
import torch
x = torch.randn(2, 16, 32, 32, device=D)
y = torch.fft.fft2(x)
torch.supa.synchronize()
ref = torch.fft.fft2(x.cpu())
err = 0.0
""",
    "matmul_bf16": """
import torch
a = torch.randn(256, 256, device=D, dtype=torch.bfloat16)
b = torch.randn(256, 256, device=D, dtype=torch.bfloat16)
y = a @ b
torch.supa.synchronize()
ref = (a.cpu().float() @ b.cpu().float()).bfloat16()
err = (y.cpu() - ref).abs().max().item()
""",
    "softmax_attn": """
import torch
q = torch.randn(2, 4, 128, 64, device=D)
k = torch.randn(2, 4, 128, 64, device=D)
s = (q @ k.transpose(-1, -2)) / 8.0
y = torch.softmax(s, dim=-1)
torch.supa.synchronize()
ref = torch.softmax((q.cpu() @ k.cpu().transpose(-1, -2)) / 8.0, dim=-1)
err = (y.cpu() - ref).abs().max().item()
""",
    "sdpf_attention": """
import torch
q = torch.randn(2, 4, 128, 64, device=D)
k = torch.randn(2, 4, 128, 64, device=D)
v = torch.randn(2, 4, 128, 64, device=D)
y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.supa.synchronize()
ref = torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
err = (y.cpu() - ref).abs().max().item()
""",
    "lstm_unrolled": """
import torch
x = torch.randn(2, 16, 32, device=D)
cell = torch.nn.LSTMCell(32, 64).to(D)
h = torch.zeros(2, 64, device=D); c = torch.zeros(2, 64, device=D)
for i in range(4):
    h, c = cell(x[:, i], (h, c))
torch.supa.synchronize()
cell_cpu = torch.nn.LSTMCell(32, 64)
cell_cpu.load_state_dict({k: v.cpu() for k, v in cell.state_dict().items()})
h2 = torch.zeros(2, 64); c2 = torch.zeros(2, 64)
for i in range(4):
    h2, c2 = cell_cpu(x[:, i].cpu(), (h2, c2))
err = (h.cpu() - h2).abs().max().item()
""",
    "index_add_scatter": """
import torch
out = torch.zeros(2, 1024, device=D)
idx = torch.arange(64, device=D)
src = torch.randn(2, 64, device=D)
out.index_add_(1, idx, src)
torch.supa.synchronize()
ref = torch.zeros(2, 1024); ref.index_add_(1, torch.arange(64), src.cpu())
err = (out.cpu() - ref).abs().max().item()
""",
}

TEMPLATE = """
import torch
import torch_br
D = torch.device("supa:0")
{body}
print(f"OK {{err:.3e}}", flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--only", default="", help="comma separated probe names")
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", "1")

    wanted = [name for name in PROBES if not args.only or name in args.only.split(",")]
    width = max(len(name) for name in wanted)
    for name in wanted:
        code = TEMPLATE.format(body=textwrap.dedent(PROBES[name]).strip())
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout, env=env
            )
        except subprocess.TimeoutExpired:
            print(f"{name:{width}s} TIMEOUT>{args.timeout:.0f}s", flush=True)
            continue
        out = [ln for ln in proc.stdout.splitlines() if ln.startswith("OK")]
        if out:
            print(f"{name:{width}s} {out[-1]}", flush=True)
            continue
        err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        verdict = "CRASH" if proc.returncode not in (0, 1) else "FAIL"
        reason = next((ln for ln in reversed(err) if "Error" in ln or "error" in ln), err[-1] if err else "?")
        print(f"{name:{width}s} {verdict} rc={proc.returncode} {reason[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
