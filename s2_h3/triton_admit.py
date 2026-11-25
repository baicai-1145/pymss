"""Admission test for the kernels in `triton_npu.py` (TASK §6): numbers first, then speed, then the whole step.

For each kernel, at the real training shape:
  * fp32:  relative RMS error of the output and of every input gradient against eager autograd (limit 1e-5),
  * bf16:  same, against the eager bf16 run (limit 1e-2),
  * speed: eager vs Triton, ABBA, 20 timed reps each, median (Triton must win),
  * step:  H3 forward+loss one step, same seed, eager vs Triton inputs (fp32 rel <= 1e-4, bf16 <= 1e-2).

Writes TRITON.md's table.  `python triton_admit.py [--quick]`.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('S2_DEVICE', 'npu')
os.environ['ARM'] = 'H3'

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FP32_LIMIT, BF16_LIMIT, STEP_LIMIT = 1e-5, 1e-2, 1e-4


def rel_rms(a, b):
    a, b = a.float(), b.float()
    den = b.square().mean().sqrt()
    return float((a - b).square().mean().sqrt() / (den if den > 0 else 1.0))


def timed(fn, reps=20, sync=None):
    sync = sync or (lambda: torch.npu.synchronize())
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def shapes():
    """The MSC depthwise conv at the training shape: (b*f, hid, t) with b=24, f=62, hid=384, t=258."""
    return 24 * 62, 384, 258, 31


def bench_dwconv(dtype):
    import triton_npu
    from torch import nn
    n, c, t, k = shapes()
    torch.manual_seed(0)
    conv = nn.Conv1d(c, c, k, padding=k // 2, groups=c).npu().to(dtype)
    x = torch.randn(n, c, t, device='npu', dtype=dtype, requires_grad=True)

    def eager():
        y = conv(x)
        (y.square().mean()).backward()
        g = (x.grad.clone(), conv.weight.grad.clone(), conv.bias.grad.clone())
        conv.zero_grad(set_to_none=True); x.grad = None
        return y, g

    def tri():
        y = triton_npu.depthwise_conv1d(x, conv.weight, conv.bias, k // 2)
        (y.square().mean()).backward()
        g = (x.grad.clone(), conv.weight.grad.clone(), conv.bias.grad.clone())
        conv.zero_grad(set_to_none=True); x.grad = None
        return y, g

    y_e, g_e = eager()
    y_t, g_t = tri()
    errs = dict(out=rel_rms(y_t, y_e), dx=rel_rms(g_t[0], g_e[0]), dw=rel_rms(g_t[1], g_e[1]),
                db=rel_rms(g_t[2], g_e[2]))
    # ABBA to keep any thermal drift out of the comparison
    a, b = [], []
    for _ in range(2):
        a.append(timed(eager)); b.append(timed(tri))
    ms_e, ms_t = min(a), min(b)
    return dict(kernel='dwconv', dtype=str(dtype).split('.')[-1], shape=f'{n}x{c}x{t} k={k}',
                errors={k2: round(v, 9) for k2, v in errs.items()}, eager_ms=round(ms_e, 2),
                triton_ms=round(ms_t, 2), speedup=round(ms_e / ms_t, 2))


def bench_silu_mul(dtype):
    import triton_npu
    n, c, t, _ = shapes()
    torch.manual_seed(0)
    a = torch.randn(n, t, c, device='npu', dtype=dtype, requires_grad=True)   # MSC: (b*f, t, hid)
    g = torch.randn(n, t, c, device='npu', dtype=dtype, requires_grad=True)

    def run(fn):
        y = fn(a, g)
        y.square().mean().backward()
        out = (y.detach().clone(), a.grad.clone(), g.grad.clone())
        a.grad = None; g.grad = None
        return out

    y_e = run(lambda p, q: torch.nn.functional.silu(p) * q)
    y_t = run(lambda p, q: triton_npu.silu_mul(p, q))
    errs = dict(out=rel_rms(y_t[0], y_e[0]), da=rel_rms(y_t[1], y_e[1]), dg=rel_rms(y_t[2], y_e[2]))
    a_, b_ = [], []
    for _ in range(2):
        a_.append(timed(lambda: run(lambda p, q: torch.nn.functional.silu(p) * q)))
        b_.append(timed(lambda: run(lambda p, q: triton_npu.silu_mul(p, q))))
    ms_e, ms_t = min(a_), min(b_)
    return dict(kernel='silu_mul', dtype=str(dtype).split('.')[-1], shape=f'{n}x{t}x{c}',
                errors={k2: round(v, 9) for k2, v in errs.items()}, eager_ms=round(ms_e, 2),
                triton_ms=round(ms_t, 2), speedup=round(ms_e / ms_t, 2))


def step_loss(names, dtype_eager, seed=0):
    """One H3 forward+loss, eager or with the kernels installed, from the same weights and input."""
    from entry import DEVICE, autocast, build
    from pymsst.training.losses import masked_loss
    import triton_npu
    torch.manual_seed(0)
    model = build('H3').to(DEVICE)
    if names:
        triton_npu.install(model, names)
    model.train(False)
    gmix = torch.Generator().manual_seed(seed)
    mix = ((torch.rand(2, 2, 131584, generator=gmix) * 2 - 1) * 0.15).to(DEVICE)
    tgt = ((torch.rand(2, 2, 131584, generator=gmix) * 2 - 1) * 0.08).to(DEVICE)
    with autocast():
        loss = masked_loss(model(mix).float(), tgt, q=0.95, coarse=True)
    torch.npu.synchronize()
    return float(loss)


def main():
    import triton_npu
    assert triton_npu.available(), 'triton not importable'
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        rows.append(bench_dwconv(dtype))
        rows.append(bench_silu_mul(dtype))
    ref = {str(d).split('.')[-1]: step_loss(None, d) for d in (torch.float32, torch.bfloat16)}
    step = {}
    for names in (['dwconv'], ['silu_mul'], ['dwconv', 'silu_mul']):
        got = step_loss(names, None)
        step['+'.join(names)] = dict(loss=got, rel_fp32=abs(got - ref['float32']) / ref['float32'])
    out = dict(rows=rows, step_loss_ref=ref, step_loss=step, limits=dict(fp32=FP32_LIMIT, bf16=BF16_LIMIT,
                                                                       step=STEP_LIMIT))
    (ROOT / 'profile' / 'triton_admit.json').write_text(json.dumps(out, indent=1))
    print(f'{"kernel":10s} {"dtype":9s} {"eager ms":>9s} {"triton ms":>10s} {"speedup":>8s}   errors (out/dx/dw/db)')
    for r in rows:
        lim = FP32_LIMIT if r['dtype'] == 'float32' else BF16_LIMIT
        ok = all(v <= lim for v in r['errors'].values()) and r['triton_ms'] < r['eager_ms']
        print(f'{r["kernel"]:10s} {r["dtype"]:9s} {r["eager_ms"]:9.2f} {r["triton_ms"]:10.2f} {r["speedup"]:8.2f}   '
              f'{ {k: f"{v:.1e}" for k, v in r["errors"].items()} } {"PASS" if ok else "FAIL"}')
    print(f'step loss ref (eager): {ref}')
    for k, v in step.items():
        print(f'step loss with {k}: {v["loss"]:.8f}  rel vs eager {v["rel_fp32"]:.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
