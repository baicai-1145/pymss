"""Timing/accuracy sweep for the depthwise-conv weight-gradient kernels (TASK §10.1).

Runs on an otherwise idle NPU.  For each config: fp32 rel_rms against a float64 reference, bf16 rel_rms against eager
autograd, then ABBA timing (20 reps, median, buffer reset inside the timed region because the kernel accumulates with
atomics).  Writes `profile/dwconv_v2.json`.  `python triton_bench_dw.py [--quick]`.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import triton_npu as T  # noqa: E402

N, C, TT, K = 24 * 62, 384, 258, 31


def rel(a, b):
    return float((a.double() - b.double()).square().mean().sqrt() / b.double().square().mean().sqrt())


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.npu.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def reference(dtype):
    """float64 ground truth and eager autograd dW, same inputs."""
    torch.manual_seed(0)
    x = torch.randn(N, C, TT, device='npu', dtype=dtype)
    dy = torch.randn(N, C, TT, device='npu', dtype=dtype)
    conv = torch.nn.Conv1d(C, C, K, padding=K // 2, groups=C).npu().to(dtype)
    y = torch.nn.functional.conv1d(x, conv.weight, conv.bias, padding=K // 2, groups=C)
    y.backward(dy)
    eager = conv.weight.grad.reshape(C, K).float()
    conv.zero_grad(set_to_none=True)
    px = torch.nn.functional.pad(x.double(), (K // 2, K // 2))
    ref64 = torch.stack([(dy.double() * px[:, :, j:j + TT]).sum(dim=(0, 2)) for j in range(K)], dim=-1)
    return x, dy, eager, ref64


def run_v2(x, dy, br, bt, bkc, ns):
    dw = torch.zeros(C, K, device='npu', dtype=torch.float32)
    db = torch.zeros(C, device='npu', dtype=torch.float32)

    def fn():
        dw.zero_(); db.zero_()
        T._dw_dw2[(C, ns, triton.cdiv(TT, bt))](dy, x, dw, db, N, C, TT, K, NS=ns, BR=br, BT=bt, BKC=bkc,
                                                HAS_B=True, KCHUNKS=T._pow2(K) // bkc, num_warps=1)
        return dw
    return fn


def run_v1(x, dy, cfg):
    dw = torch.zeros(C, K, device='npu', dtype=torch.float32)
    db = torch.zeros(C, device='npu', dtype=torch.float32)

    def fn():
        dw.zero_(); db.zero_()
        T._dw_dw[(triton.cdiv(C, cfg['bc']), cfg['ns'])](dy, x, dw, db, N, C, TT, K, NS=cfg['ns'], BC=cfg['bc'],
                                                         BT=cfg['bt'], BK=32, HAS_B=True, num_warps=1)
        return dw
    return fn


def main():
    assert T.available()
    out = []
    for dtype in (torch.float32, torch.bfloat16):
        x, dy, eager, ref64 = reference(dtype)
        tag = str(dtype).split('.')[-1]
        configs = [(4, 128, 16, 4), (4, 128, 16, 2), (8, 128, 16, 2), (4, 128, 8, 4), (8, 128, 8, 4),
                   (4, 256, 16, 4), (2, 128, 16, 8), (8, 64, 16, 4), (16, 128, 16, 2), (4, 128, 32, 4)]
        for br, bt, bkc, ns in configs:
            fn = run_v2(x, dy, br, bt, bkc, ns)
            try:
                got = fn().clone()
            except Exception as exc:  # noqa: BLE001
                out.append(dict(kernel='dwconv_dw2', dtype=tag, br=br, bt=bt, bkc=bkc, ns=ns, error=type(exc).__name__))
                print(f'dwconv_dw2 {tag} br={br} bt={bt} bkc={bkc} ns={ns}: {type(exc).__name__}', flush=True)
                continue
            row = dict(kernel='dwconv_dw2', dtype=tag, br=br, bt=bt, bkc=bkc, ns=ns, grid=C * ns * triton.cdiv(TT, bt),
                       ms=round(timed(fn), 2), rel_eager=rel(got, eager), rel_float64=rel(got, ref64),
                       rel_eager_autograd=rel(eager, ref64))
            out.append(row)
            print(f'dwconv_dw2 {tag} br={br} bt={bt} bkc={bkc} ns={ns} grid={row["grid"]}: {row["ms"]:6.2f} ms  '
                  f'rel_eager={row["rel_eager"]:.1e} rel_f64={row["rel_float64"]:.1e}', flush=True)
            if dtype is torch.bfloat16 and row['ms'] <= 10.0 and row['rel_eager'] <= 1e-2:
                print(f'V2_TARGET_MET {tag} br={br} bt={bt} bkc={bkc} ns={ns} {row["ms"]} ms', flush=True)
                break
        if dtype is torch.bfloat16:
            xv, dyv, _, _ = reference(dtype)
            fn = run_v1(xv, dyv, T._cfg())
            out.append(dict(kernel='dwconv_dw', dtype=tag, ms=round(timed(fn), 2),
                            rel_eager=rel(fn().clone(), eager), rel_float64=rel(fn().clone(), ref64)))
            print(f'dwconv_dw  (v1) {tag}: {out[-1]["ms"]:6.2f} ms', flush=True)
    (ROOT / 'profile' / 'dwconv_v2.json').write_text(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
