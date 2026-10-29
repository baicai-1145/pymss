"""Contract probe: supa_fused_mlp_linear_act_linear_fwd vs our FeedForward core.

Our FFN is Linear(384->1536) -> GELU -> Linear(1536->384) (eager: 2 GEMMs +
elementwise, 9-13.3 ms in-model).  torch_br ships a fused single-kernel path
`supa_fused_mlp_linear_act_linear_fwd`.  This probe discovers the binding
signature (pybind errors reveal the expected C++ signature), validates
numerics against the eager path (exact vs tanh GELU references), and times
both on the production token counts.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-fused-mlp.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import torch_br
    device = torch.device('supa:0')
    op = torch_br.supa_fused_mlp_linear_act_linear_fwd
    op_bg = torch_br.fused_mlp_bias_gelu_fwd
    op_gen = torch_br.supa_fused_mlp_fwd
    op_out = torch_br.supa_fused_mlp_out_fwd
    results = []

    dim, hidden, tokens = 384, 1536, 53444
    torch.manual_seed(0)
    x3 = (torch.randn(62, 862, dim, device=device) * 0.3).to(torch.bfloat16)
    w1 = (torch.randn(hidden, dim, device=device) * 0.02).to(torch.bfloat16)
    b1 = (torch.randn(hidden, device=device) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(dim, hidden, device=device) * 0.02).to(torch.bfloat16)
    b2 = (torch.randn(dim, device=device) * 0.02).to(torch.bfloat16)

    def eager(x, act):
        h = torch.nn.functional.linear(x, w1, b1)
        h = act(h)
        return torch.nn.functional.linear(h, w2, b2)

    ref_exact = eager(x3, torch.nn.functional.gelu)
    ref_tanh = eager(x3, lambda t: torch.nn.functional.gelu(t, approximate='tanh'))
    scale = ref_exact.float().norm()

    def attempt(tag, call):
        record = {'tag': tag}
        try:
            out = call()
            torch.supa.synchronize()
            got = out.float()
            record['shape'] = list(out.shape)
            record['rel_exact'] = round(float((got - ref_exact.float()).norm() / scale), 6)
            record['rel_tanh'] = round(float((got - ref_tanh.float()).norm() / scale), 6)
            record['finite'] = bool(got.isfinite().all())
            for _ in range(3):
                call()
            torch.supa.synchronize()
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                call()
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['fused_median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            message = str(error).splitlines()
            record['error'] = (message[-1] if message else type(error).__name__)[:260]
        results.append(record)
        print('FMLP', json.dumps(record), flush=True)

    # eager baseline timing
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        eager(x3, torch.nn.functional.gelu)
        torch.supa.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    print('FMLP', json.dumps({'tag': 'eager3d', 'eager_median_ms': round(statistics.median(samples), 3)}), flush=True)

    # contrib shows these return lists: [output, ...aux]
    # augmented-ones trick folds b1 (inside gelu) into the bias-free fused op
    ones = torch.ones(62, 862, 1, device=device, dtype=torch.bfloat16)
    w1_aug = torch.cat((w1, b1.unsqueeze(1).to(torch.bfloat16)), dim=1).contiguous()   # (1536, 385)

    def fused_aug():
        x_aug = torch.cat((x3, ones), dim=-1)
        return op_bg(x_aug, w1_aug, w2)[0] + b2

    # two-step: eager linear1(+bias)+gelu, then candidate fused output-half kernels
    h = torch.nn.functional.gelu(torch.nn.functional.linear(x3, w1, b1))
    ref_half = torch.nn.functional.linear(h, w2, b2)

    def attempt_half(tag, call):
        record = {'tag': tag}
        try:
            out = call()
            if isinstance(out, (list, tuple)):
                out = out[0]
            torch.supa.synchronize()
            got = out.float()
            record['shape'] = list(out.shape)
            record['rel_half'] = round(float((got - ref_half.float()).norm() / ref_half.float().norm()), 6)
            record['finite'] = bool(got.isfinite().all())
            for _ in range(3):
                call()
            torch.supa.synchronize()
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                call()
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            message = str(error).splitlines()
            record['error'] = (message[-1] if message else type(error).__name__)[:200]
        results.append(record)
        print('FMLP_HALF', json.dumps(record), flush=True)

    attempt_half('out-h-wb', lambda: op_out(h, w2, b2))
    attempt_half('out-h-w', lambda: op_out(h, w2))
    attempt_half('out-h-wbw', lambda: op_out(h, w2, b2, None))

    attempt('lall-act-gelu', lambda: op(x3, w1, w2, b1, b2, act_mode='gelu')[0])
    attempt('lall-act-GELU', lambda: op(x3, w1, w2, b1, b2, act_mode='GELU')[0])
    attempt('lall-act-Gelu', lambda: op(x3, w1, w2, b1, b2, act_mode='Gelu')[0])
    attempt('lall-act-gelu_tanh', lambda: op(x3, w1, w2, b1, b2, act_mode='gelu_tanh')[0])
    attempt('lall-act-tanh', lambda: op(x3, w1, w2, b1, b2, act_mode='tanh')[0])
    attempt('lall-act-1', lambda: op(x3, w1, w2, b1, b2, act_mode='1')[0])
    attempt('lall-nobias-gelu', lambda: op(x3, w1, w2, act_mode='gelu')[0])
    attempt('bg-3d+downbias', lambda: op_bg(x3, w1, w2, b1)[0] + b2)
    attempt('bg-3d-nobias', lambda: op_bg(x3, w1, w2)[0])
    attempt('lall-3d-wbwb', lambda: op(x3, w1, w2, b1, b2)[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('FMLP_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
