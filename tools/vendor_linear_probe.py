"""Compare full-model Biren linear layouts with identical bf16 weights.

All variants use explicit FMHA. Cached transposes are process-local diagnostics,
not a production weight cache. Timings exclude cache construction and profiling.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default='.tmp/br-linear-ab.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    import torch.nn.functional as F
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.vendor_shims import sdpa_equivalent
    original = F.linear
    installed_attention = F.scaled_dot_product_attention
    transposes = {}
    counters = collections.Counter()
    active = ['original']

    def flattened(x, weight, bias=None):
        counters[active[0]] += 1
        out = original(x.reshape(-1, x.shape[-1]), weight, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    def cached_mm(x, weight, bias=None):
        counters[active[0]] += 1
        key = (weight.data_ptr(), weight._version, weight.dtype, tuple(weight.shape))
        if key not in transposes:
            transposes[key] = weight.t().contiguous()
        flat = x.reshape(-1, x.shape[-1])
        out = torch.mm(flat, transposes[key]) if bias is None else torch.addmm(bias, flat, transposes[key])
        return out.reshape(*x.shape[:-1], weight.shape[0])

    functions = {'original': original, 'flat2d': flattened, 'cached_mm': cached_mm}
    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    durations = {name: [] for name in functions}
    outputs = {}
    try:
        with torch.no_grad():
            try:
                F.scaled_dot_product_attention = sdpa_equivalent
                reference_fp32 = model(x).clone()
                torch.supa.synchronize()
            finally:
                F.scaled_dot_product_attention = installed_attention
            model.to(dtype=torch.bfloat16)
            print('LINEAR_FP32_REFERENCE_READY', flush=True)
            for name, function in functions.items():
                F.linear = function
                active[0] = name
                print('LINEAR_WARM_BEGIN', name, flush=True)
                for _ in range(2):
                    model(x)
                torch.supa.synchronize()
                print('LINEAR_WARMED', name, flush=True)
            counters.clear()
            names = list(functions)
            for index in range(args.repeat):
                order = names if index % 2 == 0 else list(reversed(names))
                for name in order:
                    active[0] = name
                    F.linear = functions[name]
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = model(x)
                    torch.supa.synchronize()
                    durations[name].append((time.perf_counter() - start) * 1000)
                print('LINEAR_PAIR', index, {name: durations[name][-1] for name in functions}, flush=True)
            reference = outputs['original']
            errors = {}
            for name, output in outputs.items():
                diff = output - reference
                errors[name] = {'relative_l2': (diff.norm() / reference.norm()).item(),
                                'relative_l2_vs_fp32': ((output-reference_fp32).norm() / reference_fp32.norm()).item(),
                                'finite': bool(output.isfinite().all())}
            result = {'ms': durations, 'median_ms': {name: statistics.median(values)
                      for name, values in durations.items()}, 'errors': errors,
                      'candidate_linear_calls': dict(counters), 'cached_weights': len(transposes)}
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2))
            print('LINEAR_RESULT', json.dumps(result), flush=True)
    finally:
        F.linear = original
        F.scaled_dot_product_attention = installed_attention
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
