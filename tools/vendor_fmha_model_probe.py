"""AB/BA compare manual and explicit suDNN FMHA in a full bf16 Roformer.

Preserve the inherited SDK PYTHONPATH when launching. Native calls never silently
fall back. Input/output spectra remain fp32; only the neural network is bf16.
"""
from __future__ import annotations
import argparse
import collections
import json
import statistics
import time
import traceback
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    if os.environ['PYMSS_SDPA_BACKEND'] != 'sudnn':
        parser.error('this probe requires PYMSS_SDPA_BACKEND=sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.vendor_shims as shims
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    installed = F.scaled_dot_product_attention
    manual = shims.sdpa_equivalent
    calls = collections.Counter()
    failures = []
    native_helper = shims.sudnn_sdpa

    def audit_native(query, key, value, *, scale=None):
        tag = str(tuple(query.shape))
        calls[tag] += 1
        try:
            return native_helper(query, key, value, scale=scale)
        except Exception:
            failures.append(tag)
            print('MODEL_FMHA_FAILED', tag, flush=True)
            raise
    shims.sudnn_sdpa = audit_native
    fmha = installed

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    times = {'manual': [], 'fmha': []}
    outputs = {}
    try:
        with torch.no_grad():
            for name, function in (('manual', manual), ('fmha', fmha)):
                F.scaled_dot_product_attention = function
                print('MODEL_WARM_BEGIN', name, flush=True)
                for _ in range(2):
                    model(x)
                torch.supa.synchronize()
                print('MODEL_WARMED', name, dict(calls), flush=True)
            calls.clear()
            for index in range(args.repeat):
                order = ('manual', 'fmha') if index % 2 == 0 else ('fmha', 'manual')
                for name in order:
                    F.scaled_dot_product_attention = manual if name == 'manual' else fmha
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = model(x)
                    torch.supa.synchronize()
                    times[name].append((time.perf_counter() - start) * 1000)
                print('MODEL_FMHA_PAIR', index, {name: times[name][-1] for name in times}, flush=True)
            assert sum(calls.values()) == args.repeat * 24, dict(calls)
            ref, out = outputs['manual'], outputs['fmha']
            diff = out - ref
            medians = {name: statistics.median(values) for name, values in times.items()}
            print('MODEL_FMHA_RESULT', json.dumps({'ms': times, 'medians': medians,
                  'speedup': medians['manual'] / medians['fmha'], 'native_calls': dict(calls),
                  'native_failures': failures, 'finite': bool(out.isfinite().all()),
                  'max_relative': (diff.abs().max() / ref.abs().max()).item(),
                  'relative_l2': (diff.norm() / ref.norm()).item()}), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        F.scaled_dot_product_attention = installed
        shims.sudnn_sdpa = native_helper
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
