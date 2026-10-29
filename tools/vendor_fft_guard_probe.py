"""Diagnostic-only A/B to separate FFT guard synchronization from real latency.

The candidate disables the silent-zero check only in this fresh process, and must
match the guarded result exactly before timing. Never install this as production
behavior: silent-zero FFT failures still require detection.
"""
import collections
import json
import os
import statistics
import time


def main():
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.npu_fft as fft
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, 441000, device='supa:0') * 0.1
    guard = fft._degenerate
    timings = collections.defaultdict(list)
    outputs = {}
    try:
        with torch.no_grad():
            for name in ('guarded', 'diagnostic_no_guard'):
                fft._degenerate = guard if name == 'guarded' else lambda out, source: False
                for _ in range(2):
                    outputs[name] = model(x)
                torch.supa.synchronize()
            torch.testing.assert_close(outputs['guarded'], outputs['diagnostic_no_guard'], rtol=0, atol=0)
            for index in range(3):
                order = ('guarded', 'diagnostic_no_guard') if index % 2 == 0 else ('diagnostic_no_guard', 'guarded')
                for name in order:
                    fft._degenerate = guard if name == 'guarded' else lambda out, source: False
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = model(x)
                    torch.supa.synchronize()
                    timings[name].append((time.perf_counter() - start) * 1000)
                print('FFT_GUARD_PAIR', index, {name: timings[name][-1] for name in timings}, flush=True)
            torch.testing.assert_close(outputs['guarded'], outputs['diagnostic_no_guard'], rtol=0, atol=0)
            print('FFT_GUARD_RESULT', json.dumps({'ms': timings,
                  'median_ms': {name: statistics.median(values) for name, values in timings.items()},
                  'output_equal': True, 'diagnostic_only': True}), flush=True)
    finally:
        fft._degenerate = guard
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
