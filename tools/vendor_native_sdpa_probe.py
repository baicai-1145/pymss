"""Audit genuine native Biren SDPA without silent fallback or patched aliases.

Save the builtin before importing pymss_core, then run exactly one case per process.
Example: python tools/vendor_native_sdpa_probe.py --case freq --chunk 1024 --dtype bf16
"""
from __future__ import annotations
import argparse
import contextlib
import json
import time
import traceback

import torch
import torch.nn.functional as F

NATIVE_SDPA = F.scaled_dot_product_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['tiny', 'time', 'freq'], default='freq')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--backend', choices=['default', 'math', 'flash', 'efficient'], default='default')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    print('NATIVE_PROVENANCE', json.dumps({'original': repr(NATIVE_SDPA),
          'current': repr(F.scaled_dot_product_attention),
          'distinct_from_shim': NATIVE_SDPA is not F.scaled_dot_product_attention,
          'torch': torch.__version__, 'config': vars(args)}), flush=True)
    assert NATIVE_SDPA is not F.scaled_dot_product_attention
    assert not getattr(NATIVE_SDPA, '_vendor_shim', False)
    shape = {'tiny': (1, 2, 62, 64), 'time': (62, 8, 862, 64), 'freq': (862, 8, 62, 64)}[args.case]
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    torch.manual_seed(0)
    q, k, v = [torch.randn(*shape, dtype=dtype, device='supa:0') for _ in range(3)]
    batch, heads, length, dim = shape
    if args.chunk:
        q, k, v = [x.reshape(batch * heads, length, dim) for x in (q, k, v)]
    backend_names = {'math': 'MATH', 'flash': 'FLASH_ATTENTION', 'efficient': 'EFFICIENT_ATTENTION'}

    def backend_context():
        if args.backend == 'default':
            return contextlib.nullcontext()
        import torch.nn.attention as attention
        return attention.sdpa_kernel(getattr(attention.SDPBackend, backend_names[args.backend]))

    def call_once(round_index, verbose=True):
        pieces = []
        size = args.chunk or q.shape[0]
        for index, start in enumerate(range(0, q.shape[0], size)):
            operands = [x[start:start + size] for x in (q, k, v)]
            torch.supa.synchronize()
            begin = time.perf_counter()
            if verbose:
                print('NATIVE_BEGIN', round_index, index, list(operands[0].shape), flush=True)
            with backend_context():
                part = NATIVE_SDPA(*operands, dropout_p=0.0, scale=dim ** -0.5)
            host = (time.perf_counter() - begin) * 1000
            torch.supa.synchronize()
            wall = (time.perf_counter() - begin) * 1000
            if verbose:
                print('NATIVE_CALL', json.dumps({'round': round_index, 'part': index,
                      'shape': list(operands[0].shape), 'host_ms': host, 'wall_ms': wall,
                      'device': str(part.device), 'dtype': str(part.dtype)}), flush=True)
            pieces.append(part)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    try:
        with torch.no_grad():
            for index in range(2):
                output = call_once(index)
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    call_once('profile', verbose=False)
                rows = sorted(prof.key_averages(), key=lambda row: row.self_cpu_time_total, reverse=True)
                print('NATIVE_PROFILE', json.dumps([{'op': row.key, 'count': row.count,
                      'self_cpu_ms': row.self_cpu_time_total / 1000,
                      'total_cpu_ms': row.cpu_time_total / 1000} for row in rows[:18]]), flush=True)
            print('NATIVE_OK', json.dumps({'finite': bool(output.isfinite().all()),
                                          'shape': list(output.shape)}), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
