"""Call Biren's explicit FMHA operator and check its layout against fp32 attention.

The installed SUPA enable_flash_sdp flag does not control device operators. This
probe calls supa::supa_scaled_dot_product_attention directly instead of F.sdpa.
Run one case/layout/dtype per process; never catch an error and claim a fallback passed.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
import traceback
import faulthandler
import threading
import os
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['tiny', 'time', 'freq'], default='tiny')
    parser.add_argument('--layout', choices=['bhsd', 'bshd', 'folded', 'thd'], default='bhsd')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--operator', choices=['sdpa', 'sudnn', 'sueager'], default='sdpa')
    parser.add_argument('--seq-len', type=int, help='override sequence length for alignment probes')
    args = parser.parse_args()
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.vendor_shims import sdpa_equivalent
    shape = {'tiny': (1, 8, 62, 64), 'time': (62, 8, 862, 64), 'freq': (862, 8, 62, 64)}[args.case]
    if args.seq_len is not None:
        shape = (*shape[:2], args.seq_len, shape[-1])
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    torch.manual_seed(0)
    canonical = [torch.randn(*shape, dtype=dtype, device='supa:0') for _ in range(3)]
    if args.layout == 'bhsd':
        inputs = canonical
    elif args.layout == 'bshd':
        inputs = [x.transpose(1, 2).contiguous() for x in canonical]
    elif args.layout == 'folded':
        inputs = [x.reshape(shape[0] * shape[1], shape[2], shape[3]) for x in canonical]
    else:
        inputs = [x.transpose(1, 2).reshape(shape[0] * shape[2], shape[1], shape[3]).contiguous()
                  for x in canonical]
    names = {'sdpa': 'supa_scaled_dot_product_attention',
             'sudnn': 'sudnn_scaled_dot_product_attention_fwd',
             'sueager': 'sueager_scaled_dot_product_attention_fwd'}
    operator = getattr(torch.ops.supa, names[args.operator])
    print('FMHA_BEGIN', json.dumps({'config': vars(args), 'input_shape': list(inputs[0].shape),
                                  'schema': str(operator._schemas)}), flush=True)
    def call():
        result = operator(*inputs, None, 0.0, False, shape[-1] ** -0.5, 'FMHA', None, None)
        return result if args.operator == 'sdpa' else result[0]
    def snapshot():
        channels = Counter()
        for thread in Path('/proc/self/task').iterdir():
            try:
                channels[thread.joinpath('wchan').read_text().strip()] += 1
            except OSError:
                pass
        children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip()
        print('FMHA_WAIT', json.dumps({'pid': os.getpid(), 'thread_waits': dict(channels),
                                      'children': children}), flush=True)
    timer = threading.Timer(10, snapshot)
    timer.daemon = True
    timer.start()
    faulthandler.dump_traceback_later(15, repeat=False)
    try:
        with torch.no_grad():
            out = call()
            print('FMHA_OP_RETURNED', flush=True)
            torch.supa.synchronize()
            print('FMHA_SYNC_RETURNED', flush=True)
            timer.cancel()
            faulthandler.cancel_dump_traceback_later()
            durations = []
            for _ in range(args.repeat):
                torch.supa.synchronize()
                start = time.perf_counter()
                out = call()
                torch.supa.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            if args.layout == 'bshd':
                out = out.transpose(1, 2)
            elif args.layout == 'folded':
                out = out.reshape(shape)
            elif args.layout == 'thd':
                out = out.reshape(shape[0], shape[2], shape[1], shape[3]).transpose(1, 2)
            assert tuple(out.shape) == shape, (out.shape, shape)
            reference = sdpa_equivalent(*(x.float() for x in canonical), scale=shape[-1] ** -0.5)
            diff = out.float() - reference
            print('FMHA_RESULT', json.dumps({'config': vars(args), 'ms': durations,
                  'median_ms': statistics.median(durations), 'output_dtype': str(out.dtype),
                  'finite': bool(out.isfinite().all()),
                  'max_relative': (diff.abs().max() / reference.abs().max()).item(),
                  'relative_l2': (diff.norm() / reference.norm()).item()}), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        timer.cancel()
        faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
