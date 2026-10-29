"""Test whether native SDPA's isneginf CPU fallback causes the slowdown.

The dispatch override lives only in this fresh probe process, never in production.
"""
from __future__ import annotations
import json
import statistics
import time

import torch
import torch.nn.functional as F
NATIVE_SDPA = F.scaled_dot_product_attention


def main():
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    assert NATIVE_SDPA is not F.scaled_dot_product_attention
    device = torch.device('supa:0')
    print('RELATED_APIS', json.dumps({
        'torch_supa': [name for name in dir(torch.supa) if any(s in name.lower() for s in ('attention', 'flash', 'sdpa'))],
        'vendor_ops': [name for name in torch._C._dispatch_get_all_op_names()
                       if any(s in name.lower() for s in ('attention', 'sdpa')) and not name.startswith('aten::')],
    }), flush=True)
    for op in ('aten::isneginf', 'aten::_safe_softmax', 'aten::_scaled_dot_product_flash_attention',
               'aten::_scaled_dot_product_efficient_attention'):
        print('DISPATCH', json.dumps({'op': op,
            'privateuse1': torch._C._dispatch_has_kernel_for_dispatch_key(op, 'PrivateUse1')}), flush=True)

    torch.manual_seed(0)
    q, k, v = [torch.randn(1024, 62, 64, device=device, dtype=torch.bfloat16) for _ in range(3)]
    def call():
        return NATIVE_SDPA(q, k, v, dropout_p=0.0, scale=64 ** -0.5)

    def measure(label):
        call()
        torch.supa.synchronize()
        durations = []
        for _ in range(3):
            torch.supa.synchronize()
            start = time.perf_counter()
            out = call()
            torch.supa.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            call()
            torch.supa.synchronize()
        rows = [{'op': row.key, 'count': row.count, 'self_cpu_ms': row.self_cpu_time_total / 1000,
                 'total_cpu_ms': row.cpu_time_total / 1000} for row in prof.key_averages()]
        selected = [row for row in rows if any(token in row['op'] for token in
                    ('isneginf', '_to_cpu', '_to_copy', '_copy_from', 'where', 'softmax', 'eq'))]
        print('ISNEGINF_AB', json.dumps({'mode': label, 'ms': durations,
              'median_ms': statistics.median(durations), 'profile': selected}), flush=True)
        return out.clone()

    with torch.no_grad():
        baseline = measure('native')
        library = torch.library.Library('aten', 'IMPL', 'PrivateUse1')
        def isneginf_device(x):
            if x.is_complex():
                raise RuntimeError('isneginf does not support complex inputs')
            if not x.is_floating_point():
                return torch.zeros_like(x, dtype=torch.bool)
            return torch.eq(x, float('-inf'))
        library.impl('isneginf', isneginf_device)
        for dtype in (torch.float32, torch.bfloat16):
            values_cpu = torch.tensor([float('-inf'), float('inf'), float('nan'), 0., -1.], dtype=dtype)
            got = torch.isneginf(values_cpu.to(device)).cpu()
            assert torch.equal(got, torch.isneginf(values_cpu))
        candidate = measure('device_isneginf')
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
        print('ISNEGINF_RESULT', json.dumps({'output_equal': True,
              'finite': bool(candidate.isfinite().all())}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
