"""Compare fp32 and explicit bf16 Roformer with identical weights, without autocast.

Inputs and spectral reconstruction stay fp32. Run in a fresh process with
PYMSS_SDPA_BACKEND=manual. Timings alternate A/B then B/A after separate warmups.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--boundary', action='store_true', help='temporary entry hook for unfixed cores')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer

    def build():
        return BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                          time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')

    torch.manual_seed(0)
    models = {'fp32': build()}
    models['bf16'] = build()
    models['bf16'].load_state_dict(models['fp32'].state_dict())
    models['bf16'].to(dtype=torch.bfloat16)
    if args.boundary:
        models['bf16'].band_split.register_forward_pre_hook(
            lambda module, inputs: (inputs[0].to(torch.bfloat16),))
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1

    class FailedOp(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            try:
                return func(*args, **(kwargs or {}))
            except Exception:
                tensors = [{'shape': list(a.shape), 'dtype': str(a.dtype),
                            'stride': list(a.stride())} for a in args if isinstance(a, torch.Tensor)]
                print('FAILED_OP', json.dumps({'op': str(func), 'tensors': tensors}), flush=True)
                raise

    durations = {name: [] for name in models}
    outputs = {}
    try:
        with torch.no_grad():
            for name, model in models.items():
                with FailedOp():
                    for _ in range(2):
                        model(x)
                    torch.supa.synchronize()
                print('WARMED', name, flush=True)
            for index in range(args.repeat):
                order = ('fp32', 'bf16') if index % 2 == 0 else ('bf16', 'fp32')
                for name in order:
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = models[name](x)
                    torch.supa.synchronize()
                    durations[name].append((time.perf_counter() - start) * 1000)
                print('PAIR', index, {name: durations[name][-1] for name in durations}, flush=True)
            ref, out = outputs['fp32'], outputs['bf16']
            diff = out.float() - ref
            print('BF16_RESULT', json.dumps({
                'boundary_hook': args.boundary, 'ms': durations,
                'median_ms': {name: statistics.median(times) for name, times in durations.items()},
                'speedup': statistics.median(durations['fp32']) / statistics.median(durations['bf16']),
                'finite': bool(out.isfinite().all()),
                'max_relative': (diff.abs().max() / ref.abs().max()).item(),
                'relative_l2': (diff.norm() / ref.norm()).item(),
                'output_dtype': str(out.dtype),
            }), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
