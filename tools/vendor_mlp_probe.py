"""In-model AB/BA compare Biren's fused GELU MLP with the default FFN.

The vendor out-variant returns an activation used for backward; its first mutable
argument holds the final output. Dropout must be disabled. Diagnostic only.
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
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default='.tmp/br-mlp-ab.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    import torch.nn.functional as F
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.modules.bs_roformer.transformer import FeedForward
    from pymss_core.vendor_shims import sdpa_equivalent
    original = FeedForward.forward
    installed_attention = F.scaled_dot_product_attention
    calls = collections.Counter()

    def fused(self, x):
        assert not self.training
        normalized = self.net[0](x)
        up, down = self.net[1], self.net[4]
        flat = normalized.reshape(-1, normalized.shape[-1])
        buffer = flat.new_empty(flat.shape[0], down.out_features)
        torch.ops.supa.fused_mlp_out_bias_gelu_fwd(
            buffer, flat, up.weight, up.bias, down.weight, down.bias)
        calls[str(tuple(normalized.shape))] += 1
        return buffer.reshape(*normalized.shape[:-1], down.out_features)

    functions = {'original': original, 'native_mlp': fused}
    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')
    x = torch.randn(1, 2, 441000, device='supa:0') * 0.1
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
            for name, function in functions.items():
                FeedForward.forward = function
                print('MLP_WARM_BEGIN', name, flush=True)
                for _ in range(2):
                    model(x)
                torch.supa.synchronize()
                print('MLP_WARMED', name, flush=True)
            calls.clear()
            for index in range(args.repeat):
                order = list(functions) if index % 2 == 0 else list(reversed(functions))
                for name in order:
                    FeedForward.forward = functions[name]
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = model(x)
                    torch.supa.synchronize()
                    durations[name].append((time.perf_counter() - start) * 1000)
                print('MLP_PAIR', index, {name: durations[name][-1] for name in functions}, flush=True)
            errors = {}
            reference = outputs['original']
            for name, output in outputs.items():
                errors[name] = {'relative_l2': ((output-reference).norm()/reference.norm()).item(),
                                'relative_l2_vs_fp32': ((output-reference_fp32).norm()/reference_fp32.norm()).item(),
                                'finite': bool(output.isfinite().all())}
            assert sum(calls.values()) == args.repeat * 24, dict(calls)
            result = {'ms': durations, 'median_ms': {name: statistics.median(values)
                      for name, values in durations.items()}, 'errors': errors, 'native_calls': dict(calls)}
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2))
            print('MLP_RESULT', json.dumps(result), flush=True)
    finally:
        FeedForward.forward = original
        F.scaled_dot_product_attention = installed_attention
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
