"""In-model AB/BA compare RMSNorm implementations on Biren bf16+FMHA.

Original aten.rms_norm dispatches pow -> item on this SDK. Test both the explicit
vendor inference kernel and a square-with-multiply formulation. No production
function is changed by this probe.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', default='.tmp/br-rmsnorm-ab.json')
    parser.add_argument('--candidate', choices=['square_fp32', 'vendor_fused', 'vendor_fp32_weight', 'vendor_standard', 'vendor_padded512'], default='square_fp32')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    import torch.nn.functional as F
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.modules.bs_roformer.transformer import RMSNorm
    from pymss_core.vendor_shims import sdpa_equivalent
    installed_attention = F.scaled_dot_product_attention
    original = RMSNorm.forward

    def fused(self, x):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return torch.ops.supa.supa_rmsnorm_infer(flat, self.gamma, 1e-12).reshape_as(x)

    def square(self, x):
        values = x.float()
        inverse = torch.rsqrt((values * values).mean(dim=-1, keepdim=True) + 1e-12)
        return (values * inverse * self.gamma.float()).to(dtype=x.dtype)

    def fused_fp32_weight(self, x):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return torch.ops.supa.supa_rmsnorm_infer(flat, self.gamma.float(), 1e-12).reshape_as(x)

    def standard(self, x):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        return torch.ops.supa.supa_rmsnorm(flat, self.gamma, 1e-12, False).reshape_as(x)

    padded_weights = {}

    def padded512(self, x):
        assert x.shape[-1] == 384, tuple(x.shape)
        key = (self.gamma.data_ptr(), self.gamma._version, str(self.gamma.device))
        if key not in padded_weights:
            padded_weights[key] = F.pad(self.gamma.float() * ((384 / 512) ** 0.5), (0, 128))
        # mean_512(pad(x)^2) = (384/512)*mean_384(x^2).
        # Scale gamma and eps to preserve the original normalization formula.
        padded = F.pad(x.reshape(1, -1, 384), (0, 128))
        out = torch.ops.supa.supa_rmsnorm_infer(padded, padded_weights[key], 1e-12 * (384 / 512))
        return out[..., :384].reshape(x.shape).contiguous()

    candidates = {'square_fp32': square, 'vendor_fused': fused,
                  'vendor_fp32_weight': fused_fp32_weight, 'vendor_standard': standard,
                  'vendor_padded512': padded512}
    functions = {'original': original, args.candidate: candidates[args.candidate]}
    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1
    durations = {name: [] for name in functions}
    outputs = {}
    profiles = {}
    try:
        with torch.no_grad():
            try:
                F.scaled_dot_product_attention = sdpa_equivalent
                reference_fp32 = model(x).clone()
                torch.supa.synchronize()
            finally:
                F.scaled_dot_product_attention = installed_attention
            model.to(dtype=torch.bfloat16)
            print('RMS_FP32_REFERENCE_READY', flush=True)
            for name, function in functions.items():
                RMSNorm.forward = function
                print('RMS_WARM_BEGIN', name, flush=True)
                for _ in range(2):
                    model(x)
                torch.supa.synchronize()
                print('RMS_WARMED', name, flush=True)
            names = list(functions)
            for index in range(args.repeat):
                order = names if index % 2 == 0 else list(reversed(names))
                for name in order:
                    RMSNorm.forward = functions[name]
                    torch.supa.synchronize()
                    start = time.perf_counter()
                    outputs[name] = model(x)
                    torch.supa.synchronize()
                    durations[name].append((time.perf_counter() - start) * 1000)
                print('RMS_PAIR', index, {name: durations[name][-1] for name in functions}, flush=True)
            reference = outputs['original']
            errors = {}
            for name, output in outputs.items():
                diff = output - reference
                fp32_diff = output - reference_fp32
                errors[name] = {'max_relative': (diff.abs().max() / reference.abs().max()).item(),
                                'relative_l2': (diff.norm() / reference.norm()).item(),
                                'relative_l2_vs_fp32': (fp32_diff.norm() / reference_fp32.norm()).item(),
                                'finite': bool(output.isfinite().all())}
            for name, function in functions.items():
                RMSNorm.forward = function
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    model(x)
                    torch.supa.synchronize()
                profiles[name] = [{'op': row.key, 'count': row.count,
                                   'self_cpu_ms': row.self_cpu_time_total / 1000,
                                   'total_cpu_ms': row.cpu_time_total / 1000}
                                  for row in prof.key_averages() if any(token in row.key for token in
                                  ('_local_scalar_dense', '_to_cpu', 'rmsnorm', 'rms_norm', 'sudnn_scaled'))]
            result = {'ms': durations, 'median_ms': {name: statistics.median(values)
                      for name, values in durations.items()}, 'errors': errors, 'profiles': profiles}
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2))
            print('RMS_RESULT', json.dumps(result), flush=True)
    finally:
        RMSNorm.forward = original
        F.scaled_dot_product_attention = installed_attention
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
