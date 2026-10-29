"""End-to-end check of the integrated vendor RoPE path (production entry points).

Uses the real installation path -- prepare_device_alias('biren') installs the
rope shim via the device profile, and prepare_model_for_vendor_rope() permutes
weights -- with no probe monkeypatching.  Compares full-model latency and
fp32-anchor accuracy against the unprepared (original rotary) configuration.
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
    parser.add_argument('--output', default='.tmp/br-rope-integration.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    assert os.environ.get('PYMSS_ROPE_BACKEND', 'vendor') == 'vendor', 'default must stay vendor'
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core import vendor_shims
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.vendor_shims import prepare_model_for_vendor_rope

    assert 'rope' in vendor_shims.installed_shims(), 'rope shim missing from BIREN profile install'
    print('INTEGRATION_SHIMS', sorted(vendor_shims.installed_shims()), flush=True)

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1

    def timed_forward():
        start = time.perf_counter()
        output = model(x)
        torch.supa.synchronize()
        return output, (time.perf_counter() - start) * 1000

    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.supa.synchronize()
        original, _ = timed_forward()
        original = original.float()

        fp32_model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                                time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')
        fp32_model.load_state_dict({key: value.float() for key, value in model.state_dict().items()})
        import torch.nn.functional as functional
        shim = functional.scaled_dot_product_attention

        def math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, *, scale=None, enable_gqa=False):
            if query.device.type != 'supa' or query.dtype != torch.float32:
                return shim(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            scale = query.shape[-1] ** -0.5 if scale is None else scale
            shape = query.shape
            q3 = query.reshape(-1, shape[-2], shape[-1])
            k3 = key.reshape(-1, shape[-2], shape[-1])
            v3 = value.reshape(-1, shape[-2], shape[-1])
            pieces = []
            for start in range(0, q3.shape[0], 64):
                scores = (q3[start:start + 64] @ k3[start:start + 64].transpose(-1, -2)) * scale
                pieces.append(torch.softmax(scores, dim=-1) @ v3[start:start + 64])
            return torch.cat(pieces, dim=0).reshape(shape)

        functional.scaled_dot_product_attention = math_sdpa
        try:
            fp32_out = fp32_model(x).float()
            torch.supa.synchronize()
        finally:
            functional.scaled_dot_product_attention = shim
        del fp32_model

        results = {}
        saved_raw = {key: value.clone() for key, value in model.state_dict().items()}
        for label in ('original', 'prepared', 'prepared_reload'):
            if label == 'prepared_reload':
                # Raw checkpoint load after preparation must be re-permuted by
                # the registered post-hook and stay on the vendor path.
                model.load_state_dict(saved_raw)
            if label == 'prepared':
                prepared = prepare_model_for_vendor_rope(model)
                assert len(prepared) == 24, f'expected 24 to_qkv, got {len(prepared)}'
            for _ in range(2):
                model(x)
            torch.supa.synchronize()
            samples = []
            for _ in range(args.repeat):
                output, ms = timed_forward()
                samples.append(ms)
            results[label] = {
                'median_ms': round(statistics.median(samples), 1),
                'samples_ms': [round(v, 1) for v in samples],
                'rel_to_fp32': round(float((output.float() - fp32_out).norm() / fp32_out.norm()), 6),
                'rel_to_original': round(float((output.float() - original).norm() / original.norm()), 6),
                'finite': bool(output.isfinite().all()),
            }
            print('INTEGRATION_RESULT', json.dumps(results[label]), label, flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print('INTEGRATION_ARTIFACT', str(output_path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
