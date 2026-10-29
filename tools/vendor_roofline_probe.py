"""Roofline attribution for BS-Roformer transformer layers on Biren.

Measures device ceilings (HBM copy bandwidth, large-GEMM bf16 TFLOP/s) and then
each layer component at its REAL model shapes: qkv/out/FFN GEMMs, native FMHA
for time (496,862,64) vs freq (6896,62,64), and the current rotary elementwise
path. Compares achieved TFLOP/s and GB/s against both ceilings to classify each
component as compute-bound, bandwidth-bound, or below-both (latency/layout).
Serialized single-process probe; no performance claim beyond this report.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

FLOP_PER_BYTE_BALANCE = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-roofline.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast, cached_rotary_cos_sin

    device = torch.device('supa:0')
    report = {}

    def timed(fn, repeat=7, warmup=3):
        for _ in range(warmup):
            fn()
        torch.supa.synchronize()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            torch.supa.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    # 1) HBM bandwidth ceiling: device-to-device copy (read+write).
    size = 256 * 1024 * 1024  # elements, bf16 -> 512 MB per tensor
    source = torch.randn(size, device=device, dtype=torch.bfloat16)
    target = torch.empty_like(source)
    copy_ms = timed(lambda: target.copy_(source), repeat=5, warmup=2)
    bandwidth_gbs = 2 * size * 2 / copy_ms / 1e6
    report['hbm_copy_gbs'] = round(bandwidth_gbs, 1)
    print('ROOF_HBM_COPY_GBPS', round(bandwidth_gbs, 1), flush=True)

    # elementwise 1 read + 1 write reference at activation size (41 MB)
    act = torch.randn(62, 862, 8, 64, device=device, dtype=torch.bfloat16)
    scale_ms = timed(lambda: act * 2.0)
    act_gbs = 2 * act.numel() * 2 / scale_ms / 1e6
    print('ROOF_ELEMENTWISE_41MB_GBPS', round(act_gbs, 1), flush=True)

    # 2) Compute ceiling: large bf16 GEMM.
    gemm = 6144
    a = torch.randn(gemm, gemm, device=device, dtype=torch.bfloat16)
    b = torch.randn(gemm, gemm, device=device, dtype=torch.bfloat16)
    gemm_ms = timed(lambda: a @ b, repeat=3, warmup=1)
    peak_tflops = 2 * gemm ** 3 / gemm_ms / 1e9
    report['gemm_peak_tflops'] = round(peak_tflops, 1)
    print('ROOF_GEMM_PEAK_TFLOPS', round(peak_tflops, 1), flush=True)
    balance = peak_tflops * 1e12 / (bandwidth_gbs * 1e9)
    print('ROOF_BALANCE_FLOP_PER_BYTE', round(balance, 1), flush=True)

    tokens = 62 * 862
    linears = {
        'qkv_53444x384x1152': (tokens, 384, 1152),
        'out_53444x384x384': (tokens, 384, 384),
        'ffn_up_53444x384x1536': (tokens, 384, 1536),
        'ffn_down_53444x1536x384': (tokens, 1536, 384),
    }
    for label, (m, k, n) in linears.items():
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        ms = timed(lambda: torch.nn.functional.linear(x, w))
        tflops = 2 * m * k * n / ms / 1e9
        gbs = 2 * (m * k + k * n + m * n) / ms / 1e6
        print(f'ROOF_LINEAR {label} {round(ms,2)}ms {round(tflops,2)}TF/s {round(gbs,1)}GB/s', flush=True)

    # 3) Native FMHA at both model shapes.
    fmha = torch.ops.supa.sudnn_scaled_dot_product_attention_fwd
    for label, (batch, seq) in {'fmha_time_496x862x64': (496, 862), 'fmha_freq_6896x62x64': (6896, 62)}.items():
        q = torch.randn(batch, seq, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, seq, 64, device=device, dtype=torch.bfloat16)
        scale = 64 ** -0.5
        def call():
            fmha(q, k, v, None, 0.0, False, scale, 'FMHA', None, None)
        ms = timed(call)
        flops = 2 * 2 * batch * 8 * 0 + 4 * batch * seq * seq * 64
        tflops = (2 * batch * seq * seq * 64 * 2) / ms / 1e9
        gbs = 2 * 3 * batch * seq * 64 / ms / 1e6
        print(f'ROOF_FMHA {label} {round(ms,2)}ms {round(tflops,2)}TF/s {round(gbs,1)}GB/s', flush=True)

    # 4) Current rotary path at both shapes (cos/sin cached like the model).
    class FakeRotary:
        pass
    for label, shape in {'rotary_time_62x862x8x64': (62, 862, 8, 64), 'rotary_freq_862x62x8x64': (862, 62, 8, 64)}.items():
        x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        seq = shape[1]
        freqs = torch.randn(1, seq, 1, shape[-1] // 2, device=device)
        cos, sin = freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
        ms = timed(lambda: apply_rotary_emb_fast(cos, sin, x))
        gbs = 6 * x.numel() * 2 / ms / 1e6  # ~read x, read cos/sin, write out, plus stack copies
        print(f'ROOF_ROTARY {label} {round(ms,2)}ms {round(gbs,1)}GB/s', flush=True)

    # 5) Per-layer utilization summary vs both ceilings.
    per_layer_flops = {'time': 236.4e9, 'freq': 196.0e9}
    per_layer_ms = {'time': 48.9, 'freq': 66.4}
    for kind in ('time', 'freq'):
        achieved = per_layer_flops[kind] / per_layer_ms[kind] / 1e6
        print(f'ROOF_LAYER {kind} {round(achieved,2)}TF/s vs peak {round(peak_tflops,1)} '
              f'({round(100*achieved/peak_tflops,1)}% compute) '
              f'implied {round(per_layer_flops[kind]/2/per_layer_ms[kind]/1e6,1)}GB/s vs {round(bandwidth_gbs,1)} '
              f'({round(100*per_layer_flops[kind]/2/per_layer_ms[kind]/1e6/bandwidth_gbs,1)}% bw, assuming 2 FLOP/byte AI)', flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print('ROOF_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
