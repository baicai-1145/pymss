"""Torch-composed RoPE on the halves layout vs the vendor op.

The vendor supa_rope_apply runs at ~15 GB/s effective on the model shapes
(~10 ms GPU per joint q/k call).  With weights permuted, q/k arrive already in
[evens | odds] halves layout, so a composed rotation needs only contiguous
slices -- no strided ::2 access.  This probe times several compositions
(fp32/bf16 math, cat vs preallocated slice writes) against the vendor op on the
exact production shapes and compares outputs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def build_tables(seq, dim, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)   # (seq, dim/2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-compose.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')

    def run(label, batch, seq, dim, function, vendor_ref):
        x = (torch.randn(batch, seq, dim, device=device) * 0.5).to(torch.bfloat16)
        for _ in range(3):
            function(x)
        torch.supa.synchronize()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            out = function(x)
            torch.supa.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        if vendor_ref is not None:
            error = float((out.float() - vendor_ref(x).float()).norm() / vendor_ref(x).float().norm())
        else:
            error = 0.0
        record = {'label': label, 'batch': batch, 'seq': seq,
                  'median_ms': round(statistics.median(samples), 3),
                  'min_ms': round(min(samples), 3), 'rel_vs_vendor': round(error, 6),
                  'finite': bool(out.isfinite().all())}
        print('COMPOSE', json.dumps(record), flush=True)
        return record

    results = []
    for batch, seq in ((496, 862), (6896, 62)):
        dim = 64
        cos_f32, sin_f32 = build_tables(seq, dim, device, torch.float32)
        cos_b16, sin_b16 = build_tables(seq, dim, device, torch.bfloat16)
        cf_full = torch.cat((cos_f32, cos_f32), -1)[None].contiguous()
        sf_full = torch.cat((sin_f32, sin_f32), -1)[None].contiguous()
        half = dim // 2

        def vendor(x, cf=cf_full, sf=sf_full):
            return torch.ops.supa.supa_rope_apply(x, cf, sf)

        def compose_fp32_cat(x, c=cos_f32, s=sin_f32):
            left, right = x[..., :half].float(), x[..., half:].float()
            return torch.cat((left * c - right * s, right * c + left * s), dim=-1).to(torch.bfloat16)

        def compose_fp32_prealloc(x, c=cos_f32, s=sin_f32):
            left, right = x[..., :half].float(), x[..., half:].float()
            out = torch.empty_like(x)
            out[..., :half] = (left * c - right * s).to(torch.bfloat16)
            out[..., half:] = (right * c + left * s).to(torch.bfloat16)
            return out

        def compose_bf16_cat(x, c=cos_b16, s=sin_b16):
            left, right = x[..., :half], x[..., half:]
            return torch.cat((left * c - right * s, right * c + left * s), dim=-1)

        def compose_bf16_prealloc(x, c=cos_b16, s=sin_b16):
            left, right = x[..., :half], x[..., half:]
            out = torch.empty_like(x)
            out[..., :half] = left * c - right * s
            out[..., half:] = right * c + left * s
            return out

        results.append(run('vendor', batch, seq, dim, vendor, None))
        for name, fn in (('compose_fp32_cat', compose_fp32_cat),
                         ('compose_fp32_prealloc', compose_fp32_prealloc),
                         ('compose_bf16_cat', compose_bf16_cat),
                         ('compose_bf16_prealloc', compose_bf16_prealloc)):
            results.append(run(name, batch, seq, dim, fn, vendor))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('COMPOSE_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
