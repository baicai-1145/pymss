"""Contract probe for supa_rope_fwd (works where supa_rope rejected 4-D input).

supa::supa_rope_fwd(Tensor input, float base, int Batch, int Seq_len, int Head,
int Head_size, bool is_cached, bool is_half_mode=True, int dim=0, int Tp_num=0,
float factor=0., ...) -> Tensor

First probe round: fwd accepted (62,862,8,64) bf16 and returned (496,862,64).
This probe pins its pairing semantics (halves vs interleaved via dual
references), input layout (4-D (B,S,H,D) vs 3-D (BH,S,1,D)), numerics at both
production shapes (incl. the 6896-batch freq side), and speed vs the chunked
supa_rope_apply baseline.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-fwd.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    op = torch.ops.supa.supa_rope_fwd
    results = []

    def attempt(tag, batch, seq, heads, dim, layout, is_half=True):
        record = {'tag': tag, 'batch': batch, 'seq': seq, 'heads': heads, 'dim': dim,
                  'layout': layout, 'is_half': is_half}
        try:
            x4 = (torch.randn(batch, seq, heads, dim, device=device) * 0.5).to(torch.bfloat16)
            x = x4 if layout == '4d' else x4.permute(0, 2, 1, 3).reshape(batch * heads, seq, dim).contiguous()
            b_, h_ = (batch, heads) if layout == '4d' else (batch * heads, 1)
            out = op(x, 10000.0, b_, seq, h_, dim, False, is_half, 0)
            torch.supa.synchronize()
            inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
            angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
            cos, sin = angles.cos()[None], angles.sin()[None]
            flat = x4.reshape(batch * heads, seq, dim).float()
            even, odd = flat[..., ::2], flat[..., 1::2]
            ref_halves = torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            ref_inter = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
            got = out.reshape(batch * heads, seq, dim).float()
            record['rel_l2_halves'] = round(float((got - ref_halves).norm() / ref_halves.norm()), 6)
            record['rel_l2_interleaved'] = round(float((got - ref_inter).norm() / ref_inter.norm()), 6)
            record['finite'] = bool(got.isfinite().all())
            for _ in range(3):
                op(x, 10000.0, b_, seq, h_, dim, False, is_half, 0)
            torch.supa.synchronize()
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                op(x, 10000.0, b_, seq, h_, dim, False, is_half, 0)
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: ' + str(error).splitlines()[-1][:140]
        results.append(record)
        print('FWD', json.dumps(record), flush=True)

    attempt('small-4d', 2, 17, 8, 64, '4d')
    attempt('small-3d', 2, 17, 8, 64, '3d')
    attempt('time-4d', 62, 862, 8, 64, '4d')
    attempt('time-3d', 62, 862, 8, 64, '3d')
    attempt('freq-4d', 862, 62, 8, 64, '4d')
    attempt('freq-3d', 862, 62, 8, 64, '3d')
    attempt('time-4d-neox', 62, 862, 8, 64, '4d', is_half=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('FWD_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
