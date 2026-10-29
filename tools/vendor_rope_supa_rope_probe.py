"""Contract probe for supa_rope (self-computing-angles variant, no tables).

supa::supa_rope(Tensor input, float base, int Batch, int Seq_len, int Head,
int Head_size, bool is_cached, bool is_half_mode=True, int dim=0, ...) -> Tensor

Potential win over supa_rope_apply: no dim0<=1024 launch cap (unverified), no
fp32 table tensors, one call per tensor at the FULL model shape.  This probe
establishes input layout (4-D B,S,H,D vs 3-D BH,S,1,D), pairing semantics for
is_half_mode, and speed on the exact production shapes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def halves_reference(x_bh_s_d, cos_half, sin_half):
    even, odd = x_bh_s_d[..., ::2].float(), x_bh_s_d[..., 1::2].float()
    return torch.cat((even * cos_half - odd * sin_half,
                      odd * cos_half + even * sin_half), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-suparope.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    op = torch.ops.supa.supa_rope
    op_fwd = torch.ops.supa.supa_rope_fwd
    results = []

    def attempt(tag, batch, seq, heads, dim, layout, is_half, is_cached, base=10000.0):
        record = {'tag': tag, 'batch': batch, 'seq': seq, 'heads': heads, 'dim': dim,
                  'layout': layout, 'is_half': is_half, 'is_cached': is_cached}
        try:
            x4 = (torch.randn(batch, seq, heads, dim, device=device) * 0.5).to(torch.bfloat16)
            x = x4 if layout == '4d' else x4.reshape(batch * heads, seq, dim).contiguous()
            b_, s_, h_ = (batch, seq, heads) if layout == '4d' else (batch * heads, seq, 1)
            out = op(x, base, b_, s_, h_, dim, is_cached, is_half, 0)
            torch.supa.synchronize()
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
            ref = halves_reference(x4.reshape(batch * heads, seq, dim), angles.cos()[None], angles.sin()[None])
            ref_inter = torch.stack((x4[..., ::2].float() * angles.cos()[None, None] - x4[..., 1::2].float() * angles.sin()[None, None],
                                     x4[..., 1::2].float() * angles.cos()[None, None] + x4[..., ::2].float() * angles.sin()[None, None]), -1).flatten(-2).reshape(batch * heads, seq, dim)
            got = out.reshape(batch * heads, seq, dim).float()
            record['rel_l2'] = round(float((got - ref).norm() / ref.norm()), 6)
            record['rel_l2_interleaved'] = round(float((got - ref_inter).norm() / ref_inter.norm()), 6)
            record['finite'] = bool(got.isfinite().all())
            for _ in range(3):
                op(x, base, b_, s_, h_, dim, is_cached, is_half, 0)
            torch.supa.synchronize()
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                op(x, base, b_, s_, h_, dim, is_cached, is_half, 0)
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: ' + str(error).splitlines()[-1][:140]
        results.append(record)
        print('SUPAROPE', json.dumps(record), flush=True)

    # semantics + layout on the small shape first
    attempt('small-4d-half', 2, 17, 8, 64, '4d', True, False)
    attempt('small-3d-half', 16, 17, 1, 64, '3d', True, False)
    attempt('small-4d-neox', 2, 17, 8, 64, '4d', False, False)
    # production shapes, full batch, both layouts
    attempt('time-4d', 62, 862, 8, 64, '4d', True, False)
    attempt('time-3d', 496, 862, 1, 64, '3d', True, False)
    attempt('freq-4d', 862, 62, 8, 64, '4d', True, False)
    attempt('freq-3d', 6896, 62, 1, 64, '3d', True, False)
    # fwd variant if the main one works
    try:
        x4 = (torch.randn(62, 862, 8, 64, device=device) * 0.5).to(torch.bfloat16)
        out = op_fwd(x4, 10000.0, 62, 862, 8, 64, False, True, 0)
        torch.supa.synchronize()
        print('SUPAROPE_FWD_OK', list(out.shape), bool(out.isfinite().all()), flush=True)
    except Exception as error:
        print('SUPAROPE_FWD_ERR', str(error).splitlines()[-1][:140], flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('SUPAROPE_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
