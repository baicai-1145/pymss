"""Where does the freq-side chunked rope time go?

supa_rope_apply needs dim0<=1024, so the freq side (6896,62,64) runs as 7
chunks + a cat.  Standalone single-call on the time shape costs 0.455 ms, so
if the chunked freq path costs ~10-20 ms the loss is in launch overhead, small
chunks, or the cat.  This probe times each ingredient separately on-device.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch


def bench(function, repeat=9, warmup=3):
    for _ in range(warmup):
        function()
    torch.supa.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        torch.supa.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-freqcost.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    op = torch.ops.supa.supa_rope_apply
    records = {}

    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    for tag, rows, seq in (('time', 496, 862), ('freq', 6896, 62)):
        angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
        cf = torch.cat((angles.cos(), angles.cos()), -1)[None].contiguous()
        sf = torch.cat((angles.sin(), angles.sin()), -1)[None].contiguous()
        x = (torch.randn(rows, seq, 64, device=device) * 0.5).to(torch.bfloat16)
        if rows <= 1024:
            records[f'{tag}_single'] = bench(lambda: op(x, cf, sf))
        step = 985
        pieces = [piece.contiguous() for piece in x.split(step, dim=0)]
        records[f'{tag}_ops_only_7chunks'] = bench(lambda: [op(piece, cf, sf) for piece in pieces])
        chunks = [op(piece, cf, sf) for piece in pieces]  # warm outputs
        records[f'{tag}_cat_only'] = bench(lambda: torch.cat(chunks, dim=0))
        records[f'{tag}_chunked_full'] = bench(lambda: torch.cat([op(piece, cf, sf) for piece in pieces], dim=0))
        records[f'{tag}_step1024'] = bench(lambda: torch.cat([op(piece, cf, sf) for piece in x.split(1024, dim=0)], dim=0))
        # in-place slice writes instead of cat
        def slice_write():
            out = torch.empty_like(x)
            for index, piece in enumerate(pieces):
                out[index * step:(index + 1) * step if (index + 1) * step < rows else rows] = op(piece, cf, sf)
            return out
        records[f'{tag}_slice_write'] = bench(slice_write)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    for key, value in records.items():
        print(f'FREQCOST {key} {value}', flush=True)
    print('FREQCOST_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
