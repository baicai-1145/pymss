"""Per-op accuracy of supa_rope_apply vs sequence length.

The model A/B shows vendor rotary 2.6x farther from fp32 than the original
path (4.065% vs 1.571%) although the standalone contract probe measured only
0.157% at seq=17. Attention layers damp error growth, so the model-level gap
must come from per-op error that grows with sequence length. This probe
measures op error vs fp32 reference at seq in {17,62,63,64,128,256,512,640,
768,832,860,862,863,864,1024} with realistic rotary tables (theta base 10000),
for fp32 and bf16 tables, batch 16, dim 64.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-seq-accuracy.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    device = torch.device('supa:0')
    results = []
    for seq in (17, 62, 63, 64, 128, 256, 512, 640, 768, 832, 860, 862, 863, 864, 1024):
        positions = torch.arange(seq, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64))
        angles = positions[:, None] * inv_freq[None, :]          # (seq, 32)
        cos_half, sin_half = angles.cos(), angles.sin()          # op pairs (i, i+32)
        table_cos = torch.cat([cos_half, cos_half], dim=-1)[None].contiguous()
        table_sin = torch.cat([sin_half, sin_half], dim=-1)[None].contiguous()
        x = (torch.randn(16, seq, 64, device=device) * 0.5).to(torch.bfloat16)
        # Op convention (validated by vendor_fusion_contract_probe generic block):
        # split-half pairing. left = components 0..31, right = 32..63.
        left, right = x[..., :32].float(), x[..., 32:].float()
        reference = torch.cat((left * cos_half[None].float() - right * sin_half[None].float(),
                               right * cos_half[None].float() + left * sin_half[None].float()), -1)
        row = {'seq': seq}
        cf, sf = table_cos, table_sin
        for label in ('fp32_table',):
            try:
                out = torch.ops.supa.supa_rope_apply(x, cf, sf)
                torch.supa.synchronize()
                error = float((out.float() - reference).norm() / reference.norm())
                row[label] = round(error, 6)
                row[label + '_finite'] = bool(out.isfinite().all())
            except RuntimeError as error_out:
                row[label] = None
                row[label + '_error'] = str(error_out).splitlines()[-1][:100]
        results.append(row)
        print('ROPE_SEQ_ACC', json.dumps(row), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('ROPE_SEQ_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
