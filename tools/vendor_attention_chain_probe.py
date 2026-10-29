"""Layout-chain probe: one regroup copy for qkv instead of three scattered copies.

Current integrated path per attention: q,k each pay a copy inside the vendor
rotate permute, and v pays one in the SDPA shim's ``.contiguous()``.  This probe
replaces Attention.forward for the SUPA bf16 rotary path with a version that
regroups the qkv projection ONCE into (3, B, H, S, D); rotary and the FMHA
adapter then consume free views.  Compares latency and fp32-anchored accuracy
against the current integrated path on identical weights.
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
    parser.add_argument('--output', default='.tmp/br-attention-chain.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core import vendor_shims
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.modules.bs_roformer.transformer as transformer
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from pymss_core.vendor_shims import _rope_chunked, prepare_model_for_vendor_rope

    calls = {'count': 0}
    real_apply = vendor_shims._supa_rope_apply

    def counting_apply(x, cos, sin):
        calls['count'] += 1
        return real_apply(x, cos, sin)

    vendor_shims._supa_rope_apply = counting_apply

    def rope_bhqd(rotary, q, k):
        b, h, n, d = q.shape
        cos, sin = transformer.cached_rotary_cos_sin(rotary, n, q.device, torch.float32)
        cf = torch.cat((cos, cos), dim=-1).reshape(1, n, d).contiguous()
        sf = torch.cat((sin, sin), dim=-1).reshape(1, n, d).contiguous()
        outputs = []
        for tensor in (q, k):
            outputs.append(_rope_chunked(tensor.reshape(b * h, n, d), cf, sf, b * h).view(b, h, n, d))
        return outputs

    original_forward = transformer.Attention.forward

    def regroup_forward(self, x):
        heads = self.heads
        dim_head = self.to_qkv.out_features // (3 * heads)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        b, n, _ = qkv.shape
        trio = qkv.view(b, n, 3, heads, dim_head).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = trio[0], trio[1], trio[2]              # (B, H, S, D) contiguous
        if self.rotary_embed is not None:
            q, k = rope_bhqd(self.rotary_embed, q, k)
        if self.pope_embed is not None:
            raise NotImplementedError('probe path covers rotary only')
        out = self._attention(q, k, v)
        return self.to_out((out.transpose(1, 2) * self.to_gates(x).sigmoid().unsqueeze(-1)).flatten(start_dim=-2))

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    prepared = prepare_model_for_vendor_rope(model)
    assert len(prepared) == 24, prepared
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
        current, _ = timed_forward()
        current = current.float()

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
        for label, forward in (('current', original_forward), ('regroup', regroup_forward),
                               ('current_b', original_forward)):
            transformer.Attention.forward = forward
            calls['count'] = 0
            try:
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
                    'rel_to_current': round(float((output.float() - current).norm() / current.norm()), 6),
                    'finite': bool(output.isfinite().all()),
                    'rope_calls': calls['count'],
                }
                print('CHAIN_RESULT', json.dumps(results[label]), label, flush=True)
            finally:
                transformer.Attention.forward = original_forward

    vendor_shims._supa_rope_apply = real_apply
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print('CHAIN_ARTIFACT', str(output_path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
