"""In-model AB/BA compare: native supa_rope_apply vs current rotary path.

Validated contract from vendor_fusion_contract_probe (rope-apply-bhsd-neox,
rel-L2 0.157%): x = (batch*heads, seq, head_dim) interleaved pairs, tables =
fp32 (1, seq, head_dim) built as concat([half, half], -1). This probe swaps
transformer.rotate_qk_fast_bnhd for a vendor implementation and compares
full-model latency plus elementwise output error against the original path.
The returned tensors are transposed views so Attention.forward's transpose(1,2)
yields the contiguous (B, H, S, D) the SDPA shim reshapes for free.
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
    parser.add_argument('--output', default='.tmp/br-rope-model-ab.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    import pymss_core.modules.bs_roformer.transformer as transformer
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer

    original_rotate = transformer.rotate_qk_fast_bnhd
    calls = {'count': 0}
    # supa_rope_apply launch contract: dim0 (batch*heads) must be <= 1024.
    # Verified by vendor_rope_shape_sweep2: (496,862)/(124,862)/(862,62)/(16,4096)
    # pass; (2048,17)/(1724,62)/(6896,62) fail or hang. Sequence up to 4096 fine.
    rope_max_batch = 1024

    def rope_chunked(t, cf, sf, bh):
        if bh <= rope_max_batch:
            return torch.ops.supa.supa_rope_apply(t, cf, sf)
        chunks = -(-bh // rope_max_batch)
        step = -(-bh // chunks)
        return torch.cat([torch.ops.supa.supa_rope_apply(piece, cf, sf)
                          for piece in t.split(step, dim=0)], dim=0)

    def make_vendor_rotate(table_dtype=torch.float32):
        def vendor_rotate(rotary_embed, q, k):
            b, s, h, d = q.shape
            bh = b * h
            cos, sin = transformer.cached_rotary_cos_sin(rotary_embed, s, q.device, torch.float32)
            cf = torch.cat([cos, cos], dim=-1).reshape(1, s, d).to(table_dtype).contiguous()
            sf = torch.cat([sin, sin], dim=-1).reshape(1, s, d).to(table_dtype).contiguous()
            # Requires weights permuted per head to [evens|odds] output order so q/k
            # arrive already in the op's split-half pairing (see permute_qkv_rows).
            qf = q.permute(0, 2, 1, 3).reshape(bh, s, d)
            kf = k.permute(0, 2, 1, 3).reshape(bh, s, d)
            if not qf.is_contiguous():
                qf = qf.contiguous()
            if not kf.is_contiguous():
                kf = kf.contiguous()
            q_out = rope_chunked(qf, cf, sf, bh)
            k_out = rope_chunked(kf, cf, sf, bh)
            calls['count'] += 2
            return q_out.view(b, h, s, d).transpose(1, 2), k_out.view(b, h, s, d).transpose(1, 2)
        return vendor_rotate

    heads = 8
    dim_head = 64
    evens_odds = list(range(0, dim_head, 2)) + list(range(1, dim_head, 2))
    row_index = ([head * dim_head + j for head in range(heads) for j in evens_odds]
                 + [heads * dim_head + head * dim_head + j for head in range(heads) for j in evens_odds]
                 + list(range(2 * heads * dim_head, 3 * heads * dim_head)))

    def permute_qkv_rows():
        saved = []
        for name, module in model.named_modules():
            if name.endswith('.to_qkv') or name == 'to_qkv':
                saved.append((module, module.weight.data, None if module.bias is None else module.bias.data))
                module.weight.data = module.weight.data[row_index]
                if module.bias is not None:
                    module.bias.data = module.bias.data[row_index]
        assert saved, 'no to_qkv modules found'
        return saved

    def restore_qkv_rows(saved):
        for module, weight, bias in saved:
            module.weight.data = weight
            if bias is not None:
                module.bias.data = bias

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, int(44100 * args.seconds), device='supa:0') * 0.1

    def forward():
        start = time.perf_counter()
        output = model(x)
        torch.supa.synchronize()
        return output, (time.perf_counter() - start) * 1000

    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.supa.synchronize()
        reference, _ = forward()
        reference = reference.float()
        # fp32 anchor: same weights, full fp32 forward (weights stay a master copy).
        fp32_model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                                time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0')
        fp32_model.load_state_dict({key: value.float() for key, value in model.state_dict().items()})

        def fp32_forward():
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
                start = time.perf_counter()
                output = fp32_model(x)
                torch.supa.synchronize()
                return output, (time.perf_counter() - start) * 1000
            finally:
                functional.scaled_dot_product_attention = shim
        fp32_out, fp32_ms = fp32_forward()
        fp32_out = fp32_out.float()
        print('ROPE_ANCHOR', json.dumps({'fp32_ms': round(fp32_ms, 1),
                                         'original_rel_to_fp32': round(float((reference - fp32_out).norm() / fp32_out.norm()), 6)}), flush=True)
        results = {}
        variants = [('original', original_rotate, False),
                    ('vendor_permqkv', make_vendor_rotate(), True),
                    ('original_b', original_rotate, False)]
        for label, rotate, permuted in variants:
            transformer.rotate_qk_fast_bnhd = rotate
            saved_rows = permute_qkv_rows() if permuted else None
            try:
                calls['count'] = 0
                for _ in range(2):
                    model(x)
                torch.supa.synchronize()
                samples = []
                for _ in range(args.repeat):
                    output, ms = forward()
                    samples.append(ms)
            finally:
                if saved_rows is not None:
                    restore_qkv_rows(saved_rows)
            error = float((output.float() - reference).norm() / reference.norm())
            to_fp32 = float((output.float() - fp32_out).norm() / fp32_out.norm())
            max_rel = float(((output.float() - reference).abs() / reference.abs().clamp_min(1e-6)).max())
            results[label] = {'median_ms': round(statistics.median(samples), 1),
                              'samples_ms': [round(value, 1) for value in samples],
                              'relative_l2': round(error, 6),
                              'relative_l2_to_fp32': round(to_fp32, 6),
                              'max_relative': round(max_rel, 6),
                              'finite': bool(output.isfinite().all()), 'vendor_calls': calls['count']}
            print('ROPE_MODEL', json.dumps(results[label]), label, flush=True)
        transformer.rotate_qk_fast_bnhd = original_rotate

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print('ROPE_MODEL_ARTIFACT', str(output_path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
