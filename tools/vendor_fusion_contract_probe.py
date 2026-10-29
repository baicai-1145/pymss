"""One isolated native-fusion contract check per process; not a speed benchmark."""
from __future__ import annotations
import argparse
import json
import math
import time


CASES = [
    'rms-infer-bf16-w32-2d', 'rms-infer-bf16-w16-3d', 'rms-infer-f32-w32-2d',
    'rms-native-bf16-w16-2d', 'rms-pool-bf16-w16-2d', 'rms-infer-bf16-w32-3d',
    'rope-apply-half', 'rope-apply-full', 'rope-apply-4d-half', 'rope-infer-v2',
    'mlp-out-row', 'mlp-out-col', 'mlp-pack-row', 'mlp-out-buffer-row',
    'rope-apply-bsh-neox', 'rope-apply-bsh-pair', 'rope-apply-bshd-neox',
    'rms-infer-bf16-w32-3d-unit', 'rms-infer-bf16-w32-3d-d512',
    'rope-apply-sbd-neox', 'rope-apply-sbd-half', 'rope-apply-sbhd-half',
    'rope-apply-bhsd-neox', 'rope-apply-bhsd-pair',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--head-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=17)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import torch_br  # noqa: F401
    torch.supa.set_device(0)
    torch.manual_seed(0)
    case = args.case
    try:
        with torch.no_grad():
            if case.startswith('rms-'):
                dtype = torch.float32 if '-f32-' in case else torch.bfloat16
                wdtype = torch.float32 if '-w32-' in case else torch.bfloat16
                hidden = 512 if case.endswith('-d512') else 384
                shape = (2, 32, hidden) if '-3d' in case else (64, hidden)
                x = torch.randn(shape, device='supa:0', dtype=dtype)
                weight = (torch.randn(hidden, device='supa:0') * 0.2 + 1).to(wdtype)
                if case.endswith('-unit'):
                    weight.fill_(1)
                reference = F.rms_norm(x.float(), (hidden,), weight.float(), 1e-12)
                if case.startswith('rms-infer-'):
                    function = lambda: torch.ops.supa.supa_rmsnorm_infer(x, weight, 1e-12)
                elif case.startswith('rms-native-'):
                    function = lambda: torch.ops.supa.supa_rmsnorm(x, weight, 1e-12, False)
                else:
                    buffer = torch.empty_like(x)
                    function = lambda: torch.ops.supa.rmsnorm_with_pool(buffer, x, weight, 1e-12)
            elif case.startswith('rope-'):
                batch, seq, heads, dim = 2, args.seq_len, 8, args.head_size
                x = torch.randn(batch * seq, heads, dim, device='supa:0', dtype=torch.bfloat16)
                inverse = 1 / (10000 ** (torch.arange(0, dim, 2, device='supa:0').float() / dim))
                angles = torch.arange(seq, device='supa:0')[:, None] * inverse
                c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
                cb, sb = c.repeat(batch, 1)[:, None, :], s.repeat(batch, 1)[:, None, :]
                left, right = x[..., :dim//2].float(), x[..., dim//2:].float()
                reference = torch.cat((left * cb.float() - right * sb.float(),
                                       right * cb.float() + left * sb.float()), -1)
                if case.startswith('rope-apply-bhsd'):
                    x = x.reshape(batch, seq, heads, dim).transpose(1, 2).reshape(batch * heads, seq, dim).contiguous()
                    if case.endswith('-pair'):
                        ev, od = x[..., ::2].float(), x[..., 1::2].float()
                        reference = torch.stack((ev*c[None].float()-od*s[None].float(),
                                                 od*c[None].float()+ev*s[None].float()), -1).flatten(-2)
                        cf, sf = c.float().repeat_interleave(2, dim=-1)[None], s.float().repeat_interleave(2, dim=-1)[None]
                    else:
                        reference = reference.reshape(batch, seq, heads, dim).transpose(1, 2).reshape_as(x)
                        cf, sf = c.float().repeat(1, 2)[None], s.float().repeat(1, 2)[None]
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cf.contiguous(), sf.contiguous())
                elif case.startswith(('rope-apply-sbd', 'rope-apply-sbhd')):
                    x = x.reshape(batch, seq, heads, dim).transpose(0, 1).contiguous()
                    reference = reference.reshape(batch, seq, heads, dim).transpose(0, 1).contiguous()
                    if case.startswith('rope-apply-sbd-'):
                        x = x.flatten(1, 2)
                        reference = reference.flatten(1, 2)
                    if case.endswith('-half'):
                        cf, sf = c.float()[None], s.float()[None]
                    else:
                        cf, sf = c.float().repeat(1, 2)[None], s.float().repeat(1, 2)[None]
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cf.contiguous(), sf.contiguous())
                elif case.startswith('rope-apply-bsh'):
                    x = x.reshape(batch, seq, heads, dim)
                    if case.endswith('-pair'):
                        cf, sf = c.float().repeat_interleave(2, dim=-1)[None], s.float().repeat_interleave(2, dim=-1)[None]
                        ev, od = x[..., ::2].float(), x[..., 1::2].float()
                        reference = torch.stack((ev*c[None,:,None,:].float()-od*s[None,:,None,:].float(),
                                                 od*c[None,:,None,:].float()+ev*s[None,:,None,:].float()), -1).flatten(-2)
                    else:
                        cf, sf = c.float().repeat(1, 2)[None], s.float().repeat(1, 2)[None]
                        reference = reference.reshape_as(x)
                    if not case.startswith('rope-apply-bshd'):
                        x = x.flatten(-2)
                        reference = reference.flatten(-2)
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cf.contiguous(), sf.contiguous())
                elif case == 'rope-apply-half':
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cb.contiguous(), sb.contiguous())
                elif case == 'rope-apply-full':
                    cf, sf = cb.repeat(1, 1, 2), sb.repeat(1, 1, 2)
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cf, sf)
                elif case == 'rope-apply-4d-half':
                    x = x.reshape(batch, seq, heads, dim)
                    reference = reference.reshape_as(x)
                    cb4, sb4 = c[None, :, None, :].contiguous(), s[None, :, None, :].contiguous()
                    function = lambda: torch.ops.supa.supa_rope_apply(x, cb4, sb4)
                else:
                    q = x.reshape(batch * seq, heads * dim).clone()
                    key = q.clone()
                    position = torch.arange(seq, device='supa:0').repeat(batch).long()
                    function = lambda: torch.ops.supa.supa_rope_infer_v2(
                        q, key, s, c, position, dim, 'Half')[0].reshape_as(reference)
            else:
                x = torch.randn(64, 384, device='supa:0', dtype=torch.bfloat16)
                up = torch.randn(1536, 384, device='supa:0', dtype=torch.bfloat16) / (384 ** 0.5)
                down = torch.randn(384, 1536, device='supa:0', dtype=torch.bfloat16) / (1536 ** 0.5)
                ub = torch.randn(1536, device='supa:0', dtype=torch.bfloat16) * 0.1
                db = torch.randn(384, device='supa:0', dtype=torch.bfloat16) * 0.1
                reference = F.linear(F.gelu(F.linear(x.float(), up.float(), ub.float())), down.float(), db.float())
                if case == 'mlp-out-col':
                    uw, dw = up.t().contiguous(), down.t().contiguous()
                else:
                    uw, dw = up, down
                if case.startswith('mlp-out-'):
                    buffer = torch.empty(64, 384, device='supa:0', dtype=x.dtype)
                    def function():
                        intermediate = torch.ops.supa.fused_mlp_out_bias_gelu_fwd(buffer, x, uw, ub, dw, db)
                        return buffer if 'buffer' in case else intermediate
                else:
                    function = lambda: torch.ops.supa.fused_mlp_bias_gelu_fwd(x, uw, dw, ub)[0] + db
            torch.supa.synchronize()
            start = time.perf_counter()
            output = function()
            torch.supa.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            assert output.shape == reference.shape, (output.shape, reference.shape)
            difference = output.float() - reference
            fit_scale = (output.float() * reference).sum() / (reference * reference).sum()
            def metric(value):
                number = value.item()
                return number if math.isfinite(number) else None
            metrics = {'case': case, 'head_size': args.head_size, 'shape': list(output.shape),
                       'dtype': str(output.dtype), 'first_call_wall_ms': elapsed,
                       'fit_scale': metric(fit_scale),
                       'scaled_relative_l2': metric((output.float()-fit_scale*reference).norm()/reference.norm()),
                       'relative_l2': metric(difference.norm()/reference.norm()),
                       'max_relative': metric(difference.abs().max()/reference.abs().max()),
                       'finite': bool(output.isfinite().all())}
            valid = metrics['finite'] and metrics['relative_l2'] is not None and metrics['relative_l2'] < 0.02
            metrics['validation_passed'] = valid
            print('CONTRACT_RESULT', json.dumps(metrics, allow_nan=False), flush=True)
            return 0 if valid else 2
    except Exception as error:
        print('CONTRACT_ERROR', json.dumps({'case': case, 'type': type(error).__name__,
                                            'error': str(error)[-800:]}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
