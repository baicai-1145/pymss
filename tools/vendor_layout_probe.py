"""Capture real Biren tensor storage/layout metadata without timing claims."""
from __future__ import annotations
import collections
import contextlib
import io
import json
import os
from pathlib import Path
import re


def main():
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    import torch_br.supa._debug as debug
    from pymss_core.device import prepare_device_alias
    prepare_device_alias('biren')
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    import pymss_core.modules.bs_roformer.transformer as transformer
    records = []
    counters = collections.Counter()

    def select(info):
        if isinstance(info, dict):
            answer = {}
            for key, value in info.items():
                name = str(key)
                if isinstance(value, dict):
                    nested = select(value)
                    if nested:
                        answer[name] = nested
                elif 'ptr' not in name.lower() and any(word in name.lower() for word in
                     ('layout', 'mem', 'arch', 'region', 'format', 'numa', 'dtype', 'stride', 'offset', 'size', 'shape')):
                    answer[name] = value if isinstance(value, (str, int, float, bool, list, tuple, type(None))) else str(value)
            return answer
        if isinstance(info, torch.Tensor):
            return {'info_type': type(info).__name__}
        return re.sub(r'0x[0-9a-fA-F]+', '<ptr>', str(info))[:2400]

    def record(label, tensor):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            info = debug.get_tensor_info(tensor)
        row = {'label': label, 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
               'dtype': str(tensor.dtype), 'contiguous': tensor.is_contiguous(),
               'storage_offset': tensor.storage_offset(), 'br_layout': str(getattr(tensor, 'br_layout', None)),
               'info': select(info), 'info_type': type(info).__name__}
        if capture.getvalue():
            row['debug_output'] = re.sub(r'0x[0-9a-fA-F]+', '<ptr>', capture.getvalue())[:2400]
        records.append(row)
        print('LAYOUT_TENSOR', json.dumps(row, default=str), flush=True)

    torch.manual_seed(0)
    model = BSRoformer(dim=384, depth=12, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval().to('supa:0').to(torch.bfloat16)
    x = torch.randn(1, 2, 441000, device='supa:0') * 0.1
    handles = []
    norm_count = qkv_count = 0
    for name, module in model.named_modules():
        if isinstance(module, transformer.RMSNorm) and (norm_count < 2 or name == 'final_norm'):
            norm_count += 1
            def before(mod, inputs, label=name):
                record('rms_input:' + label, inputs[0])
            def norm_after(mod, inputs, output, label=name):
                record('rms_output:' + label, output)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(norm_after))
        if name.endswith('to_qkv') and qkv_count < 2:
            qkv_count += 1
            def after(mod, inputs, output, label=name):
                record('linear_qkv_output:' + label, output)
            handles.append(module.register_forward_hook(after))
    rotate_original = transformer.rotate_qk_fast_bnhd
    operator_original = torch.ops.supa.sudnn_scaled_dot_product_attention_fwd

    def rotate(embed, q, k):
        counters['rotary'] += 1
        if counters['rotary'] <= 2:
            record('rotary_input_q:' + str(counters['rotary']), q)
            record('rotary_input_k:' + str(counters['rotary']), k)
        out = rotate_original(embed, q, k)
        if counters['rotary'] <= 2:
            record('rotary_output_q:' + str(counters['rotary']), out[0])
        return out

    def native(q, k, v, *args, **kwargs):
        counters['native'] += 1
        if counters['native'] <= 2:
            for label, tensor in (('q', q), ('k', k), ('v', v)):
                record('fmha_operand_' + label + ':' + str(counters['native']), tensor)
        return operator_original(q, k, v, *args, **kwargs)

    transformer.rotate_qk_fast_bnhd = rotate
    torch.ops.supa.sudnn_scaled_dot_product_attention_fwd = native
    try:
        with torch.no_grad():
            output = model(x)
            torch.supa.synchronize()
            print('LAYOUT_MODEL_FINITE', bool(output.isfinite().all()), flush=True)
        destination = Path('/tmp/code/pymss/.tmp/br-runtime-layouts.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(records, indent=2, default=str))
        print('LAYOUT_ARTIFACT', str(destination), len(records), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        transformer.rotate_qk_fast_bnhd = rotate_original
        torch.ops.supa.sudnn_scaled_dot_product_attention_fwd = operator_original
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
