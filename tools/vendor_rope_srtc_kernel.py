"""Custom G-mode RoPE kernel via SRTC (runtime compile) + driver-launch from Python.

Replaces the chunked supa_rope_apply path for the freq side (6896,62,64), where
7 mandatory chunks cost 12.4 ms because the vendor op is ~4x less efficient on
short-seq shapes.  The kernel is elementwise split-half rotation with fp32 math:

    out[b, s, i]      = x[b, s, i]*cos[s,i]   - x[b, s, i+32]*sin[s,i]
    out[b, s, i+32]   = x[b, s, i+32]*cos[s,i] + x[b, s, i]*sin[s,i]

Tables are fp32 (1, SEQ, 64) = concat([half, half]).  Input channels are
already [evens|odds] ordered (permuted to_qkv weights).  Layout gate: linear
indexing is only used after get_tensor_info confirms plain single-region UMA
layout on input AND output; anything else raises (no silent fallback).
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def find_libraries():
    roots = ['/usr/local/birensupa', '/usr/local/lib/python3.10/dist-packages/torch_br/lib']
    candidates = []
    for root in roots:
        candidates += subprocess.run(['find', root, '-name', '*.so'], capture_output=True, text=True).stdout.split()
    rtc_lib = None
    drv_candidates = []
    for path in candidates:
        if 'inj' in path.lower() or 'supti' in path.lower():
            continue
        try:
            symbols = subprocess.run(['nm', '-D', '--defined-only', path], capture_output=True, text=True).stdout
        except FileNotFoundError:
            continue
        if rtc_lib is None and 'surtcCreateProgram' in symbols and 'surtcCompileProgram' in symbols:
            rtc_lib = path
        if 'sudrvModuleLoadData' in symbols and 'sudrvModuleLaunchKernel' in symbols \
                and 'sudrvModuleGetFunction' in symbols:
            drv_candidates.append(path)
    if drv_candidates:
        drv_candidates.sort(key=lambda p: (('drv' not in p.lower()), len(p), p))
        return rtc_lib, drv_candidates[0]
    return rtc_lib, (drv_candidates[0] if drv_candidates else None)


KERNEL_SOURCE = r'''
{include_line}
struct FloatBits { unsigned int u; float f; };
__device__ static inline float bf16_to_f32(unsigned short bits) {
    FloatBits cvt; cvt.u = (unsigned int)bits << 16; return cvt.f;
}
__device__ static inline unsigned short f32_to_bf16(float value) {
    FloatBits cvt; cvt.f = value;
    unsigned int u = cvt.u;
    unsigned int rounded = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return (unsigned short)rounded;
}
extern "C" __global__ void rope_halves_kernel(
    const unsigned short* __restrict__ x,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    unsigned short* __restrict__ out,
    long long pairs,
    long long seq)
{
    long long index = (long long)block_idx.x * block_dim.x + thread_idx.x;
    long long stride = (long long)block_dim.x * grid_dim.x;
    for (long long p = index; p < pairs; p += stride) {
        long long rs = p >> 5;
        long long i = p & 31;
        long long s = rs % seq;
        long long base = rs * 64;
        float c = cos_t[s * 64 + i];
        float sn = sin_t[s * 64 + i];
        float a = bf16_to_f32(x[base + i]);
        float b = bf16_to_f32(x[base + i + 32]);
        out[base + i] = f32_to_bf16(a * c - b * sn);
        out[base + i + 32] = f32_to_bf16(b * c + a * sn);
    }
}
'''

#: SRTC has no std headers; try lean device includes first, then bare source.
_INCLUDE_VARIANTS = ('#include <supa_device.h>', '#include <device/supa_device.h>', '')


class SrtcRope:
    def __init__(self):
        import torch
        from pymss_core.device import prepare_device_alias
        prepare_device_alias('biren')
        self.torch = torch
        rtc_path, drv_path = find_libraries()
        if not rtc_path or not drv_path:
            raise RuntimeError(f'SRTC libs missing: rtc={rtc_path} drv={drv_path}')
        self.rtc = ctypes.CDLL(rtc_path)
        self.drv = ctypes.CDLL(drv_path)
        self.module = None
        self.function = None
        self._compile()

    def _compile(self):
        failures = []
        for include_line in _INCLUDE_VARIANTS:
            source = KERNEL_SOURCE.replace('{include_line}', include_line)
            program = ctypes.c_void_p()
            source_c = ctypes.c_char_p(source.encode())
            name = ctypes.c_char_p(b'rope_halves_kernel')
            status = self.rtc.surtcCreateProgram(ctypes.byref(program), source_c, name,
                                                 ctypes.c_int(0), None, None)
            if status != 0:
                failures.append(f'create({include_line!r}) -> {status}')
                continue
            status = self.rtc.surtcCompileProgram(program, ctypes.c_int(0), None)
            if status == 0:
                break
            log_size = ctypes.c_size_t()
            self.rtc.surtcGetProgramLogSize(program, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value or 1)
            self.rtc.surtcGetProgramLog(program, log)
            failures.append(f'compile({include_line!r}) -> {status}: {log.value.decode(errors="replace")[-300:]}')
        else:
            raise RuntimeError('all SRTC variants failed: ' + ' | '.join(failures))
        size = ctypes.c_size_t()
        self.rtc.surtcGetSUBinSize(program, ctypes.byref(size))
        image = ctypes.create_string_buffer(size.value)
        self.rtc.surtcGetSUBin(program, image)
        self.module = ctypes.c_void_p()
        status = self.drv.sudrvModuleLoadData(ctypes.byref(self.module), image)
        if status != 0:
            raise RuntimeError(f'sudrvModuleLoadData -> {status}')
        self.function = ctypes.c_void_p()
        status = self.drv.sudrvModuleGetFunction(ctypes.byref(self.function), self.module,
                                                 ctypes.c_char_p(b'rope_halves_kernel'))
        if status != 0:
            raise RuntimeError(f'sudrvModuleGetFunction -> {status}')

    def layout_ok(self, tensor):
        import torch_br.supa._debug as debug
        info = debug.get_tensor_info(tensor)
        if isinstance(info, list):
            info = info[0] if info else {}
        layout = str(info.get('layout', ''))
        arch = str(info.get('mem_arch_enum', info.get('mem_arch', '')))
        regions = info.get('region_num', 99)
        return layout in ('BUFFER_ANY', 'COLMAJOR', 'ROWMAJOR') and 'UMA' in arch and int(regions) == 1, {
            'layout': layout, 'mem_arch': arch, 'region_num': regions}

    def run_debug_small(self):
        torch = self.torch
        device = torch.device('supa:0')
        rows, seq, dim = 2, 8, 64
        cos_t = torch.zeros(seq, dim, device=device)
        sin_t = torch.zeros(seq, dim, device=device)
        cos_t[:, :32] = 1.0; cos_t[:, 32:] = 1.0   # identity rotation
        sin_t.zero_()
        x = torch.ones(rows, seq, dim, device=device, dtype=torch.bfloat16)
        out = torch.full((rows, seq, dim), float('nan'), device=device, dtype=torch.bfloat16)
        gate = [self.layout_ok(t)[0] for t in (x, cos_t, sin_t, out)]
        print('SRTC_DEBUG gate', gate, 'function', bool(self.function), flush=True)
        try:
            self.launch(x, cos_t, sin_t, out)
            torch.supa.synchronize()
            values = out[0, 0, :6].float().tolist()
            print('SRTC_DEBUG out_first6', values, 'all_one', bool((out.float() == 1).all()), flush=True)
        except Exception as error:
            print('SRTC_DEBUG error', type(error).__name__, str(error)[:200], flush=True)

    def launch(self, x, cos_t, sin_t, out):
        torch = self.torch
        for tensor in (x, cos_t, sin_t, out):
            ok, info = self.layout_ok(tensor)
            if not ok:
                raise RuntimeError(f'unsafe layout for linear indexing: {info}')
        rows, seq, dim = x.shape
        pairs = rows * seq * 32
        block = 256
        grid = min((pairs + block - 1) // block, 4096)
        pointers = [
            ctypes.c_void_p(int(torch.ops.supa._get_data_ptr(x))),
            ctypes.c_void_p(int(torch.ops.supa._get_data_ptr(cos_t))),
            ctypes.c_void_p(int(torch.ops.supa._get_data_ptr(sin_t))),
            ctypes.c_void_p(int(torch.ops.supa._get_data_ptr(out))),
        ]
        pairs_c = ctypes.c_longlong(pairs)
        seq_c = ctypes.c_longlong(seq)
        args = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in
                                       (*pointers, pairs_c, seq_c)])
        stream = ctypes.c_void_p(0)
        torch.supa.synchronize()
        status = self.drv.sudrvModuleLaunchKernel(self.function,
                                                  ctypes.c_uint(grid), 1, 1,
                                                  ctypes.c_uint(block), 1, 1,
                                                  ctypes.c_uint(0), stream, args, None)
        torch.supa.synchronize()
        if status != 0:
            raise RuntimeError(f'sudrvModuleLaunchKernel -> {status}')
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.tmp/br-rope-srtc.json')
    args = parser.parse_args()
    os.environ.setdefault('PYMSS_SDPA_BACKEND', 'sudnn')
    import torch
    kernel = SrtcRope()
    print('SRTC_COMPILED', flush=True)
    kernel.run_debug_small()
    device = torch.device('supa:0')
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    results = []

    def run(tag, rows, seq):
        angles = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
        cf = torch.cat((angles.cos(), angles.cos()), -1)[None].contiguous().reshape(seq, 64).contiguous()
        sf = torch.cat((angles.sin(), angles.sin()), -1)[None].contiguous().reshape(seq, 64).contiguous()
        x = (torch.randn(rows, seq, 64, device=device) * 0.5).to(torch.bfloat16).contiguous()
        op = torch.ops.supa.supa_rope_apply
        if rows <= 1024:
            ref = op(x, cf[None].contiguous(), sf[None].contiguous())
        else:
            step = 985
            ref = torch.cat([op(p, cf[None].contiguous(), sf[None].contiguous())
                             for p in x.split(step, dim=0)], dim=0)
        torch.supa.synchronize()
        out = torch.empty_like(x)
        record = {'tag': tag, 'rows': rows, 'seq': seq}
        try:
            kernel.launch(x, cf, sf, out)
            record['rel_l2_vs_apply'] = round(float((out.float() - ref.float()).norm() / ref.float().norm()), 6)
            record['finite'] = bool(out.isfinite().all())
            for _ in range(3):
                kernel.launch(x, cf, sf, out)
            torch.supa.synchronize()
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                kernel.launch(x, cf, sf, out)
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['kernel_median_ms'] = round(statistics.median(samples), 3)
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                if rows <= 1024:
                    op(x, cf[None].contiguous(), sf[None].contiguous())
                else:
                    step = 985
                    torch.cat([op(p, cf[None].contiguous(), sf[None].contiguous())
                               for p in x.split(step, dim=0)], dim=0)
                torch.supa.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            record['apply_median_ms'] = round(statistics.median(samples), 3)
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {str(error)[:300]}'
        results.append(record)
        print('SRTC', json.dumps(record), flush=True)

    run('time', 496, 862)
    run('freq', 6896, 62)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print('SRTC_ARTIFACT', str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
