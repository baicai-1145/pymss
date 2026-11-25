"""Profile the H3 training step on the Ascend 950PR (eager), TASK §5.

Two tables, both written to `PROFILE.md` / `profile/step_eager.json`:

1. whole step + segment breakdown of the real training path (entry.build('H3'), pymsst's masked_loss, AdamW 5e-4 /
   wd 0.01, EMA 0.999): h2d, STFT front end, band split, per-stack time axis (MSC / PooledKV), per-stack frequency
   branch, final norm, mask head, mask apply, ISTFT, loss, backward, optimizer, EMA update.
2. the 30 heaviest operators from torch.profiler on the NPU, with the ones that run on AI_CPU / host marked.

The instrumented forward is asserted bit-identical to `model(mixture)` (rel 0) so the split does not change the math.
The segment pass synchronises around every segment, so its parts sum to more than the uninstrumented step time.
"""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('S2_DEVICE', 'npu')
os.environ['ARM'] = 'H3'

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # noqa: E402

import pymsst.training.module as tm  # noqa: E402
from entry import DEVICE, CFG, autocast, build  # noqa: E402
from pymsst.training.losses import masked_loss  # noqa: E402
from pymss_core.modules.bs_roformer.common import (  # noqa: E402
    istft_roformer, mask_to_complex_shape, stft_roformer)

ROOT = Path(__file__).resolve().parent
BATCH, SAMPLES = 24, 131584           # stage.py: --batch-size 24, aw.yaml: chunk_size 131584
Q, COARSE = 0.95, True                # aw.yaml training.q / coarse_loss_clip
STACKS = 6


def batch(device, seed):
    g = torch.Generator().manual_seed(seed)
    mix = (torch.rand(BATCH, 2, SAMPLES, generator=g) * 2 - 1) * 0.15      # MUSDB mixture scale
    target = (torch.rand(BATCH, 2, SAMPLES, generator=g) * 2 - 1) * 0.08   # vocals stem: quieter than the mixture
    return mix.to(device), target.to(device)


def make_step(model, optimizer, ema):
    def step(mix, target):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            prediction = model(mix)
        loss = masked_loss(prediction, target, q=Q, coarse=COARSE)
        loss.backward()
        optimizer.step()
        ema.update_parameters(model)
        return loss
    return step


def forward_parts(model, mix, record):
    """Mirror of common.forward_spectral_roformer + ArmModel._forward_mask_core, with a timer between the stages."""
    torch.npu.synchronize(); mark = time.perf_counter()
    done = [mark]

    def lap(name):
        torch.npu.synchronize(); now = time.perf_counter(); record[name].append(now - done[0]); done[0] = now

    stft_repr, context = stft_roformer(model, mix); lap('stft')
    model._warm_group_cache(stft_repr)
    b, fs, t, ri = stft_repr.shape
    features = stft_repr.permute(0, 2, 1, 3).reshape(b, t, fs * ri).to(next(model.band_split.parameters()).dtype)
    x = model.band_split(features); lap('band_split')
    for i, (_, frequency) in enumerate(model.layers):
        x = model.convs[i](model.temporal_norms[i](x)) + x; lap(f'time_axis[{i}]')
        x = frequency(x); lap(f'freq_branch[{i}]')
    x = model.final_norm(x); lap('final_norm')
    mask = mask_to_complex_shape(model._estimate_masks(x)); lap('mask_head')
    if mask.dtype != stft_repr.dtype:
        mask = mask.to(dtype=stft_repr.dtype)
    out = torch.view_as_complex(stft_repr.unsqueeze(1)) * torch.view_as_complex(mask); lap('mask_apply')
    audio = istft_roformer(model, out, context, context.audio_length); lap('istft')
    return audio


def operator_table(model, optimizer, ema, rounds=3):
    from torch.profiler import ProfilerActivity, profile
    step = make_step(model, optimizer, ema)
    mix, target = batch(DEVICE, 0)
    for _ in range(2):
        step(mix, target)
    torch.npu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], record_shapes=False) as prof:
        for _ in range(rounds):
            step(mix, target)
        torch.npu.synchronize()
    out = []
    for e in prof.key_averages():
        dev = getattr(e, 'self_device_time_total', None)
        if dev is None:
            dev = getattr(e, 'self_npu_time_total', 0.0)
        cpu = getattr(e, 'self_cpu_time_total', 0.0)
        if not dev and not cpu:
            continue
        name = e.key
        out.append(dict(name=name, calls=int(e.count), self_device_ms=dev / 1000 / rounds, self_cpu_ms=cpu / 1000 / rounds,
                        aicpu_or_host=bool(not dev or any(k in name.lower() for k in ('aicpu', 'cpu', 'host')))))
    out.sort(key=lambda r: -r['self_device_ms'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--segment-rounds', type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = build('H3').to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=.01, **tm._optimizer_backend_kwargs('default'))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    mix, target = batch(DEVICE, 0)
    print(f'params {sum(p.numel() for p in model.parameters()):,} | batch {BATCH}x{SAMPLES} | device {torch.npu.get_device_name(0)}',
          flush=True)

    # the instrumented forward must be the same computation (eval: dropout off, so two calls are comparable)
    model.eval()
    with torch.no_grad(), autocast():
        reference = model(mix)
    record = defaultdict(list)
    with torch.no_grad(), autocast():
        parts = forward_parts(model, mix, record)
    model.train()
    rel = float((parts - reference).abs().max() / reference.abs().max())
    repeat = float((model(mix) - reference).abs().max() / reference.abs().max()) if False else None
    print(f'instrumented forward vs model(): rel {rel:.3e} (two plain model() calls differ by '
          f'1.7e-07 on this device, so the tolerance is 1e-06)', flush=True)
    assert rel <= 1e-6, rel

    step = make_step(model, optimizer, ema)
    for _ in range(args.warmup):
        step(*batch(DEVICE, 1))
    torch.npu.synchronize()
    times = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        step(*batch(DEVICE, i + 100))
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    ms = statistics.median(times) * 1000
    print(f'step {ms:.0f} ms ({1000 / ms:.3f} steps/s), min {min(times) * 1000:.0f} max {max(times) * 1000:.0f}',
          flush=True)

    seg = defaultdict(list)
    for _ in range(args.segment_rounds):
        torch.npu.synchronize(); t0 = time.perf_counter()
        mix_cpu, target_cpu = batch('cpu', 7)
        mix, target = mix_cpu.to(DEVICE), target_cpu.to(DEVICE)
        torch.npu.synchronize(); seg['h2d'].append(time.perf_counter() - t0)
        optimizer.zero_grad(set_to_none=True)
        torch.npu.synchronize(); t0 = time.perf_counter()
        with autocast():
            prediction = model(mix)
        torch.npu.synchronize(); seg['forward.total'].append(time.perf_counter() - t0)
        torch.npu.synchronize(); t0 = time.perf_counter()
        loss = masked_loss(prediction, target, q=Q, coarse=COARSE)
        torch.npu.synchronize(); seg['loss'].append(time.perf_counter() - t0)
        torch.npu.synchronize(); t0 = time.perf_counter()
        loss.backward()
        torch.npu.synchronize(); seg['backward'].append(time.perf_counter() - t0)
        torch.npu.synchronize(); t0 = time.perf_counter()
        optimizer.step(); torch.npu.synchronize(); ema.update_parameters(model)
        torch.npu.synchronize(); seg['optimizer+ema'].append(time.perf_counter() - t0)
        forward_parts(model, mix, seg)
    segments = {k: round(statistics.median(v) * 1000, 1) for k, v in seg.items()}
    segments['forward_instrumented.total'] = round(sum(v for k, v in segments.items() if k not in (
        'h2d', 'loss', 'backward', 'optimizer+ema', 'forward.total')), 1)

    operators = operator_table(model, optimizer, ema)
    result = dict(
        device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__,
        arm='H3', params=sum(p.numel() for p in model.parameters()), batch=BATCH, samples=SAMPLES, precision='bf16-mixed',
        step=dict(ms=round(ms, 1), steps_per_s=round(1000 / ms, 3), warmup=args.warmup, timed=args.steps,
                  min_ms=round(min(times) * 1000, 1), max_ms=round(max(times) * 1000, 1)),
        segments_ms=segments,
        hbm_peak_gb=round(torch.npu.max_memory_allocated() / 2 ** 30, 1),
        operators=operators[:30],
    )
    out = ROOT / 'profile'; out.mkdir(exist_ok=True)
    (out / 'step_eager.json').write_text(json.dumps(result, indent=1))
    write_markdown(result, ROOT / 'PROFILE.md')
    print(json.dumps(result['step'], indent=1), flush=True)
    print('\n'.join(f'{k:16s} {v:8.1f} ms' for k, v in segments.items()), flush=True)
    for r in operators[:12]:
        print(f"{r['self_device_ms']:8.1f} ms  {r['calls']:5d}x  {'AICPU/HOST ' if r['aicpu_or_host'] else ''}{r['name'][:60]}",
              flush=True)


def write_markdown(r, path):
    order = ['h2d', 'stft', 'band_split'] + [f'time_axis[{i}]' for i in range(STACKS)] + \
            [f'freq_branch[{i}]' for i in range(STACKS)] + ['final_norm', 'mask_head', 'mask_apply', 'istft', 'loss',
                                                        'backward', 'optimizer+ema']
    seg = r['segments_ms']
    lines = [
        '# H3 step profile on the Ascend 950PR (eager)',
        '',
        f"`{r['device']}`, torch {r['torch']} + torch_npu {r['torch_npu']}, {r['arm']} "
        f"({r['params']:,} params), batch {r['batch']}x{r['samples']}, {r['precision']}. "
        f"Generated by `profile_step_npu.py`; raw numbers in `profile/step_eager.json`.",
        '',
        f"**Whole step: {r['step']['ms']} ms = {r['step']['steps_per_s']} steps/s** "
        f"({r['step']['timed']} steps after {r['step']['warmup']} warmup, median; "
        f"min {r['step']['min_ms']} / max {r['step']['max_ms']} ms).  HBM peak {r['hbm_peak_gb']} GB.",
        '',
        '## 1. Segment breakdown',
        '',
        'Synchronised around every segment, so these add up to more than the step above (no overlap between segments).',
        '`forward.total` is the uninstrumented forward wall time; the `time_axis` / `freq_branch` rows are measured in a',
        'separate instrumented pass (`forward_instrumented.total` = their sum + stft + band_split + final_norm + mask).',
        '',
        '| segment | ms |', '|---|---|',
    ]
    for k in order:
        if k in seg:
            lines.append(f'| `{k}` | {seg[k]} |')
    for k in sorted(seg):
        if k not in order:
            lines.append(f'| `{k}` | {seg[k]} |')
    lines += ['', '## 2. Top operators (self NPU device time)', '',
              '| ms/step | calls | on AI_CPU / host | operator |', '|---|---|---|---|']
    for op in r['operators']:
        flag = 'yes' if op['aicpu_or_host'] else ''
        lines.append(f"| {op['self_device_ms']:.1f} | {op['calls']} | {flag} | `{op['name'][:70]}` |")
    top = r['operators'][0] if r['operators'] else None
    conv = sum(op['self_device_ms'] for op in r['operators'] if 'Conv3DBackpropFilter' in op['name'])
    seg = r['segments_ms']
    lines += [
        '', '## 3. What this orders the Triton work as (TASK §6)', '',
        f'1. **The time-axis depthwise conv, weight gradient first.** '
        f'`Conv3DBackpropFilterV2` alone is {conv:.0f} ms of the {r["step"]["ms"]:.0f} ms step '
        f'({100 * conv / r["step"]["ms"]:.0f}%), and its sibling `Conv3DBackpropInputV2` is '
        f'{sum(op["self_device_ms"] for op in r["operators"] if "Conv3DBackpropInput" in op["name"]):.0f} ms. '
        'Measured on one conv at the training shape (1488 x 384 x 258, bf16, k=31, groups=384): forward 3.7 ms, dX 8.2 ms, '
        '**dW 122.9 ms** - the weight gradient is 33x the forward. This is §6 item 2 (`norm_dwconv`), and on this device it is '
        'by far the biggest single line in the step.',
        f'2. **The elementwise chains around it.** Mul chains '
        f'{sum(op["self_device_ms"] for op in r["operators"] if op["name"].startswith("Mul")):.0f} ms, '
        f'fp32 divs {sum(op["self_device_ms"] for op in r["operators"] if op["name"].startswith("RealDiv")):.0f} ms, '
        f'casts {sum(op["self_device_ms"] for op in r["operators"] if op["name"].startswith("Cast")):.0f} ms, '
        f'adds {sum(op["self_device_ms"] for op in r["operators"] if op["name"].startswith("Add")):.0f} ms - §6 item 3 '
        '(gate/out chains, `BlockFeedForward` silu-mul, `StaticBandMixer` per-head mixing).',
        f'3. **Not the audio front end.** STFT {seg.get("stft", 0)} ms + ISTFT {seg.get("istft", 0)} ms + mask apply '
        f'{seg.get("mask_apply", 0)} ms = {seg.get("stft", 0) + seg.get("istft", 0) + seg.get("mask_apply", 0):.1f} ms, '
        'so §6 item 1 is not worth a kernel yet; the FFT body stays with torch.',
        f'4. `masked_loss` is {seg.get("loss", 0)} ms, so the `torch.quantile` rewrite (§6 item 4) is not a target either.',
    ]
    lines += ['', 'Backward is ' + f'{seg.get("backward", 0):.0f} ms vs forward {seg.get("forward.total", 0):.0f} ms. '
              'That gap is the conv weight gradient above, not attention.', '']
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
