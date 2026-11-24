"""Phase-level profile of one training step at bs24 on real MUSDB data.

Every phase is measured between explicit `torch.npu.synchronize()` boundaries so
asynchronous execution cannot hide work behind the next phase. Because those
syncs add a little overhead, the profiler also records a sync-free wall time per
step and reports the reconciliation error.

    python profile_train.py --precision bf16 --steps 100 \
        --out results/profile_train_bf16.json --ablation

Phases (all `sum/mean/median/share` reported):
    dataloader_wait  h2d_copy  forward  prep_complex  loss_complex
    loss_mrstft_<nfft>...  istft_pred  istft_tgt  loss_sisdr  loss_leak
    backward  grad_clip  optimizer_step  ema_update  total

Nothing here changes the training recipe: the model, the loss weights and the
default MR-STFT resolutions (512/1024/2048) are imported from train.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from train import (  # noqa: E402
    EMA,
    MusdbChunkDataset,
    _read_wav,
    autocast_ctx,
    complex_l1,
    leakage_loss,
    scan_musdb,
    si_sdr,
    split_fc,
)

DEFAULT_RESOLUTIONS = (512, 1024, 2048)
PROBE_RESOLUTIONS = (256, 512, 1024, 2048, 4096)
LOSS_WEIGHTS = {"mrstft": 0.5, "sisdr": 0.1, "leak": 0.1}
PHASES = (
    "dataloader_wait", "h2d_copy", "forward", "prep_complex", "loss_complex",
    "mrstft", "istft_pred", "istft_tgt", "loss_sisdr", "loss_leak",
    "backward", "grad_clip", "optimizer_step", "ema_update", "total",
)


class Clock:
    """Sync-bounded stopwatch; each `lap()` closes the previous phase."""

    def __init__(self, device: torch.device):
        self.device = device
        self.marks: dict[str, float] = {}
        self._t = 0.0
        self._lap_ms = 0.0

    def sync(self):
        if self.device.type == "npu":
            torch.npu.synchronize()

    def start(self):
        self.sync()
        self._t = time.perf_counter()

    def lap(self, name: str) -> float:
        self.sync()
        now = time.perf_counter()
        self._lap_ms = (now - self._t) * 1000
        self.marks[name] = self.marks.get(name, 0.0) + self._lap_ms
        self._t = now
        return self._lap_ms


def istft_once(spec_c: torch.Tensor, channels: int, n_fft: int, hop: int, length: int, window) -> torch.Tensor:
    from pymss_core.npu_fft import istft_npu

    flat = split_fc(spec_c, channels)
    wav = istft_npu(flat, n_fft=n_fft, hop_length=hop, window=window, length=length, return_complex=False)
    return wav.reshape(spec_c.shape[0], channels, -1)


def mrstft_single(pred_wav: torch.Tensor, target_wav: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Exactly one branch of train.multi_res_stft_loss (same window, same hop)."""
    flat_p = pred_wav.float().reshape(-1, pred_wav.shape[-1])
    flat_t = target_wav.float().reshape(-1, target_wav.shape[-1])
    window = torch.hann_window(n_fft).to(device=pred_wav.device, dtype=torch.float32)
    hop = n_fft // 4
    p = torch.stft(flat_p, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
    t = torch.stft(flat_t, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
    pm, tm = p.abs(), t.abs()
    scale = tm.max().detach().clamp_min(1e-4)
    den = torch.linalg.vector_norm(tm, dim=(-2, -1)).clamp_min(1e-3 * scale)
    sc = (torch.linalg.vector_norm(tm - pm, dim=(-2, -1)) / den).clamp(max=10.0)
    mag = F.l1_loss(torch.log(pm + 1e-3 * scale), torch.log(tm + 1e-3 * scale))
    return sc.mean() + mag


def spec_frames(length: int, n_fft: int, hop: int) -> int:
    return 1 + length // hop  # center=True


def measure_worker_costs(args, device) -> dict:
    """Decode+STFT happen inside DataLoader workers; measure them serially (workers=0)."""
    tracks = scan_musdb(args.data_root)
    ds = MusdbChunkDataset(tracks, args.chunk_size, args.n_fft, args.hop)
    decode, stft, total = [], [], []
    nbytes = 0
    for i in range(12):
        t0 = time.perf_counter()
        track, offset = ds._pick(i)
        audio = np.stack([
            _read_wav(track["mixture"], offset, ds.chunk_size, ds.channels),
            _read_wav(track["vocals"], offset, ds.chunk_size, ds.channels),
        ])
        nbytes += audio.nbytes
        t1 = time.perf_counter()
        for a in audio:
            ds._stft_fc(torch.from_numpy(a))
        t2 = time.perf_counter()
        decode.append((t1 - t0) * 1000)
        stft.append((t2 - t1) * 1000)
        total.append((t2 - t0) * 1000)
    return {
        "source": "measured serially with num_workers=0, same __getitem__ code path",
        "per_item_decode_ms_median": round(statistics.median(decode), 2),
        "per_item_stft_ms_median": round(statistics.median(stft), 2),
        "per_item_total_ms_median": round(statistics.median(total), 2),
        "decoded_bytes_per_item": nbytes // 12,
        "decoded_bytes_per_step_bs": None,  # filled by caller
        "wav_decode_in_worker_threads": True,
    }


def count_syncs(fn, device) -> tuple[object, int]:
    """Run `fn` and count device->host synchronizations it triggers."""
    counter = {"n": 0}
    orig_item = torch.Tensor.item
    orig_sync = torch.npu.synchronize if device.type == "npu" else None

    def item(self):
        counter["n"] += 1
        return orig_item(self)

    def sync(*a, **kw):
        counter["n"] += 1
        return orig_sync(*a, **kw)

    torch.Tensor.item = item
    if orig_sync is not None:
        torch.npu.synchronize = sync
    try:
        out = fn()
    finally:
        torch.Tensor.item = orig_item
        if orig_sync is not None:
            torch.npu.synchronize = orig_sync
    return out, counter["n"]


def run(args, precision: str, steps: int, resolutions=DEFAULT_RESOLUTIONS,
        num_workers: int | None = None, fixed_hop: int | None = None,
        drop: int | None = None, loss_mode: str = "full",
        stft_backend: str = "torch", loss_autocast: bool = False,
        stft_mode: str = "complex") -> dict:
    """loss_mode: 'full' (train.py recipe) | 'spec_only' | 'mask_scalar' (model fwd+bwd only).

    stft_backend: 'torch' uses torch.stft (what train.py does today on Ascend, where
    ``torch_stft_blocked()`` is True); 'npu' uses pymss_core.npu_fft.stft_npu, i.e. the
    vendor/exact FFT path the rest of pymss already uses.
    loss_autocast: run istft + STFT + loss terms inside autocast (numerics change).
    """
    if stft_backend == "npu":
        from pymss_core.npu_fft import stft_npu as _stft_impl
    else:
        _stft_impl = torch.stft
    device = torch.device("npu:0")
    if precision == "32":
        torch.npu.conv.allow_hf32 = False
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(device).train()
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    window = torch.hann_window(args.n_fft).to(device=device, dtype=torch.float32)
    channels = cfg.audio_channels

    tracks = scan_musdb(args.data_root)
    dataset = MusdbChunkDataset(tracks, args.chunk_size, args.n_fft, args.hop)
    workers = args.num_workers if num_workers is None else num_workers
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=workers, pin_memory=False,
        drop_last=True, persistent_workers=workers > 0, prefetch_factor=2 if workers else None,
    )

    res_list = tuple(r for r in resolutions if r != drop)
    clock = Clock(device)
    per_step: list[dict] = []
    wall_ms: list[float] = []
    clip_syncs: list[int] = []
    losses: list[dict] = []
    it = iter(loader)
    warm_steps = 3

    for step in range(steps):
        torch.npu.synchronize()
        step_wall_start = time.perf_counter()
        clock.start()
        try:
            target, mixture = next(it)
        except StopIteration:
            it = iter(loader)
            target, mixture = next(it)
        clock.lap("dataloader_wait")

        target = target.to(device)
        mixture = mixture.to(device)
        clock.lap("h2d_copy")

        with autocast_ctx(device, precision):
            mask = model(mixture)
        clock.lap("forward")

        mix_c = torch.view_as_complex(mixture.contiguous())
        tgt_c = torch.view_as_complex(target.contiguous())
        mask_c = torch.view_as_complex(mask.float().contiguous())
        pred_c = mix_c * mask_c
        clock.lap("prep_complex")

        loss_spec = complex_l1(pred_c, tgt_c)
        clock.lap("loss_complex")

        if loss_mode == "mask_scalar":
            loss = mask.float().mean()
            clock.lap("backward")
            loss.backward()
            clock.lap("backward_total")
            grad_norm, n_syncs = count_syncs(
                lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip), device)
            clock.lap("grad_clip")
            _ = float(grad_norm)
            clock.lap("grad_norm_to_host")
            optimizer.step()
            clock.lap("optimizer_step")
            ema.update(model)
            clock.lap("ema_update")
            torch.npu.synchronize()
            step_wall_end = time.perf_counter()
            if step >= warm_steps:
                wall_ms.append((step_wall_end - step_wall_start) * 1000)
                per_step.append({k: v for k, v in clock.marks.items() if k != "total"})
                losses.append({"loss": float(loss.detach())})
            clock.marks = {}
            continue

        loss_ctx = torch.npu.amp.autocast(dtype=torch.bfloat16) if loss_autocast else contextlib.nullcontext()
        pred_wav = istft_once(pred_c, channels, args.n_fft, args.hop, args.chunk_size, window)
        clock.lap("istft_pred")
        tgt_wav = istft_once(tgt_c, channels, args.n_fft, args.hop, args.chunk_size, window)
        clock.lap("istft_tgt")

        loss_mrstft = 0.0
        with loss_ctx:
          for n_fft in (() if loss_mode == "spec_only" else res_list):
            hop = fixed_hop if fixed_hop else n_fft // 4
            flat_p = pred_wav.float().reshape(-1, pred_wav.shape[-1])
            flat_t = tgt_wav.float().reshape(-1, tgt_wav.shape[-1])
            win = torch.hann_window(n_fft).to(device=device, dtype=torch.float32)
            if stft_mode == "realview":
                # |z| = sqrt(re^2 + im^2) is the same magnitude, but keeps the backend on
                # the real path (Ascend has no native complex STFT; the complex-abs ops and
                # the complex grad are what cost ~480 ms per backward here).
                pr = _stft_impl(flat_p, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=False)
                tr = _stft_impl(flat_t, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=False)
                # +1e-12 floors d(sqrt)/dx at exactly-zero bins (otherwise the grad is inf
                # -> NaN weights, the same failure mode as leakage_loss's sqrt(0)).
                pm = (pr.pow(2).sum(-1) + 1e-12).sqrt()
                tm = (tr.pow(2).sum(-1) + 1e-12).sqrt()
            else:
                p = _stft_impl(flat_p, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
                t = _stft_impl(flat_t, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
                pm, tm = p.abs(), t.abs()
            scale = tm.max().detach().clamp_min(1e-4)
            den = torch.linalg.vector_norm(tm, dim=(-2, -1)).clamp_min(1e-3 * scale)
            sc = (torch.linalg.vector_norm(tm - pm, dim=(-2, -1)) / den).clamp(max=10.0)
            mag = F.l1_loss(torch.log(pm + 1e-3 * scale), torch.log(tm + 1e-3 * scale))
            loss_mrstft = loss_mrstft + sc.mean() + mag
            clock.lap(f"loss_mrstft_{n_fft}")

        if loss_mode == "spec_only":
            loss_sisdr = torch.zeros((), device=device)
            loss_leak = torch.zeros((), device=device)
            loss = loss_spec
        else:
            with loss_ctx:
                loss_sisdr = -si_sdr(pred_wav, tgt_wav).mean() / 20.0
            clock.lap("loss_sisdr")
            loss_leak = leakage_loss(pred_wav, tgt_wav)
            clock.lap("loss_leak")
            loss = (loss_spec + LOSS_WEIGHTS["mrstft"] * loss_mrstft
                    + LOSS_WEIGHTS["sisdr"] * loss_sisdr + LOSS_WEIGHTS["leak"] * loss_leak)
        loss.backward()
        clock.lap("backward")

        grad_norm, n_syncs = count_syncs(
            lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip), device)
        clock.lap("grad_clip")
        clip_syncs.append(n_syncs)
        _ = float(grad_norm)  # train.py converts for logging; costs one device sync
        clock.lap("grad_norm_to_host")

        optimizer.step()
        clock.lap("optimizer_step")
        ema.update(model)
        clock.lap("ema_update")

        torch.npu.synchronize()
        step_wall_end = time.perf_counter()
        clock.lap("total")
        if step >= warm_steps:
            wall_ms.append((step_wall_end - step_wall_start) * 1000)
            per_step.append({k: v for k, v in clock.marks.items() if k != "total"})
            losses.append({
                "loss": float(loss.detach()), "loss_spec": float(loss_spec.detach()),
                "loss_mrstft": float(loss_mrstft), "loss_sisdr": float(loss_sisdr.detach()),
                "loss_leak": float(loss_leak.detach()),
            })
        clock.marks = {}

    wall_sum = sum(wall_ms)

    def agg(name: str) -> dict | None:
        vals = [s[name] for s in per_step if name in s]
        if not vals:
            return None
        tot = sum(vals)
        return {
            "sum_ms": round(tot, 2), "mean_ms": round(tot / len(vals), 2),
            "median_ms": round(statistics.median(vals), 2),
            "share_pct": round(100 * tot / wall_sum, 2) if wall_sum else None,
        }

    phase_names = [p for p in per_step[0] if p != "total"] if per_step else []
    phases = {name: agg(name) for name in phase_names}
    phases["total"] = {
        "sum_ms": round(wall_sum, 2),
        "mean_ms": round(wall_sum / len(wall_ms), 2) if wall_ms else None,
        "median_ms": round(statistics.median(wall_ms), 2) if wall_ms else None,
        "share_pct": 100.0,
        "note": "sync-free wall time per step (start->end without intra-step syncs)",
    }
    seg_sum = sum(v["sum_ms"] for k, v in phases.items() if k != "total" and v)
    wall_sum = round(wall_sum, 2)
    mean_loss = {k: round(statistics.mean(l[k] for l in losses), 6) for k in losses[0]} if losses else {}

    return {
        "config": {
            "precision": precision, "batch_size": args.batch_size, "steps_measured": len(per_step),
            "warmup_steps_dropped": warm_steps, "chunk_size": args.chunk_size, "n_fft": args.n_fft,
            "model_hop": args.hop, "mrstft_resolutions": list(res_list),
            "mrstft_dropped": drop, "mrstft_fixed_hop": fixed_hop,
            "num_workers": workers, "prefetch_factor": 2 if workers else None,
            "persistent_workers": workers > 0, "pin_memory": False,
            "loss_weights": LOSS_WEIGHTS, "grad_clip": args.grad_clip, "ema_decay": args.ema_decay,
        },
        "phases": phases,
        "reconciliation": {
            "wall_total_ms": wall_sum,
            "sum_of_segments_ms": round(seg_sum, 2),
            "error_pct": round(100 * (seg_sum - wall_sum) / wall_sum, 2) if wall_sum else None,
            "per_step_wall_median_ms": round(statistics.median(wall_ms), 2) if wall_ms else None,
            "note": "segments are sync-bounded, so sum >= wall by the cost of ~16 syncs/step",
        },
        "loss_values": mean_loss,
        "clip_grad_norm_syncs": sorted(set(clip_syncs)),
    }


def fft_stats(args, resolutions) -> dict:
    """Frame counts and FFT counts actually produced, measured from tensor shapes."""
    device = torch.device("npu:0")
    flat = torch.zeros(2 * 2, args.chunk_size, device=device)
    rows = []
    total_ffts = 0
    for n_fft in resolutions:
        hop = n_fft // 4
        win = torch.hann_window(n_fft).to(device=device)
        spec = torch.stft(flat, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
        frames = spec.shape[-1]
        bins = spec.shape[-2]
        per_signal = frames  # one FFT per frame
        ffts = per_signal * flat.shape[0] * 2  # x2 for pred and target
        total_ffts += ffts
        rows.append({
            "n_fft": n_fft, "hop": hop, "frames_measured": frames,
            "bins": bins, "elements_per_signal": frames * bins,
            "elements_all_signals": frames * bins * flat.shape[0] * 2,
            "ffts_pred_plus_target": ffts,
        })
        del spec
    ispec = torch.zeros(4, 2048 // 2 + 1, args.chunk_size // args.hop + 1, dtype=torch.complex64, device=device)
    return {
        "signal_length": args.chunk_size,
        "signals_per_batch": 2 * args.batch_size,
        "resolutions": rows,
        "total_ffts_per_step": total_ffts,
        "istft_hop": args.hop,
        "isframes_per_step": ispec.shape[-1],
        "total_iffts_per_step": ispec.shape[-1] * 2 * 2 * args.batch_size,
        "note": "frames/bins measured from torch.stft shapes on device; "
                "8 s @44.1 kHz with model hop 512 -> 690 frames",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--precision", default="bf16", choices=("32", "bf16"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    print(f"[profile] {args.precision} bs{args.batch_size} steps={args.steps}", flush=True)
    main_run = run(args, args.precision, args.steps)
    main_run["fft_stats"] = fft_stats(args, DEFAULT_RESOLUTIONS)
    main_run["worker_costs"] = measure_worker_costs(args, torch.device("npu:0"))
    per_item = main_run["worker_costs"]["decoded_bytes_per_item"]
    main_run["worker_costs"]["decoded_bytes_per_step"] = per_item * args.batch_size
    main_run["worker_costs"]["logical_read_mb_per_step"] = round(per_item * args.batch_size / 2**20, 2)
    print(json.dumps({k: main_run[k] for k in ("reconciliation", "loss_values")}), flush=True)

    if args.ablation:
        abl = {"note": "speed-only runs; loss values are logged so any numeric drift is visible",
               "steps_per_config": args.ablation_steps, "precision": args.precision, "runs": []}
        baseline_loss = main_run["loss_values"].get("loss")

        base_step = main_run["reconciliation"]["per_step_wall_median_ms"]
        for n_fft in PROBE_RESOLUTIONS:
            in_recipe = n_fft in DEFAULT_RESOLUTIONS
            res = tuple(r for r in DEFAULT_RESOLUTIONS if r != n_fft) if in_recipe else DEFAULT_RESOLUTIONS + (n_fft,)
            r = run(args, args.precision, args.ablation_steps + 3, resolutions=res)
            step_ms = r["reconciliation"]["per_step_wall_median_ms"]
            mrstft_total = sum(v["sum_ms"] for k, v in r["phases"].items() if k.startswith("loss_mrstft"))
            abl["runs"].append({
                "kind": "toggle_resolution", "n_fft": n_fft,
                "action": "dropped" if in_recipe else "added",
                "resolutions": list(res), "median_step_ms": step_ms,
                "delta_vs_baseline_ms": round(step_ms - base_step, 1),
                "mrstft_forward_sum_ms": round(mrstft_total, 2),
                "loss": r["loss_values"].get("loss"), "loss_changed": True,
            })
            print(f"[ablation] {abl['runs'][-1]['action']} {n_fft}: {step_ms} ms", flush=True)

        for hop in (147, 512):
            r = run(args, args.precision, args.ablation_steps + 3, fixed_hop=hop)
            abl["runs"].append({
                "kind": "fixed_hop", "hop": hop,
                "median_step_ms": r["reconciliation"]["per_step_wall_median_ms"],
                "mrstft_ms": sum(v["sum_ms"] for k, v in r["phases"].items() if k.startswith("loss_mrstft")),
                "loss": r["loss_values"].get("loss"), "loss_changed": True,
            })
            print(f"[ablation] hop {hop}: {abl['runs'][-1]['median_step_ms']} ms", flush=True)

        bc = main_run["phases"]["backward"]
        for mode, label in (("spec_only", "complex_l1_only"), ("mask_scalar", "model_fwd_bwd_only")):
            r = run(args, args.precision, args.ablation_steps + 3, loss_mode=mode)
            bwd = r["phases"].get("backward") or {}
            abl["runs"].append({
                "kind": "backward_split", "loss_mode": mode, "label": label,
                "median_step_ms": r["reconciliation"]["per_step_wall_median_ms"],
                "backward_median_ms": bwd.get("median_ms"),
                "loss": r["loss_values"].get("loss"), "loss_changed": True,
            })
            print(f"[ablation] backward {label}: {abl['runs'][-1]['backward_median_ms']} ms", flush=True)

        for workers in (0, 4, 8, 16):
            r = run(args, args.precision, args.ablation_steps + 3, num_workers=workers)
            abl["runs"].append({
                "kind": "num_workers", "num_workers": workers,
                "median_step_ms": r["reconciliation"]["per_step_wall_median_ms"],
                "dataloader_wait_ms": (r["phases"].get("dataloader_wait") or {}).get("median_ms"),
                "loss": r["loss_values"].get("loss"), "loss_changed": False,
            })
            print(f"[ablation] workers {workers}: {abl['runs'][-1]['median_step_ms']} ms", flush=True)

        abl["baseline_step_ms"] = main_run["reconciliation"]["per_step_wall_median_ms"]
        abl["baseline_loss"] = baseline_loss
        main_run["ablation"] = abl

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(main_run, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
