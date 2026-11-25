"""Long-run training driver: resumable, supervised, checkpointed, evaluated.

Everything here is one step above ``train.py``: it reuses its dataset, losses, EMA and
checkpoint helpers and adds what a multi-hour run needs.

  * last.ckpt every --ckpt-every steps (atomic), ema-<step>.pt every --ema-every steps,
    keeping the newest --keep-ema of them
  * full resume from last.ckpt (model + EMA + Adam + step); the dataset draws random
    chunks, so the resume re-seeds from the step (documented in the JSONL)
  * frozen 56-window evaluation protocol (50 test tracks x 4 relative positions), window
    list and its sha1 written once to results/eval_windows.json
  * per evaluation: median/mean SI-SDR, per-song bootstrap 95% CI, near-silent-window
    leakage, for *both* the raw weights and the EMA
  * best.pt by validation median SI-SDR (never overwritten by a worse value)
  * disk guard: frees the oldest non-best EMA/newest checkpoints below --min-free-gb

    python train_long.py --steps 10000 --save-dir /workspace/runs/overnight
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (  # noqa: E402
    N_WINDOWS,
    WINDOW_FRACTIONS,
    bootstrap_by_song,
    build_protocol,
    evaluate as evaluate_protocol,
)
from arch_factory import build as build_arch
from fused_adamw import make_optimizer
from freq_unet import FreqUNetConfig, build_model  # noqa: F401  # noqa: E402
from train import (  # noqa: E402
    EMA,
    MusdbChunkDataset,
    _atomic_save,
    complex_l1,
    leakage_loss,
    multi_res_stft_loss_fast,
    save_checkpoint,
    save_ema,
    scan_musdb,
    si_sdr,
    waveform,
)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def guard_disk(save_dir: Path, protected: set[str], min_free_gb: float, log) -> None:
    """Delete the oldest not-protected checkpoints until the volume has room again."""
    if free_gb(save_dir) >= min_free_gb:
        return
    cands = sorted(
        [p for p in save_dir.glob("ema-*.pt") if p.name not in protected],
        key=lambda p: p.stat().st_mtime,
    )
    for p in cands:
        if free_gb(save_dir) >= min_free_gb:
            break
        log(f"disk guard: rm {p.name} ({p.stat().st_size / 2**20:.0f} MB)")
        p.unlink(missing_ok=True)


def prune_ema(save_dir: Path, keep: int, protected: set[str]) -> None:
    files = sorted(save_dir.glob("ema-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    while len(files) > keep:
        p = files.pop(0)
        if p.name in protected:
            files.append(p)
            continue
        p.unlink(missing_ok=True)


def window_manifest(protocol: list[dict], chunk: int) -> dict:
    blob = json.dumps([[w["index"], w["song"], w["offset"]] for w in protocol])
    return {"n_windows": len(protocol), "chunk": chunk, "fractions": list(WINDOW_FRACTIONS),
            "sha1": hashlib.sha1(blob.encode()).hexdigest(), "windows": protocol}


def eval_both(model, ema, dataset, tracks_by_song, protocol, device, args, window) -> dict:
    raw = evaluate_protocol(model, dataset, tracks_by_song, protocol, device, args.n_fft,
                            args.hop, args.chunk_size, window, args.precision)
    backup = ema.apply_to(model)
    try:
        ema_res = evaluate_protocol(model, dataset, tracks_by_song, protocol, device, args.n_fft,
                                    args.hop, args.chunk_size, window, args.precision)
    finally:
        model.load_state_dict(backup)
    out = {}
    for name, res in (("raw", raw), ("ema", ema_res)):
        stats = bootstrap_by_song(res["windows"])
        leak = [r["leakage_rms"] for r in res["windows"] if not np.isnan(r["leakage_rms"])]
        stats["leakage_rms_median"] = float(np.median(leak)) if leak else None
        if "h2_si_sdr" in protocol[0]:
            d = np.array([r["si_sdr"] - p["h2_si_sdr"] for r, p in zip(res["windows"], protocol)])
            stats["vs_h2"] = {"median": float(np.median(d)), "mean": float(d.mean()),
                              "win": int((d > 0).sum()), "loss": int((d < 0).sum())}
        out[name] = stats
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--save-dir", type=Path, default=Path("/workspace/runs/overnight"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--precision", default="bf16", choices=("32", "bf16"))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--ema-every", type=int, default=1000)
    parser.add_argument("--keep-ema", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--v3-window", type=int, default=None, help="ablation: window 8 -> 16")
    parser.add_argument("--v3-bottleneck-depth", type=int, default=None, help="ablation: 2 -> 3")
    parser.add_argument("--v3-n-msc", type=int, default=None, help="ablation: MSC 1 -> 2")
    parser.add_argument("--optim", default="fused", choices=("eager", "foreach", "fused"),
                        help="AdamW flavour; fused is torch_npu.npu_apply_adam_w (measured slower "
                             "and not bit-compatible with torch.optim.AdamW - see fused_adamw.py)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--arch", default="v1", choices=("v1", "v3"),
                        help="v1 = depthwise band blocks (the overnight baseline), v3 = window attention")
    parser.add_argument("--eval-protocol", default="h2", choices=("h2", "own", "both"),
                        help="h2 = the archived H2 56 windows/45 songs (paired with H2), "
                             "own = our older 50-song 56-window set (not comparable)")
    parser.add_argument("--fast-loss", action="store_true", default=True)
    parser.add_argument("--no-fast-loss", dest="fast_loss", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr-schedule", default="const", choices=("const", "cosine"),
                        help="const = the overnight default; cosine = one-variable tuning knob")
    parser.add_argument("--lr-min", type=float, default=5e-5)
    parser.add_argument("--lr-schedule-start", type=int, default=0,
                        help="step at which the cosine decay starts (for resumed tuning runs)")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                        help="pin host batches so the 70 ms/step H2D copy overlaps compute")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.jsonl"

    def log(msg: str) -> None:
        print(msg, flush=True)

    train_tracks = scan_musdb(args.data_root, splits=("train",))
    test_tracks = scan_musdb(args.data_root, splits=("test",))
    dataset = MusdbChunkDataset(train_tracks, args.chunk_size, args.n_fft, args.hop)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers else None,
    )

    model = build_arch(args.arch, n_fft=args.n_fft, window=args.v3_window,
                       n_bottleneck=args.v3_bottleneck_depth, n_msc=args.v3_n_msc).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = make_optimizer(model, args.optim, args.lr, args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)
    window = torch.hann_window(args.n_fft).to(device=device, dtype=torch.float32)
    channels = 2  # audio_channels: v1 and v3 both use stereo input

    protocol_map = {}
    if args.eval_protocol in ("h2", "both"):
        # H2's archived 56 windows over 45 songs: the only set we can compare against H2
        # checkpoint-by-checkpoint. Our own 50-song set is kept behind --eval-protocol own.
        from eval_h2_protocol import h2_protocol

        h2_prot, h2_tracks, h2_manifest = h2_protocol(args.chunk_size, test_tracks)
        h2_manifest["protocol"] = "h2_56"
        protocol_map["h2"] = (h2_prot, h2_tracks)
    if args.eval_protocol in ("own", "both"):
        own_tracks = {t["mixture"].parent.name: t for t in test_tracks}
        own_prot = build_protocol(test_tracks, args.chunk_size, N_WINDOWS)
        own_manifest = window_manifest(own_prot, args.chunk_size)
        own_manifest["protocol"] = "own_50song_56win (NOT paired with H2)"
        protocol_map["own"] = (own_prot, own_tracks)
    # primary protocol drives best.pt (own keeps continuity with the v1 runs)
    primary = "own" if "own" in protocol_map else "h2"
    protocol, tracks_by_song = protocol_map[primary]
    manifest = own_manifest if primary == "own" else h2_manifest
    # one manifest per protocol: the legacy own-protocol file keeps its name so the frozen
    # sha1 check for existing runs still works
    name = "eval_windows.json" if primary == "own" else f"eval_windows_{primary}.json"
    manifest_path = Path(__file__).resolve().parent / "results" / name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("sha1") != manifest["sha1"]:
            raise SystemExit(f"window protocol drift: {old.get('sha1')} != {manifest['sha1']}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"protocol sha1 {manifest['sha1']} ({manifest['n_windows']} windows)")

    step = 0
    best = {"step": -1, "val_median": -1e9, "which": None}
    best_path = save_dir / "best.pt"
    if best_path.exists():
        prev = torch.load(best_path, map_location="cpu", weights_only=False)
        best = prev.get("meta", best)
    resume_used = None
    if args.resume != "none":
        ckpt = save_dir / "last.ckpt" if args.resume in ("auto", "") else Path(args.resume)
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            ema.load_state_dict(payload["ema"])
            optimizer.load_state_dict(payload["optimizer"])
            step = int(payload["step"])
            resume_used = str(ckpt)
            log(f"resumed {ckpt} at step {step}")

    torch.manual_seed(args.seed + step)
    rng = np.random.default_rng(args.seed + step)
    loader_iter = iter(loader)

    def lr_at(s: int) -> float:
        if args.warmup and s < args.warmup:
            return args.lr * (s + 1) / min(args.warmup, args.steps)
        if args.lr_schedule == "cosine":
            base = max(args.warmup, args.lr_schedule_start)
            span = max(1, args.steps - base)
            t = min(1.0, max(0.0, (s - base) / span))
            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))
        return args.lr

    config = {
        "params": params, "batch": args.batch_size, "accum": args.accum,
        "precision": args.precision, "lr": args.lr, "warmup": args.warmup,
        "grad_clip": args.grad_clip, "ema_decay": args.ema_decay,
        "lr_schedule": args.lr_schedule, "lr_min": args.lr_min,
        "fast_loss": args.fast_loss, "istft_adjoint": os.environ.get("PYMSS_ISTFT_ADJOINT", "1"),
        "fast_block": os.environ.get("PYMSS_FAST_BLOCK", "1"),
        "triton_silu": os.environ.get("PYMSS_TRITON_SILU", "1"),
        "loss_weights": {"spec": 1.0, "mrstft": 0.5, "sisdr": 0.1, "leakage": 0.1},
        "protocol_sha1": manifest["sha1"], "steps": args.steps,
        "resume": resume_used,
    }
    log(json.dumps({"start": config}))

    model.train()
    t_start = time.time()
    step_times: list[float] = []
    while step < args.steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        micro = {}
        loss_accum = 0.0
        for _ in range(args.accum):
            try:
                target, mixture = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                target, mixture = next(loader_iter)
            target = target.to(device, non_blocking=args.pin_memory)
            mixture = mixture.to(device, non_blocking=args.pin_memory)
            ctx = (torch.npu.amp.autocast(dtype=torch.bfloat16) if args.precision == "bf16"
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                mask = model(mixture)
                mix_c = torch.view_as_complex(mixture.contiguous())
                tgt_c = torch.view_as_complex(target.contiguous())
                mask_c = torch.view_as_complex(mask.float().contiguous())
                pred_c = mix_c * mask_c
                loss_spec = complex_l1(pred_c, tgt_c)
                pred_wav = waveform(pred_c, channels, args.n_fft, args.hop, args.chunk_size, window)
                tgt_wav = waveform(tgt_c, channels, args.n_fft, args.hop, args.chunk_size, window)
                loss_mrstft = (multi_res_stft_loss_fast if args.fast_loss else
                               multi_res_stft_loss_fast)(pred_wav, tgt_wav)
                loss_sisdr = -si_sdr(pred_wav, tgt_wav).mean() / 20.0
                loss_leak = leakage_loss(pred_wav, tgt_wav)
                loss = (loss_spec + 0.5 * loss_mrstft + 0.1 * loss_sisdr + 0.1 * loss_leak) / args.accum
            loss.backward()
            loss_accum += float(loss.detach())
            micro = {"loss_spec": float(loss_spec.detach()), "loss_mrstft": float(loss_mrstft.detach()),
                     "loss_sisdr": float(loss_sisdr.detach()), "loss_leak": float(loss_leak.detach())}
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(model)
        torch.npu.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        step += 1

        if step % args.log_every == 0:
            recent = step_times[-args.log_every:]
            rec = {"step": step, "loss": loss_accum, **micro, "grad_norm": float(grad_norm),
                   "lr": lr_at(step), "step_ms": round(1000 * float(np.median(recent)), 1),
                   "mem_mb": round(torch.npu.memory_allocated() / 2**20, 1)}
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(json.dumps(rec))

        protected = {best.get("file")} if best.get("file") else set()
        if step % args.ckpt_every == 0:
            guard_disk(save_dir, protected, args.min_free_gb, log)
            save_checkpoint(save_dir / "last.ckpt", step, model, ema, optimizer, args)
        if step % args.ema_every == 0:
            save_ema(save_dir / f"ema-{step}.pt", step, ema)
            prune_ema(save_dir, args.keep_ema, protected)

        if step % args.eval_every == 0:
            t_eval = time.time()
            order = [t for t in protocol_map if t != primary] + [primary]
            for tag in order:
                prot, tbs = protocol_map[tag]
                sc = eval_both(model, ema, dataset, tbs, prot, device, args, window)
                sc["step"] = step
                sc["eval_s"] = round(time.time() - t_eval, 1)
                fname = f"eval-{step}.json" if tag == primary else f"eval_{tag}-{step}.json"
                with (save_dir / fname).open("w") as fh:
                    fh.write(json.dumps(sc, indent=2))
                if tag != primary:
                    with log_path.open("a") as fh:
                        fh.write(json.dumps({"eval_" + tag: {k: v for k, v in sc.items() if k != "windows"}}) + "\n")
            scores = sc          # last evaluated protocol; primary is evaluated last below
            scores["eval_s"] = round(time.time() - t_eval, 1)
            for which in ("raw", "ema"):
                val = scores[which]["median_si_sdr"]
                if val > best["val_median"]:
                    best = {"step": step, "val_median": val, "which": which,
                            "file": f"ema-{step}.pt" if which == "ema" else "best-raw.pt"}
                    state = ema.state_dict() if which == "ema" else {
                        "decay": ema.decay,
                        "shadow": {k: v.float() for k, v in model.state_dict().items() if v.is_floating_point()},
                        "other": {k: v for k, v in model.state_dict().items() if not v.is_floating_point()},
                    }
                    _atomic_save({"step": step, "ema": state, "meta": best}, best_path)
            scores["best"] = best
            with (save_dir / f"eval-{step}.json").open("w") as fh:
                fh.write(json.dumps(scores, indent=2))
            with log_path.open("a") as fh:
                fh.write(json.dumps({"eval": scores}) + "\n")
            log(json.dumps({"eval": {k: v for k, v in scores.items() if k != "windows"}}))

    save_checkpoint(save_dir / "last.ckpt", step, model, ema, optimizer, args)
    save_ema(save_dir / f"ema-{step}.pt", step, ema)
    prune_ema(save_dir, args.keep_ema, {best.get("file")} if best.get("file") else set())
    summary = {"steps": step, "wall_s": round(time.time() - t_start, 1),
               "median_step_ms": round(1000 * float(np.median(step_times)), 1),
               "best": best, "config": config}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({"done": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
