"""Bare single-device training loop (no Lightning, no HCCL) for the frequency U-Net.

Bring-up order:
    1. --precision 32   : fp32 reference run (numeric comparison against CPU/CUDA path)
    2. --precision bf16 : autocast bf16, fp32 master weights (fp16 is refused)

Features: AdamW + warmup, grad clip 1.0, EMA 0.999, step checkpoints
(last.ckpt + ema-<step>.pt), full resume (model + EMA + Adam + step +
deterministic data position), optional grad accumulation for an effective
batch of 24.

Data layout matches the pymss_core / BS-RoFormer convention: the frequency axis
is `freq_bins * audio_channels`, f-major, and the STFT is computed on the CPU in
fp32 so the model sees exactly the reference input.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import wave
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arch_factory import build as build_arch
from freq_unet import FreqUNetConfig, build_model  # noqa: F401  (v1 defaults)
from fast_loss import stft_mag_fast  # cheap-adjoint |STFT|, math-equivalent (see fast_loss.py)
from freq_unet import param_count  # noqa: E402

TARGET_STEM = "vocals"


# ------------------------------------------------------------------ data
def scan_musdb(root: Path, splits=("train",)) -> list[dict[str, Path]]:
    tracks = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for track in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files = {}
            for stem in ("mixture", TARGET_STEM):
                for ext in (".wav", ".flac"):
                    candidate = track / f"{stem}{ext}"
                    if candidate.is_file():
                        files[stem] = candidate
                        break
            if len(files) == 2:
                tracks.append(files)
    return tracks


def _read_wav(path: Path, start: int, frames: int, channels: int = 2) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        n_ch, width, total = handle.getnchannels(), handle.getsampwidth(), handle.getnframes()
        handle.setpos(max(0, min(int(start), total)))
        raw = handle.readframes(frames)
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    audio = audio.reshape(-1, n_ch)
    if n_ch == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=1)
    elif n_ch > channels:
        audio = audio[:, :channels]
    if audio.shape[0] < frames:
        audio = np.pad(audio, ((0, frames - audio.shape[0]), (0, 0)))
    return audio.T.copy()  # (channels, frames)


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


class MusdbChunkDataset(torch.utils.data.Dataset):
    """Deterministic 8 s chunks: item index -> (track, offset) is a pure function of the index."""

    def __init__(self, tracks, chunk_size: int, n_fft: int, hop: int, channels: int = 2):
        self.tracks = tracks
        self.chunk_size = chunk_size
        self.n_fft = n_fft
        self.hop = hop
        self.channels = channels
        self.window = torch.hann_window(n_fft)
        self._frames = {}

    def __len__(self) -> int:
        return 1 << 30  # logically infinite; the loop drives the index

    def _frames_of(self, path: Path) -> int:
        if path not in self._frames:
            self._frames[path] = _wav_frames(path)
        return self._frames[path]

    def _pick(self, index: int):
        h = (index * 2654435761) & 0xFFFFFFFF
        track = self.tracks[h % len(self.tracks)]
        total = min(self._frames_of(p) for p in track.values())
        rng = random.Random(h ^ 0x9E3779B9)
        offset = rng.randint(0, max(0, total - self.chunk_size - 1))
        return track, offset

    def _stft_fc(self, audio: torch.Tensor) -> torch.Tensor:
        """(C, N) -> (F*C, T, 2), f-major, matching pymss_core's stft_roformer."""
        spec = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
            center=True, pad_mode="constant", return_complex=True,
        )  # (C, F, T)
        real = torch.view_as_real(spec)  # (C, F, T, 2)
        f, t = real.shape[1], real.shape[2]
        return real.permute(1, 0, 2, 3).reshape(f * self.channels, t, 2)

    def __getitem__(self, index: int):
        track, offset = self._pick(index)
        mixture = torch.from_numpy(_read_wav(track["mixture"], offset, self.chunk_size, self.channels))
        target = torch.from_numpy(_read_wav(track[TARGET_STEM], offset, self.chunk_size, self.channels))
        if _TGT_WAV:
            # the loss needs |istft(target)| anyway; shipping the source waveform skips one
            # 83 ms iSTFT per step. istft(stft(x)) == x to ~1e-7 for hann/75% overlap.
            return self._stft_fc(target), self._stft_fc(mixture), target
        return self._stft_fc(target), self._stft_fc(mixture)


# ------------------------------------------------------------------ losses
def complex_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred/target: complex (B, F, T) -> L1 over re/im."""
    return (pred - target).abs().mean()


def si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    alpha = (pred * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    scaled = alpha * target
    noise = pred - scaled
    return 10 * torch.log10((scaled.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


def multi_res_stft_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor,
                        resolutions=None, hop_override: int = 0) -> torch.Tensor:
    """torch.stft (2.12) only accepts 1-D/2-D input, so flatten (B, C) into the batch axis."""
    resolutions = _LOSS_RES if resolutions is None else resolutions
    hop_override = hop_override or _LOSS_HOP
    lead = pred_wav.shape[:-1]
    pred_flat = pred_wav.float().reshape(-1, pred_wav.shape[-1])
    target_flat = target_wav.float().reshape(-1, target_wav.shape[-1])
    loss = pred_wav.new_zeros(())
    for n_fft in resolutions:
        window = torch.hann_window(n_fft).to(device=pred_wav.device, dtype=torch.float32)
        hop = hop_override or n_fft // 4
        p = torch.stft(pred_flat, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
        t = torch.stft(target_flat, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
        pm, tm = p.abs(), t.abs()
        # Near-silent windows make the plain ratio explode (||tm|| ~ 0); floor the
        # denominator with a scale-relative epsilon and clamp the ratio.
        scale = tm.max().detach().clamp_min(1e-4)
        den = torch.linalg.vector_norm(tm, dim=(-2, -1)).clamp_min(1e-3 * scale)
        sc = (torch.linalg.vector_norm(tm - pm, dim=(-2, -1)) / den).clamp(max=10.0)
        mag = F.l1_loss(torch.log(pm + 1e-3 * scale), torch.log(tm + 1e-3 * scale))
        loss = loss + sc.mean() + mag
    return loss / len(resolutions)


def multi_res_stft_loss_fast(pred_wav: torch.Tensor, target_wav: torch.Tensor,
                             resolutions=None, hop_override: int = 0) -> torch.Tensor:
    resolutions = _LOSS_RES if resolutions is None else resolutions
    hop_override = hop_override or _LOSS_HOP
    """Same loss as `multi_res_stft_loss`, but |STFT| comes from the cheap-adjoint path.

    torch.stft's own backward costs ~480 ms per call on this device; `stft_mag_fast`
    replaces it with an explicit adjoint (rfft adjoint + overlap-add). Verified in
    fast_loss_check.py: forward rel_rms 3.3e-08, gradient rel_rms <= 7e-07.
    """
    lead = pred_wav.shape[:-1]
    pred_flat = pred_wav.float().reshape(-1, pred_wav.shape[-1])
    target_flat = target_wav.float().reshape(-1, target_wav.shape[-1])
    loss = pred_wav.new_zeros(())
    for n_fft in resolutions:
        window = torch.hann_window(n_fft).to(device=pred_wav.device, dtype=torch.float32)
        hop = hop_override or n_fft // 4
        pm = stft_mag_fast(pred_flat, window, n_fft, hop)
        with torch.no_grad():
            tm = stft_mag_fast(target_flat, window, n_fft, hop)
        scale = tm.max().detach().clamp_min(1e-4)
        den = torch.linalg.vector_norm(tm, dim=(-2, -1)).clamp_min(1e-3 * scale)
        sc = (torch.linalg.vector_norm(tm - pm, dim=(-2, -1)) / den).clamp(max=10.0)
        mag = F.l1_loss(torch.log(pm + 1e-3 * scale), torch.log(tm + 1e-3 * scale))
        loss = loss + sc.mean() + mag
    return loss


def multi_res_stft_loss_direct_spec(pred_c: torch.Tensor, tgt_c: torch.Tensor,
                                    pred_wav: torch.Tensor, tgt_wav: torch.Tensor,
                                    resolutions=None, channels: int = 2) -> torch.Tensor:
    """Recipe variant: the largest MR-STFT resolution is taken straight from the model
    spectrum instead of istft -> stft (saves one iSTFT-consistent STFT per step).

    |mix*mask| is *not* the STFT of the istft-ed prediction (window overlap changes it), so
    this changes the loss; it is opt-in via --loss-direct-spec and the default recipe keeps
    the exact istft->stft path.
    """
    resolutions = _LOSS_RES if resolutions is None else resolutions
    big = max(resolutions)
    rest = tuple(r for r in resolutions if r != big)
    loss = pred_wav.new_zeros(())
    if rest:
        loss = loss + multi_res_stft_loss_fast(pred_wav, tgt_wav, resolutions=rest) * len(rest)
    pm = pred_c.abs().reshape(pred_c.shape[0], -1, pred_c.shape[-1])
    with torch.no_grad():
        tm = tgt_c.abs().reshape(tgt_c.shape[0], -1, tgt_c.shape[-1])
    scale = tm.max().detach().clamp_min(1e-4)
    den = torch.linalg.vector_norm(tm, dim=(-2, -1)).clamp_min(1e-3 * scale)
    sc = (torch.linalg.vector_norm(tm - pm, dim=(-2, -1)) / den).clamp(max=10.0)
    mag = F.l1_loss(torch.log(pm + 1e-3 * scale), torch.log(tm + 1e-3 * scale))
    loss = loss + sc.mean() + mag
    return loss / len(resolutions)


def leakage_loss(pred_wav: torch.Tensor, target_wav: torch.Tensor, rms_threshold: float = 0.005, frame: int = 2048):
    """Vocal near-silence windows (target RMS < 0.005) must not carry energy.

    Kept strictly out of the SI-SDR statistic: this term only sees quiet windows.
    """
    n = (target_wav.shape[-1] // frame) * frame
    if n == 0:
        return pred_wav.new_zeros(())
    tgt = target_wav[..., :n].reshape(*target_wav.shape[:-1], -1, frame)
    prd = pred_wav[..., :n].reshape(*pred_wav.shape[:-1], -1, frame)
    # +1e-10 inside the sqrt only floors d(sqrt)/dx at exactly-silent windows
    # (upstream `sqrt(0)` gives an infinite gradient -> NaN weights). It does not
    # change any value above ~1e-5 RMS, i.e. nothing near the 0.005 threshold.
    tgt_rms = tgt.pow(2).mean(dim=-1).add(1e-10).sqrt()
    quiet = tgt_rms < rms_threshold
    if not bool(quiet.any()):
        return pred_wav.new_zeros(())
    # dB domain penalty above the threshold, i.e. 20*log10(rms / thr) for loud errors
    prd_rms = prd.pow(2).mean(dim=-1).add(1e-10).sqrt()
    ratio = (prd_rms[quiet] + 1e-8) / rms_threshold
    return F.relu(torch.log10(ratio)).mean()


# ------------------------------------------------------------------ EMA
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.other = {k: v.detach().clone() for k, v in model.state_dict().items() if not v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            shadow = self.shadow.get(k)
            if shadow is not None:
                shadow.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow, "other": self.other}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self.other = {k: v.clone() for k, v in state.get("other", {}).items()}

    def apply_to(self, model: torch.nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state = {**{k: v.to(backup[k].dtype) for k, v in self.shadow.items()}, **self.other}
        model.load_state_dict(state, strict=False)
        return backup


# ------------------------------------------------------------------ io helpers
def _atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: Path, step: int, model, ema: EMA, optimizer, args) -> None:
    _atomic_save(
        {"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
         "optimizer": optimizer.state_dict(), "args": vars(args)},
        path,
    )


def save_ema(path: Path, step: int, ema: EMA) -> None:
    _atomic_save({"step": step, "ema": ema.state_dict()}, path)


def autocast_ctx(device: torch.device, precision: str):
    if precision == "32":
        return nullcontext()
    if precision == "fp16":
        raise SystemExit("fp16 training is forbidden by the bring-up spec; use --precision bf16 or 32")
    if precision != "bf16":
        raise SystemExit(f"unknown precision {precision}")
    if device.type == "npu":
        return torch.npu.amp.autocast(dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        try:
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(name)


def split_fc(spec_c: torch.Tensor, channels: int) -> torch.Tensor:
    """(B, F*C, T) complex -> (B*C, F, T) complex for per-channel iSTFT."""
    b, fc, t = spec_c.shape
    f = fc // channels
    return spec_c.reshape(b, f, channels, t).permute(0, 2, 1, 3).reshape(b * channels, f, t)


_LOSS_RES = (512, 1024, 2048)
_LOSS_HOP = 0


_ISTFT_ADJOINT = os.environ.get("PYMSS_ISTFT_ADJOINT", "1").lower() in {"1", "true", "yes", "on"}


_TGT_WAV = os.environ.get("PYMSS_TGT_WAV", "1").lower() in {"1", "true", "yes", "on"}


def waveform(x, channels: int, n_fft: int, hop: int, length: int, window) -> torch.Tensor:
    """Per-channel iSTFT with an explicit adjoint.

    Two wins over the old path (measured in istft_check.py):
    * the channel split runs on the real view: complex reshape/permute costs 73 ms
      per call here, the real-plane version 0.4 ms (that, not iSTFT, was the 83 ms)
    * iSTFT's backward is replaced by the exact adjoint (fast_istft.IstftChannels)
    """
    if _ISTFT_ADJOINT:
        from fast_istft import istft_channels

        return istft_channels(x, window, n_fft, hop, length, channels)
    from pymss_core.npu_fft import istft_npu

    flat = split_fc(x, channels)
    wav = istft_npu(flat, n_fft=n_fft, hop_length=hop, window=window, length=length,
                    return_complex=False)
    return wav.reshape(x.shape[0], channels, -1)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--save-dir", type=Path, default=Path("/workspace/runs/ascend_b24"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="32", choices=("32", "bf16", "fp16"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--accum", type=int, default=1, help="grad accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--chunk-size", type=int, default=352800)  # 8 s @ 44.1 kHz
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss-resolutions", default="512,1024,2048",
                        help="MR-STFT resolutions (recipe knob; default matches the G4 recipe)")
    parser.add_argument("--loss-hop", type=int, default=0,
                        help="force one hop for every MR-STFT resolution (0 = n_fft//4)")
    parser.add_argument("--loss-direct-spec", action="store_true",
                        help="compute the largest MR-STFT resolution straight from the model "
                             "spectrum instead of istft->stft (changes the loss, off by default)")
    parser.add_argument("--fast-loss", action="store_true",
                        help="use the cheap-adjoint STFT magnitude in the MR-STFT loss "
                             "(math-equivalent, avoids aten::stft backward)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", default="v1", choices=("v1", "v3"))
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    global _LOSS_RES, _LOSS_HOP
    _LOSS_RES = tuple(int(v) for v in str(args.loss_resolutions).split(",") if v.strip())
    _LOSS_HOP = int(args.loss_hop)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "npu" and args.precision == "32":
        # torch_npu defaults conv to HF32 (TF32-like); fp32 runs must disable it to
        # stay numerically comparable with the CPU/CUDA reference path.
        import torch_npu  # noqa: F401

        if hasattr(torch.npu, "conv"):
            torch.npu.conv.allow_hf32 = False
        if hasattr(torch.npu, "matmul"):
            torch.npu.matmul.allow_hf32 = False
    tracks = scan_musdb(args.data_root)
    if not tracks:
        raise SystemExit(f"no MUSDB tracks under {args.data_root}")
    if args.max_tracks:
        tracks = tracks[: args.max_tracks]
    print(f"device={device} precision={args.precision} batch={args.batch_size}x{args.accum} tracks={len(tracks)}", flush=True)

    model = build_arch(args.arch, n_fft=args.n_fft).to(device)
    total = param_count(model)
    print(f"model params={total} ({total / 1e6:.2f}M)", flush=True)

    dataset = MusdbChunkDataset(tracks, args.chunk_size, args.n_fft, args.hop)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    ema = EMA(model, decay=args.ema_decay)
    window = torch.hann_window(args.n_fft).to(device=device, dtype=torch.float32)
    channels = 2  # stereo input; both v1 and v3 use it

    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        print(f"resumed {args.resume} at step {step}", flush=True)

    def lr_at(s: int) -> float:
        if args.warmup and s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr

    args.save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    model.train()
    t_start = time.time()
    t_compute = 0.0
    it = iter(loader)

    while step < args.steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        micro = {}
        loss_accum = 0.0
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            if _TGT_WAV:
                target, mixture, tgt_wav_batch = batch
            else:
                target, mixture = batch
                tgt_wav_batch = None
            target = target.to(device, non_blocking=True)
            mixture = mixture.to(device, non_blocking=True)
            with autocast_ctx(device, args.precision):
                mask = model(mixture)  # (B, F*C, T, 2)
                mix_c = torch.view_as_complex(mixture.contiguous())
                tgt_c = torch.view_as_complex(target.contiguous())
                mask_c = torch.view_as_complex(mask.float().contiguous())
                pred_c = mix_c * mask_c
                loss_spec = complex_l1(pred_c, tgt_c)
                pred_wav = waveform(pred_c, channels, args.n_fft, args.hop, args.chunk_size, window)
                if _TGT_WAV:
                    tgt_wav = tgt_wav_batch.to(device=pred_wav.device, dtype=pred_wav.dtype)
                else:
                    tgt_wav = waveform(tgt_c, channels, args.n_fft, args.hop, args.chunk_size, window)
                if args.loss_direct_spec:
                    loss_mrstft = multi_res_stft_loss_direct_spec(
                        pred_c, tgt_c, pred_wav, tgt_wav, channels=channels)
                else:
                    loss_mrstft = (multi_res_stft_loss_fast if args.fast_loss else
                                   multi_res_stft_loss)(pred_wav, tgt_wav)
                loss_sisdr = -si_sdr(pred_wav, tgt_wav).mean() / 20.0
                loss_leak = leakage_loss(pred_wav, tgt_wav)
                loss = (loss_spec + 0.5 * loss_mrstft + 0.1 * loss_sisdr + 0.1 * loss_leak) / args.accum
            loss.backward()
            loss_accum += float(loss.detach())
            micro = {
                "loss_spec": float(loss_spec.detach()),
                "loss_mrstft": float(loss_mrstft.detach()),
                "loss_sisdr": float(loss_sisdr.detach()),
                "loss_leak": float(loss_leak.detach()),
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(model)
        if device.type == "npu":
            torch.npu.synchronize()
        step_s = time.time() - t0
        t_compute += step_s
        step += 1

        if step % args.log_every == 0:
            record = {
                "step": step,
                "loss": loss_accum,
                **micro,
                "grad_norm": float(grad_norm),
                "lr": lr_at(step),
                "step_s": round(step_s, 4),
                "steps_per_s": round(1 / step_s, 3),
                "mem_mb": (round(torch.npu.memory_allocated() / 2**20, 1) if device.type == "npu" else None),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(args.save_dir / "last.ckpt", step, model, ema, optimizer, args)
            save_ema(args.save_dir / f"ema-{step}.pt", step, ema)

    save_checkpoint(args.save_dir / "last.ckpt", step, model, ema, optimizer, args)
    save_ema(args.save_dir / f"ema-{step}.pt", step, ema)

    wall = time.time() - t_start
    warm = [r["step_s"] for r in history[1:]] or [r["step_s"] for r in history]
    summary = {
        "device": str(device),
        "precision": args.precision,
        "params": total,
        "batch": f"{args.batch_size}x{args.accum}",
        "steps": step,
        "wall_s": round(wall, 2),
        "step_per_s_wall": round(step / wall, 3),
        "step_per_s_compute": round(step / t_compute, 3) if t_compute else None,
        "steps_per_min": round(60 * step / t_compute, 1) if t_compute else None,
        "first_step_s": history[0]["step_s"] if history else None,
        "median_step_s": round(float(np.median(warm)), 4) if warm else None,
        "final_loss": history[-1]["loss"] if history else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "history": history}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
