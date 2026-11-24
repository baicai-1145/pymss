# train_ascend — target-only frequency U-Net on Ascend 950PR

Bring-up code for the vocal frequency U-Net: model, bare training loop, numeric
alignment, Triton kernel slots, and the frozen evaluation protocol.

Full results and every environment quirk found on this box:
[`.tmp/ascend_setup_report.md`](../.tmp/ascend_setup_report.md).

## Environment (verified)

```bash
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu
python -m pip install --no-deps torch-npu==2.12.0.post3
```

`+cpu` is **mandatory**: the PyPI default (CUDA) wheel registers
`pinned_reserve_segment_size_mb` in `libtorch`, and `torch_npu` registers the same
key while `dlopen`-ing → `c10::ValueError` in C++ and an abort that Python cannot
catch. Also `torch.npu.conv.allow_hf32` defaults to `True`; fp32 runs must set it
to `False` or conv layers sit at ~4e-4 relative error instead of ~1e-6.

## Quick start

```bash
# self-check: 8 s forward+backward on CPU and NPU, grads finite
python train_ascend/selfcheck.py

# layer-by-layer alignment (fp32 and bf16)
python train_ascend/align_check.py --frames 64 --bf16

# 100-step baseline, batch 24
python train_ascend/train.py --data-root /workspace/musdb18hq \
    --steps 100 --batch-size 24 --precision bf16 \
    --save-dir /workspace/runs/baseline_bf16_b24 --json-out /tmp/last.json

# resume (model + EMA + Adam + step)
python train_ascend/train.py --resume /workspace/runs/baseline_bf16_b24/last.ckpt ...

# frozen 56-window protocol
python train_ascend/evaluate.py --ckpt /workspace/runs/baseline_bf16_b24/last.ckpt --weights raw

# 8 s inference latency vs the G4 reference
python train_ascend/bench_infer.py
```

## Model

`freq_unet.py` — `BandUNetSeparator`, 20,874,036 params (cap 21.63 M, reference
20,961,844):

```
BandSplit(62 bands, dim 192)
  enc62 LocalBandBlock -> BandDown 62->32 (pad 1)
  enc32 -> BandDown 32->16
  enc16 -> BandDown 16->8 (288 -> 128)
  4 x [ MSC dilated temporal conv (d=1,2,4,8) + 8-band full attention (RoPE, 2 heads x 64) ]
  BandUp 8->16 + gated skip(enc16) -> dec16
  BandUp 16->32 + gated skip(enc32) -> dec32
  BandUp 32->62 + gated skip(enc62) -> dec62
  final_norm -> MaskEstimator -> mask_to_complex_shape
```

Time never gets subsampled (690 frames for 8 s at hop 512). All math blocks
(`RMSNorm`, `Attention`, `BandSplit`, `MaskEstimator`, `mask_to_complex_shape`)
are imported from `pymss_core`, so the numeric path is the same one the CUDA
reference runs.

## Training

Bare loop, single device, no Lightning/HCCL: AdamW + warmup, grad clip 1.0,
EMA 0.999, `last.ckpt` + `ema-<step>.pt` every N steps, full resume, deterministic
item→(track, offset) mapping so a resume only needs the step. Loss = complex
spectral L1 + multi-resolution STFT + SI-SDR term + leakage on near-silent vocal
windows (kept out of the SI-SDR statistic). `--precision fp16` is refused.

Measured on this box (MUSDB18-HQ, batch 24): 3.35 s/step fp32, 2.74 s/step bf16
(7.2 / 8.8 samples/s vs the G4 reference's 9.3 samples/s).

## Triton

`npu_kernels_bandunet.py` holds the three fused kernels (band-split, RoPE, gated
attention out) behind `triton_ascend_available()`. **They cannot run here**:
triton-ascend publishes cp39/cp310/cp311 wheels only and this interpreter is 3.12.
Eager fallbacks are used instead and verified bit-identical.
