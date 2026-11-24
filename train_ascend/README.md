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

---

## Round 2: triton-ascend enabled, torch_npu 2.13, full re-measure

The earlier "no cp312 wheel" conclusion was **wrong** — it came from the generic
PyPI index. The Ascend-only index has cp312:

```bash
python -m pip install "triton-ascend==3.6.0.post2" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Version matrix is now: torch **2.13.0+cpu** · torch_npu **2.13.0** · triton **3.6.0** ·
triton-ascend **3.6.0.post2** · numpy 1.26.4 / scipy 1.13.1 (pinned by triton-ascend).
torch must stay `+cpu`; the CUDA wheel still aborts inside `torch_npu`'s `dlopen`.

`triton_check.py` proves compilation + execution on the NPU (`backend='npu'`,
vector-add bit-exact, first call ~0.8 s JIT).

Fused kernel results (fp16): band-split normalize chain **1.43–1.51x**, RoPE **1.84–2.05x**,
gated attention output **0.66x (slower — not used)**; the 1-D variant of the gate kernel
overflowed the UB (needs 2621440 bits vs 1769472 available) and had to become a 2-D tile.

Bigger win than any kernel: `pymss_core`'s CUDA-oriented `grouped_forward` is a
**negative** optimization on Ascend — per-band Linear beats the grouped `bmm` by 1.46x
(BandSplit) and 1.47x (MaskEstimator). Disabling it takes 8 s inference from 24.70 → 18.99 ms
(fp32) and 13.94 → 13.51 ms (bf16 + fused RoPE), i.e. 1.16x of the G4 reference.

Measured stage shares (bf16, 13.06 ms device time): band_split 23.7%, MaskEstimator 20.7%,
temporal blocks ~15%, everything else <7% each. The real bottleneck is the 62-band small-op
chain, not the mask head alone.

New tools: `profile_infer.py` (NPU-event stage breakdown), `bench_infer_fused.py`
(baseline vs per-band vs per-band+RoPE, 3 repeats).

---

## Where does the training step go? (`profile_train.py`)

Sync-bounded phase profiling of one bs24 step on real MUSDB data (100 steps,
3 warmup dropped). Results: `results/profile_train_{bf16,fp32}.json`.

| phase | bf16 ms | share |
|---|---|---|
| **backward (incl. loss graph)** | 2194.7 | 80.3% |
| forward | 202.9 | 7.4% |
| istft_tgt / istft_pred | 82.8 / 82.8 | 6.0% |
| h2d_copy (543 MB) | 77.6 | 2.8% |
| loss_mrstft 2048/1024/512 | 28.1 / 16.0 / 9.7 | 2.0% |
| optimizer_step / grad_clip / ema | 14.8 / 7.6 / 7.8 | 1.1% |
| dataloader_wait | 0.9 | 0.03% |
| total | 2732.2 | 100% |

Segments reconcile with wall time to 0.00%, so nothing is unaccounted for.

Splitting `backward` with loss-less ablations: the multi-resolution STFT loss costs
**1392 ms (51%)**, istft+SI-SDR+leakage 469 ms (17%), the model forward+backward only
763 ms (28%). The root cause is that Ascend has no native complex STFT —
`torch.stft` backward measures ~480 ms per call (see `results/stft_fwd_bwd_microbench.json`),
and the loss calls it three times per step.

Cheapest implementation-only fix (measured, 23 steps each): swap the loss's `torch.stft`
for `pymss_core.npu_fft.stft_npu` → 2098 ms, or take the magnitude from the real view
`sqrt(re^2+im^2+1e-12)` → 2084 ms. Both are algebraically the same loss (rel_rms ≤ 8.3e-7
for `stft_npu`), so neither changes the recipe. The real-view variant needs the epsilon:
plain `sqrt(0)` gave infinite gradients and NaN weights, the same trap as `leakage_loss`.
Loss-in-autocast gains nothing (−9 ms), resolutions/hop changes do help but they change
the loss definition and therefore break equivalence with the G4 reference.

---

## Peak calibration of this 950PR (`calibrate_peak.py`)

Everything downstream should be quoted against these measured denominators
(`results/peak_calib.json`):

| quantity | measured | nominal | ratio |
|---|---|---|---|
| bf16 GEMM peak | **377.5 TFLOPS** @ 16384³ | 425 (28-core tier) | 88.8% |
| fp32 GEMM peak | **23.64 TFLOPS** | TF32 212 | 11.2% of TF32 |
| fp16 GEMM peak | 370.6 TFLOPS | 425 | 87.2% |
| d2d bandwidth | **1439 GB/s** | 1.6 TB/s | 90.0% |
| single kernel launch | **4.47 µs** | — | — |

The GEMM curve matters: 512³ reaches only 14.5 TFLOPS (3.8% of peak), 2048³ 253,
4096³ 354 — you need M=K=N≥4096 before the cube cores are fed. fp32 has no tensor
path at all, which is the real reason fp32 training is 2.4x slower, not scheduling.

Elementwise on the real activation shape (690,62,192): SiLU/sigmoid sit at 90% of the
measured bandwidth roof, GELU(tanh) and RMSNorm only at 43% (ALU bound, ~2.3x headroom).

The 62-band Linear group is the clearest example of the launch problem: 62 individual
GEMMs cost 4.91 µs each of which **91% is pure launch** (4.47 µs); packing them into one
`bmm` is 10.0x faster (0.030 vs 0.298 ms), at the price of 2.0x padding waste.

Current model: 187.5 GFLOP / 13.79 ms bf16 = **13.6 TFLOPS = 3.6% of measured peak**, with
5670 aten calls and **1910 device kernels** per forward. 1910 x 4.47 µs = 8.54 ms, i.e. 62%
of the forward is launch floor. Compute would allow 0.497 ms (28x), the kernel count only
allows 1.62x — so the model is launch/granularity bound, not compute or bandwidth bound.
Getting 10x needs 1910 -> 308 kernels; the FLOPs budget (520 GFLOP) already fits the
current 187.5 GFLOP, so the model does **not** need to shrink — it needs fusion.
