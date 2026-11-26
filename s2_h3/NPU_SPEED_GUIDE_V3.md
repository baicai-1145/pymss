# H4s Ascend 950PR Speed Sprint V3 — Verified Roofline, X-Ray & 10 steps/s Playbook

All numbers and Triton kernels below were **measured live on this exact 950PR container**
(`Ascend950PR_9579`, 28 AI / 56 Vector cores, `torch_npu 2.13.0`, `triton-ascend 3.6.0`,
`H4s`, batch 24 x 131584). Numbers are fwd+bwd, so they are directly additive to ms/step.

---

## 0. Hardware Roofline & Per-Module X-Ray (at 278 ms/step)

### Hardware ceiling (measured live)
- Peak BF16 GEMM (`4096^3`): **367.8 TFLOP/s** (0.37 ms)
- `proj_in` GEMM (`383904x192 @ 192x768`): **192.1 TFLOP/s** (0.59 ms fwd; 3.79 ms fwd+bwd)
- Peak HBM bandwidth (256 MB copy): **1,343 GB/s** (0.37 ms)

So at 278 ms we are nowhere near compute-bound: the step is dominated by
**kernel-launch count, layout churn (permute/cast/repeat_interleave), and the ~94 ms host dispatch gap**.

### Per-module fwd+bwd breakdown (measured live, SUM = 233.84 ms)
| Module | fwd+bwd (ms) | Verified speedup / notes |
|---|---:|---|
| **`time[1]` (L1 MSC, full 62b, s=1)** | **70.32** | **`npu_swiglu`: 67.77 -> 34.39 ms (−33.4 ms on L1 alone!)**; NTC Triton dW: 18.17 -> 7.62 ms (−10.6 ms) |
| **`freq[3..5].attn` (`BlockSlowBandAttn`)** | **11.72 x 3 = 35.17** | sub-ops only sum to 6.64 ms → **~5 ms x 3 = 15 ms hidden overhead**; `F.sdpa` (V pad 16->64): 2.53 -> 1.42 ms |
| **`freq[1..5].ffn` (`BlockFeedForward`)** | **5.89 x 5 = 29.43** | `block_mean` 1.00 + `repeat_interleave` 0.96 + net ~3.9 |
| `time[2]` (L2 MSC, 32b s=2 k=15) | 14.79 | gets `npu_swiglu` + NTC Triton dW |
| `mask_head` (62 bands) | 13.88 | eager 14.02; batched 7-bmm slower (21.37); **NPUGraph 12.41** |
| `band_split` (62 bands) | 11.88 | **pre-batched 6-bmm: 7.07**; **NPUGraph 6.40** |
| `freq[0].ffn` (L0 FFN, full 62b s=1) | 11.10 | GELU + 2 GEMMs + dropout |
| `time[3]`, `time[5]` | 9.67 + 9.66 = 19.33 | gets `npu_swiglu` + NTC Triton dW |
| `time[4]` (`StridedPooledKV`) | 9.10 | q_stride=2, kv_stride=4, 62b |
| `freq[0..2].attn` (`StaticBandMixer`) | 4.94 x 3 = 14.82 | already 8 large `torch.bmm` |
| `time[0]` (`StridedPooledKV`) | 4.04 | **only module that threw `EE1017` under NPUGraph!** |

Remaining ~44 ms of the 278 ms: `opt+ema` (~19 ms) + `stft+istft+loss` (~8 ms)
+ residual adds + host dispatch gaps.

---

## 1. Phase 1 — Land the 6 live-verified wins (278 -> ~160–175 ms/step)

### Win 1: `torch_npu.npu_swiglu` in MSC (−45 to −50 ms/step)
- Live verified: L1 MSC full fwd+bwd **67.77 -> 34.39 ms (−33.38 ms)**.
- `npu_swiglu(uv, dim=-1)` computes `silu(first_half) * second_half`, but MSC computes
  `u * silu(v)` = `first_half * silu(second_half)`.
- **Exact 0-rel-err re-parameterization**: swap the two output halves `[0:hid]` and `[hid:2*hid]`
  of `proj_in.weight` / `proj_in.bias`, and un-swap inside `state_dict` / `load_state_dict` hooks
  so `resume-5000.ckpt` and `ema-XXXX.pt` stay 100% 1:1 compatible with CPU / H4s.
- Applies to all 4 MSC instances: `convs[1]` (MSC) and `convs[2,3,5].msc` (StridedLatentMSC).

### Win 2: channels-last `(N,T,C)` 54-core Triton-Ascend DWConv1d dW (−15 to −20 ms/step)
- Live verified on L1 `(N=1488, T=258, C=384, K=31)`:
  - baseline `dwconv_dw_bmm` (chunk=8, **186 launches**): **18.17 ms**
  - NTC Triton `_dw_ntc_tiled` (`BT=128, BC=64, NS=9`, grid=(54,)): **7.62 ms**, rel_err 2.3e-3
  - also `BT=64, BC=128, NS=18`: 7.72 ms
- Why NTC wins: `C=384` (768 B) is 100% 32B aligned, `ext.extract_slice` slices dim 0
  (contiguous row stride, no unaligned vector shift), `BT=128 x BC=64` fp32 tile fits 256 KB UB.
- Working kernel (copy & retune `BT/BC/NS` per layer):

```python
import triton, triton.language as tl
import triton.language.extra.cann.extension as ext

@triton.jit
def _dw_ntc_tiled(X_ptr, DY_ptr, OUT_ptr, N,
                  C: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    n_cb = C // BC
    cb = pid % n_cb
    sp = pid // n_cb
    n_per = (N + NS - 1) // NS
    n0 = sp * n_per
    n1 = tl.minimum(n0 + n_per, N)
    c_off = cb * BC + tl.arange(0, BC)
    pad: tl.constexpr = K // 2
    BT_PAD: tl.constexpr = BT + K - 1
    ot_pad = tl.arange(0, BT_PAD)
    ot = tl.arange(0, BT)
    n_tb: tl.constexpr = (T + BT - 1) // BT
    acc = tl.zeros((K, BC), dtype=tl.float32)
    for n in range(n0, n1):
        base = n * T * C
        for tb in range(n_tb):
            t0 = tb * BT
            tx = t0 + ot_pad - pad
            mx = (tx >= 0) & (tx < T)
            ty = t0 + ot
            my = ty < T
            x_tile = tl.load(X_ptr + base + tl.where(mx, tx, 0)[:, None] * C + c_off[None, :],
                             mask=mx[:, None], other=0.0).to(tl.float32)
            y_tile = tl.load(DY_ptr + base + tl.where(my, ty, 0)[:, None] * C + c_off[None, :],
                             mask=my[:, None], other=0.0).to(tl.float32)
            for k in range(K):
                xk = ext.extract_slice(x_tile, offsets=(k, 0), sizes=(BT, BC), strides=(1, 1))
                s = tl.sum(xk * y_tile, axis=0)
                cur = ext.extract_slice(acc, offsets=(k, 0), sizes=(1, BC), strides=(1, 1))
                acc = ext.insert_slice(acc, cur + s[None, :], offsets=(k, 0), sizes=(1, BC), strides=(1, 1))
    ok = tl.arange(0, K)
    tl.store(OUT_ptr + sp * K * C + ok[:, None] * C + c_off[None, :], acc)
```

- Keep `F.conv1d` for fwd (3.56 ms) and dX (~8.2 ms); use Triton only for dW.
- Layout: build `x_ntc = g.permute(0, 2, 1, 3).reshape(b * f, t, hid)` directly in MSC so dW
  needs **zero** extra transpose; also fold `db = dy_ntc.sum((0, 1))` into the same kernel for free.
- Extras to try: `NPUOptions(multibuffer=True, enable_ubuf_saving=True, enable_preload=True)`.

### Win 3: `FixedNpuFusedAdamW` + single-kernel flat EMA (−16 ms/step)
- Live verified:
  - `AdamW(foreach=True)`: 8.42 ms
  - `NpuFusedAdamW` (subclass overriding `zero_grad(set_to_none=True)` — the base class raises
    `AttributeError`): **1.63 ms**
  - flat EMA over `opt.get_combined_params()` with one `lerp_(p, 1-0.999)`: **<0.5 ms**
    (vs ~10 ms for the 550-tensor `AveragedModel` loop)
- Plumbing: relax `s2_aw_entry.py:43` assert to accept `NpuFusedAdamW`; verify
  `load_state_dict` from `resume-5000.ckpt`, EMA export (`ema-XXXX.pt`) and 200-step drift ≤1%.

### Win 4: native pre-batched `BandSplit` (−5 ms/step)
- Live verified: eager 62-band loop 12.07 ms -> 6 batched param groups `(G, d_in, 192)` + `torch.bmm`
  = **7.07 ms** (no per-step `torch.stack`, unlike the rejected R1b).
- Add 1:1 `state_dict` / `load_state_dict` mapping onto `to_features.{i}.{0,1}`.
- **Do NOT batch `MaskEstimator`** — live verified slower (21.37 vs 14.02 ms eager).

### Win 5: `BlockSlowBandAttention` -> `F.scaled_dot_product_attention` (−3.3 ms/step)
- Live verified: inner attention (`1560x8x62x64`, V zero-padded 16->64 then sliced back):
  2.53 -> **1.42 ms**. Also try `torch_npu.npu_fusion_attention` (arbitrary V dim, fast path).

### Win 6: pure-BF16 params + FP32 master weights in optimizer (−17.5 ms/step)
- Live verified: MaskCore `bf16-mixed` 234.88 ms -> pure BF16 params **217.38 ms**
  (−17.5 ms, kills ~1,777 weight `Cast` ops/step).
- Keep FP32 master inside the optimizer; one flat `bf16_flat.copy_(fp32_flat)` costs 0.06 ms.
- Verify 200-step drift ≤1% vs `profile/h4s_ref200.json`.

### Runtime knobs (measure each, keep only wins)
`TASK_QUEUE_ENABLE=2`, `CPU_AFFINITY_CONF=1|2`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
`torch_npu.npu.set_compile_mode(jit_compile=False)`, `allow_internal_format` (check exact API).

---

## 2. Phase 2 — crack ~165 -> ≤100 ms/step (10 steps/s)

### A. Unlock full `_forward_mask_core` ACLGraph — kills the host dispatch gap
Two exact root causes were isolated by live per-module capture tests:

1. **Dropout blocker** — `nn.Dropout` calls `aclnnDropoutV3Tensor`
   (`... not in libopapi.so`), and `torch.rand` calls `philox_engine_inputs` (forbidden during capture).
   **Fix**: graph-safe stateless-seed Triton dropout using `tl.rand(seed, offsets)` where `seed` is
   read from a 1-element int64 NPU tensor; `seed_buf.add_(1)` runs **outside** the graph before each
   `replay()` → fresh unbiased mask every step, 100% capturable.
2. **`time[0]` `EE1017` blocker** — proven per-module: `band_split` (12.07 -> **6.40 ms = 1.89x**),
   `time[1]`, `freq[0].attn`, `freq[0].ffn`, `freq[3].attn`, `mask_head` all capture cleanly;
   **only `time[0]` (`StridedPooledKVAttention`) fails with EE1017**.
   Isolate the offending line (prime suspects: SDPA cross-stream, `index_select` on `cos/sin`);
   pre-slice `cos_q` / `cos_kv` in the cache and try manual `q@k` or `npu_fusion_attention`.
   Then capture the **whole `_forward_mask_core` fwd+bwd** into one `NPUGraph` replay.
3. Also test `torch.compile(backend="npugraph_ex")` (torchair ships in `torch_npu/dynamo/torchair`)
   with `@torch.library.custom_op` wrappers around the Triton kernels.

### B. Remaining compute hotspots (target device-active ≤ 90 ms)
1. **Missing ~15 ms in `BlockSlowBandAttention`** (3 x 11.72 ms vs 6.64 ms of sub-ops):
   hunt `repeat_interleave(4,1)[:, :t]`, `to_gates.sigmoid`, `permute/view/contiguous`, `q@k` transpose.
2. **Fused Triton `block_broadcast_add`** (11 call sites): fuse
   `x + out.repeat_interleave(stride, 1)[:, :t]` fwd and `block_sum` bwd so the 4x-expanded
   `(24, 258, 62, 192)` tensor never materialises in HBM. Also fuse `block_mean` (11 call sites,
   bf16, no `cat` padding).
3. **`freq[0..5].ffn` = 40.5 ms total**: profile RMSNorm+Linear+GELU+Dropout chain, try
   `npu_gelu`, fused gelu+dropout, and check `freq[0].ffn` (11.1 ms) for redundant copies/casts.
4. **MSC dX and `stft+istft+loss` (~8 ms)**: check an NTC Triton dX, and audit fp32/complex
   AI_CPU ops left in the FFT path.

---

## Admission gate (unchanged, strict)
Every admitted milestone must pass:
1. forward + backward parity vs clean baseline (deterministic eval mode),
2. **200-step real training curve vs `profile/h4s_ref200.json`: drift ≤ 1%**,
3. 1:1 checkpoint compatibility with `resume-5000.ckpt` and `ema-XXXX.pt` (loadable on CPU/H4s).

Keep a running milestone table (`ms/step`, `steps/s`, speedup, HBM, 200-step drift) in `TRITON.md`.
**Target ≤ 105 ms/step (~10 steps/s).** Do not stop and do not resume `stage.py 10000` until the
target is hit or every Phase 1 + Phase 2 item has been benchmarked.
