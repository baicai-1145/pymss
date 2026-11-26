# Ascend 950PR Training Acceleration Guide (Target: ~10 steps/s on H4s)

Environment verified on this host:
- Hardware: Ascend950PR_9579, 28 AICores (Cube), 56 AIVector Cores, 256 KB UB per vector core
- Software: CANN 9.2.1, torch 2.13.0+cpu, torch_npu 2.13.0, triton-ascend 3.6.0
- Current H4s baseline: ~371 ms/step (2.70 steps/s), 17.3 GB HBM, NPU util ~76%

## 1. Kill Per-Step Host-Device Sync in STFT/ISTFT (CRITICAL)
- `pymss_core/device.py` defaults `fft_result_check=True` for `ASCEND`.
- `pymss_core/npu_fft.py` lines 449 & 504 call `_degenerate()` -> `float(out_planes.abs().max())` on EVERY forward step (you can see the `Converting a tensor with requires_grad=True to a scalar` warning in `runs/H4s/train-5000.log`).
- That injects **2 synchronous D2H barriers per step**, stalls the NPU queue, and makes `torch.compile` / ACLGraph capture impossible.
- Fix: set `fft_result_check=False` for `ASCEND` in `device.py` (or cache checked `n_fft` in a `_VENDOR_FFT_OK` set so the probe runs at most once per size).

## 2. Native Fused `torch_npu` / CANN 9.2.1 Operators (Zero Kernel-Maintenance, Exact Autograd)
Available in `torch_npu 2.13.0` (332 native ops):
1. **RoPE** (was 61.6 ms/step in H3 due to 4 strided slice muls):
   - `torch_npu.npu_rotary_mul(x, cos_full, sin_full, rotary_mode='interleave')` (also check `npu_interleave_rope`).
   - Verify `cos_full = cos.repeat_interleave(2, -1)`, `sin_full = sin.repeat_interleave(2, -1)` layout (BSND/BNSD) and backward gradient vs `apply_rotary_emb_fast`. Since our cos/sin do NOT require grad, the `B*N<=1024` grad constraint does not apply.
2. **SwiGLU** in MSC:
   - `torch_npu.npu_swiglu(proj_in(x), dim=-1)` computes `silu(first_half) * second_half`.
   - Note: `MSC` currently computes `u * F.silu(v) = first_half * silu(second_half)`. When resuming an existing H4s checkpoint, either swap the two halves of `proj_in.weight`/`bias` once on load/init or use a fused kernel with matching order. Verify exact bit/rel_rms equivalence!
3. **RMSNorm & Residual+RMSNorm**:
   - `torch_npu.npu_rms_norm` (already using `F.rms_norm`, verify if `npu_rms_norm` or `npu_add_rms_norm` is faster on 950PR).
4. **Optimizer & EMA**:
   - `torch_npu.optim.NpuFusedAdamW` (fuses multi-tensor AdamW into 1-2 launches instead of ~500 per-param launches).
   - Replace Python per-param loop in `AveragedModel.update_parameters` with `torch._foreach_lerp_` / `_foreach_mul_` + `_foreach_add_` (or check its exact time in profile first).
5. **Attention**:
   - Check if `torch_npu.npu_fusion_attention` is faster than PyTorch SDPA for `StridedPooledKVAttention` and `BlockSlowBandAttention`.

## 3. Correct Triton-Ascend 3.6.0 Patterns (Why H3's Triton Kernels Lost & How to Fix)
From Triton-Ascend 3.6.0 official programming guide & `triton.backends.ascend.compiler.NPUOptions`:
1. **Persistent Physical-Core Grid (NEVER launch 1500-4600+ tiny programs!)**:
   - Ascend 950PR has **56 Vector Cores** (28 AI Cores). Large grids queue in batches and pay massive core-startup overhead.
   - Always use persistent grid `(56,)` for Vector kernels (or `(28,)` for CV Cube+Vector kernels) with an inner loop:
     `for block_idx in range(pid, num_blocks, num_cores):`
   - Also benchmark `TRITON_ALL_BLOCKS_PARALLEL=1`, `multibuffer=True`, `enable_ubuf_saving=True`.
2. **Avoid MTE2 <-> Vector Sync on Masked Loads**:
   - Pass `care_padding=False` on `tl.load(..., mask=..., care_padding=False)` whenever padded lanes are masked out downstream (otherwise Triton-Ascend inserts a Vector zero-init before MTE2 DMA, serializing memory and compute).
   - Prefer `int32` indices/offsets; keep last axis 32-byte aligned (16 bf16 elements).
3. **Why H3's DWConv1d Triton kernels overflowed UB / ran slow — and the right UB-reuse layout**:
   - H3's `_dw_fwd` / `_dw_dx` / `_dw_dw` loaded a 3-D `(BC, K, BT)` tensor with negative offsets `ot + ok - K//2`. Two fatal flaws per Ascend best practice (`003-fused-cat-slice-conv1d.zh.md`):
     (a) Negative offsets degrade DMA to scalar scattered loads.
     (b) Materializing 3-D `(BC, K, BT)` expands memory by `K=31x`, blowing the 256 KB UB.
   - **Correct Ascend UB-reuse design for Depthwise Conv1d (Fwd, dX, and dW)**:
     - For H4s L1 (`N=1488, C=384, T=258, K=31`): a full padded time row is only `T + K - 1 = 288` elements (= **576 bytes in bf16**, **18 x 32B aligned!**).
     - Load a 2-D contiguous padded tile `(BC, T_pad=288)` ONCE into UB via single clean DMA (for `BC=16`, `16 * 288 * 2B = 9.2 KB` — fits 256 KB UB with huge headroom for double-buffering!).
     - Inside UB, iterate `k in range(K)` using `triton.language.extra.cann.extension.extract_slice(x_buf, offsets=(0, k), sizes=(BC, T), strides=(1, 1))` (or `insert_slice` for zero-padding in UB instead of negative GM offsets!).
     - Zero global re-reads, zero `AsStrided`, zero `im2col`, zero 3-D UB blowup!
     - For `dW`: each of the 56 Vector Cores loops over a slice of `(n, c_block)`, loads `x_pad (BC, 288)` and `dy (BC, 258)` once, accumulates `dw[BC, K]` in FP32 registers/UB across all `n` assigned to that core, and writes `dw` ONCE (at most 56 partial writes, or zero atomics via small `[56, C, K]` reduction buffer).
     - Also check `triton.language.extra.cann.extension.conv1d` built-in and benchmark larger `S2_TRITON_BMM_CHUNK` (64 / 128 / 256) as an immediate baseline since H4s HBM is only 17.3 GB.

## 4. Graph Capture / Static Compilation (Key to Reaching ~10 steps/s)
- Training shapes are 100% static (`batch=24, audio=131584`).
- Once the STFT/ISTFT D2H sync is removed, test the `torch_npu 2.13.0` compile backends on the static training step (start with model core forward+backward, then whole step):
  1. `torch.compile(model, backend='npugraphs')` (ACLGraph static capture — eliminates Python/host dispatch overhead of 1000+ small ops)
  2. `torch.compile(model, backend='inductor', mode='reduce-overhead')` (Triton-Ascend automatic elementwise fusion + ACLGraph)
  3. `torch.compile(model, backend='npugraph_ex')` (FX in-place & pattern fusion passes + ACLGraph)
- Note: Lightning wrapper may need compiling `pl_module.model` (or `_forward_mask_core`) rather than the whole LightningModule. Keep the 200-step loss curve control gate (<=1% drift vs eager) for every admitted change.
