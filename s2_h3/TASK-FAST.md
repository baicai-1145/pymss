# 任务：把 H3（BS-BandMixFormer-PKV）的 CPU 推理实时率从 0.35 压到 0.125

**只改实现，不改数学。** 权重、损失、评测窗口、超参一律不动。每一步都要过两道闸门：**数值等价**（同一输入同一权重，输出 max|Δ| 为 0，或只有 GEMM 累加顺序级别的 ≤1e-5 相对差）和**实测更快**（8 核 `taskset`+`OMP`、FP32、batch 1、8 s 块）。不过闸门就记录数字、回退，别留着。

## 已经做掉的（我做的，在 `entry.py`）

`MSC.forward` 从每层 3 次跨步拷贝（33+66+66 MB）降到 2 次（66+33 MB），`proj_out` 的 GEMM 不再吃跨步输入。本地验证：前向逐位相同，梯度除 `proj_out.weight`（GEMM 累加顺序，相对 8.4e-6）外逐位相同。**这项还没在 EPYC 上量过**——先量它。

## 目标阶梯（8 核 EPYC，实测基线 2847 ms / 8 s）

| 台阶 | RTF | 每 8 s | 手段 |
|---|---|---|---|
| 现在 | 0.35 | 2847 ms | — |
| A | 0.24 | ~1900 ms | 纯实现层（不影响质量） |
| B | 0.156 | ~1250 ms | 再砍 40% FLOP（要质量预算） |
| C | 0.125 | 1000 ms | B + compile 收益，或 16 核 |

## 第一步：先把归属钉死（不要跳过）

`fastbench.py`（已在本目录，CPU-only，8 线程）：
```bash
ARM=H3 S2_DEVICE=cpu S2_THREADS=8 S2_BENCH_OUT=fastbench-epyc.json python fastbench.py all
```
给出：深度卷积 6 个候选的 ms/max|Δ|、整前向的 baseline / channels_last / torch.compile 的 ms/max|Δ|、**每层 `temporal[i]` 与 `freq[i]` 的分段耗时**、**按算子 self 时间的 top-14**。这张表决定下面每一项的先后。已知的整数：2847 ms = 频率支路 1449 + 时间支路 1152 + mask head 139 + 其它 107；262.6 GFLOP，FFN 那 150 GFLOP 已跑到 245 GFLOP/s，其余 112 GFLOP 只有 ~50 GFLOP/s。

## 第二步：按收益顺序做（每项独立提交，带前后 ms 和 max|Δ|）

1. **深度卷积**（6 个 conv，现在约 300 ms；真实算术 4.1 GFLOP，内存下限约 4 ms/conv）。候选已在 `fastbench.py` 里：`conv1d`、`conv2d channels_last`、`shift-add loop`、`shift-add blocked t={32,64,128}`。Mac 上 shift-add 是 conv1d 的 3.8×，但 EPYC 的 L3 只有 16 MB/CCX，分块大小要按实测选。**取最快且数值通过的**；都不够快就上 `torch.compile` 编译这个 conv（inductor 会生成带寄存器分块的 C++ 内核）。
2. **`torch.compile` / freeze 整模型**：先量收益再决定是否启用。注意 `torch.compile` 下的等价性用同样闸门。这条可能顺手吃掉大量逐元素和拷贝。
3. **attention 路径**（freq 支路里除 FFN 之外的那些）：QKV 是否已经是一个 GEMM、`rotate_qk_fast_bnhd` 的 6 趟跨步元素乘能否合并成一次、`q.transpose(1,2)` 是否让 SDPA 走慢路径（seq=62 很短，batch=690，heads 8）。目标 737 → 350 ms。
4. **mask head（139 ms）+ band unite + ISTFT**：融合成一次遍历，去掉中间的 permute/contiguous。
5. **逐元素链**：RMSNorm 的 div、Gate、SiLU、residual add——用 `torch.compile` 或手工融合，减少 dispatch 和中间张量。

## 第三步：报告

每完成一项，在 `TRITON.md` 旁边的 `FASTPATH.md` 追加一行：项目、改前 ms、改后 ms、max|Δ|、是否启用。最后给一张 8 核 FP32 batch 1 的整模型表：baseline（2847）、逐项叠加后的结果、RTF。

## 纪律

- **不要动训练路径的语义**。推理快路径放在 `if not self.training` 后面；训练侧必须仍然通过 `entry.py check`（`PREFLIGHT_PASS`）。
- 每次提交前跑 `pytest`（若有）或至少 `entry.py check`。
- 数值闸门用同一份输入和权重，随机种子固定；报告 max|Δ| 时必须同时给出参考张量的量级。
- 不要在 NPU 上跑这些测量（它们抢 CPU 和内存带宽），NPU 保持空闲。
