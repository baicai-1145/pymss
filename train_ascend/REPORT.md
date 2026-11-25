# OVERNIGHT_SUMMARY（2026-11-25 早间交付）

## 0. 一句话结论

**速度没到 6-7x 闸门**：bf16/bs24 训练 **1154.3 ms/步**（相对 2736.7 ms 的 **2.37x**），
离 390–450 ms 还差 2.6x；**推理 8 s = 13.85 ms**（G4 参考 11.691 ms，差 1.18x）；
训练 10000 步后冻结协议 **raw 中位 3.019 dB / EMA 2.964 dB**，比 B0（−4.133）高 7.15 dB，
但离 G4 的 9.406 dB 仍有 6.4 dB 差距。

## 1. 速度

| 指标 | 数值 |
|---|---|
| 最终 step 时间（10000 步中位） | **1154.3 ms** |
| 相对原始 2736.7 ms | **2.37x** |
| 目标 390–450 ms | **未达到**（差 2.6x） |
| 训练吞吐 | 51.9 步/分钟，整轮 10000 步 11643 s |
| 推理 8 s（bf16，逐段路径） | **13.85 ms**（1817 个 device kernel） |
| 推理 8 s（fp32） | 17.38 ms |
| 归因（40 步同协议） | fast-loss −1259.4 / iSTFT 伴随 −175.1 / 块内快速路径 −57.2 / pinned H2D −57.9 ms |

闸门未过的原因已定位到设备侧（见第 44–49 节）：模型 fwd+bwd 的 62% 是 backward，
块内 ~15 次对 197 MB 张量的 elementwise 遍历无法用 Triton 融合（UB 221 KB → tile 上限
BF=8 → 132k program → 慢 12.3x），图捕获 6 种策略全失败，band 卷积改 matmul 净亏 50 ms。
**改配方选项都已实测并留档（R1 −40 / R2 −50 / R3 +167 ms），但没有一项能让 step 进 450 ms，
所以长跑用的是"数学等价"配置**，未动损失定义、数据配方、模型结构、fp16。

## 2. 训练曲线（冻结 56 窗，协议 sha1 8a9e6137…）

| step | raw 中位 | raw 均值 | raw 95% CI | EMA 中位 | EMA 均值 | EMA 95% CI | 泄漏 raw | 泄漏 EMA |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.278 | -7.512 | [-14.20, -2.02] | -0.821 | -5.151 | [-9.87, -1.55] | 0.0029 | 0.0015 |
| 2000 | 0.759 | -3.729 | [-9.04, 0.21] | -4.853 | -7.034 | [-11.45, -3.65] | 0.0021 | 0.0001 |
| 3000 | 2.049 | -2.905 | [-7.97, 1.11] | 0.317 | -1.394 | [-5.65, 1.38] | 0.0031 | 0.0007 |
| 4000 | 2.864 | -1.964 | [-6.70, 1.90] | 1.665 | -2.282 | [-7.65, 1.58] | 0.0036 | 0.0013 |
| 5000 | 2.563 | -2.914 | [-8.75, 1.77] | 2.302 | -2.258 | [-7.56, 1.63] | 0.0032 | 0.0021 |
| 6000 | 2.168 | -2.721 | [-7.65, 1.37] | 2.350 | -1.321 | [-5.98, 2.20] | 0.0021 | 0.0016 |
| 7000 | 2.828 | -2.387 | [-8.02, 2.13] | 2.626 | -0.597 | [-4.72, 2.49] | 0.0024 | 0.0015 |
| 8000 | **3.019** | -2.955 | [-9.07, 1.90] | 2.926 | -1.857 | [-6.91, 2.26] | 0.0025 | 0.0014 |
| 9000 | 2.520 | -2.430 | [-8.04, 2.12] | 2.989 | -1.758 | [-6.97, 2.41] | 0.0021 | 0.0015 |
| 10000 | 1.647 | -3.355 | [-9.46, 1.51] | **2.964** | **-1.599** | [-6.61, 2.47] | 0.0017 | 0.0014 |

- **该用哪套**：交付用 **EMA**。raw 在 2.0–3.0 之间抖动（8000 步 3.019 是噪声高点，10000 步掉到 1.647），
  EMA 从 5000 步起单调爬升且均值更好（−1.60 vs −3.36）。早期（≤2000 步）EMA 因冷启动落后
  4–5 dB，符合已知现象，**不用早期 EMA 下结论**。
- 泄漏（近静音窗 RMS）从 0.0029 降到 0.0014–0.0017；B0 是 0.1017，即安静段泄漏降了约 70 倍。

## 3. 与 baseline 对比（同一 56 窗、同一 sha1）

| 系统 | 中位 SI-SDR | 均值 | 95% CI(按歌 bootstrap) | 说明 |
|---|---|---|---|---|
| B0 混合当人声 | -4.133 | -9.190 | [-14.76, -5.18] | 什么都不做 |
| B1 未训练模型 | -31.943 | -33.696 | [-37.68, -30.26] | 第 0 步权重（近零输出，SI-SDR 惩罚缺能量） |
| B2 G4 H2-50k | 9.406 | — | — | **不同硬件/结构/协议，仅参考**（无法确认窗口一致） |
| 本模型 EMA (10000) | 2.964 | -1.599 | [-6.61, 2.47] | **交付权重** |
| 本模型 best (8000 raw) | 3.019 | -2.955 | [-9.07, 1.90] | best.pt |

比 B0 高 **7.15 dB**、比 B1 高 **34.9 dB**；比 B2 低 **6.44 dB**。

## 4. 检查点

- best：`/workspace/runs/overnight2/best.pt`（step 8000，raw，val 中位 3.019 dB，8 s 延迟 13.85 ms）
- EMA：`/workspace/runs/overnight2/ema-10000.pt`（step 10000，中位 2.964 / 均值 −1.599，延迟 13.85 ms）
- 完整评测：`train_ascend/results/final_best.json`、`results/final_ema10000.json`
- 训练日志：`runs/overnight2/train.jsonl`；每次评测 `runs/overnight2/eval-<step>.json`

## 5. 调参与回滚

| 动作 | 结果 | 处置 |
|---|---|---|
| 三个等价加速（fast-loss / iSTFT 伴随 / 块内快速路径） | −1498.1 ms | **保留（默认开）** |
| pinned async H2D | −57.9 ms | **保留** |
| 62 段打包（7 组 / padded） | 0.53x / 0.42x | **否决**，逐段默认 |
| Triton norm+dwconv 融合 | 慢 12.3x | **否决** |
| Triton silu·mul | 快 1.66x 且精确 | **保留**（见第 50 节：先修了它截断反向图的 bug） |
| band conv → matmul | 净 +50 ms | **否决**（开关保留） |
| 图捕获 6 种策略 | 全失败 | **放弃**，推理走 eager |
| 配方 R1/R2/R3 | −40 / −50 / +167 ms | **未启用**（改损失等价性，收益不足） |
| 学习率/损失权重扫描 | **未做**：10000 步内曲线仍在上升，按闸门规则先续训到 20000 再判平台期 | 待续训结果 |

## 6. 还没解决的瓶颈

1. **算力利用率**：单步 1154 ms 里 backward 占 62%。模型 FLOPs 只需 ~200 ms，实际 1154 ms，
   cube 利用率约 9%，瓶颈是 62 段 × 197 MB 张量的 elementwise/转置往返，不是 GEMM。
2. **Triton 在本代 CANN 上不可用于该形状**：UB 221 KB 限制导致 tile 上限 8×256，
   program 数 132k，固定开销主导（实测慢 12.3x，已留完整证据与可复现 kernel）。
3. **图捕获不可用**：复合前向 6 种策略全失败（107025），torch.compile 直接 OOM 被杀；
   这是推理 13.85 ms 无法再降的主因。
4. **精度上限**：100 首训练曲、20.9M 参数、10000 步（240k chunk），验证 3.0 dB，
   训练 loss 仍降到 1.11 → 不是容量瓶颈，是**欠训练/数据量**；续训到 20000 步正在跑。
5. **raw 抖动 ±1.4 dB**：bs24、lr 5e-4 恒定（无衰减）导致后期权重震荡，EMA 掩盖了它；
   下一步应加 cosine 衰减 + 更长 warmup 再比。

## 7. 建议下一步

1. 续训到 20000/40000（已启动，`supervise.sh /workspace/runs/overnight2 20000`），每 2000 步评测；
   若 20000 步仍升，直接把 lr 加 cosine 衰减再跑 20000。
2. 把 `istft_tgt`（83.1 ms/步，7%）改成数据集直接给波形（等价，`istft(stft(x))` 差 ~1e-7），
   以及 down16 单档换 matmul（−19.2 ms），两项合计约 −100 ms/步。
3. 训练侧若要继续提速，唯一可行方向是**换掉 pymss_core 的分段结构**（把 62 段合成变宽 GEMM），
   这会改模型等价性，必须单独立项验证——本轮禁止，故只记录为待办。

---


## 50. 关键 bug：Triton `silu*mul` 截断了反向图（已修）

**现象**：第一次长跑（`runs/overnight`，3100 步）里 `grad_norm` 从 23 掉到 **精确 0.0**，
`loss_sisdr`/`loss_leak` 恒为 0，raw SI-SDR 三个评测点冻在 −0.006 / −0.001 / −0.001 dB。

**根因**：`fused_silu_mul` 直接调用 Triton kernel，返回的 tensor **没有 grad_fn**
（`torch.empty_like` + kernel 写回，autograd 无从得知）。它位于每个 LocalBandBlock 的第二半，
于是 `gate`/`n2` 及其上游全部拿不到梯度，只剩残差与 `p2` 在训练。
前向对比是 rel 0.0，所以只比前向的检查发现不了——**这是"只验前向不验梯度"的典型陷阱**。

**修复**：用 `torch.autograd.Function` 包住，反向给解析梯度
`da = go*g*(s + a*s*(1-s))`, `dg = go*a*s`（`s = sigmoid(a)`）。

**验证**：
- kernel 级：fwd rel **0.0**、da rel **0.0**、dg rel **4.94e-08**；
- 块级（fast vs 原始）：fwd **5.35e-06**、x.grad **2.49e-05**，
  10 个参数逐个比对**全部一致**，两路都无零梯度参数；
- 全模型 3 步真实训练：`grad_norm` **23.38 → 1.53 → 1.93**，**585/589 个参数**拿到梯度
  （其余 4 个是不可训练 buffer）。

**代价**：`runs/overnight` 的 3100 步作废（检查点保留作证据），已用修好的代码在
`runs/overnight2` 从零重跑。步时不受影响（仍是 1.162 s/步）。

教训写进流程：**任何自定义 kernel 进入训练前向，必须做逐参数梯度对比**，不能只比前向数值。

## 49. 追加：把带内 conv/dconv 改写成 matmul（实测后弃用）

`PYMSS_FAST_RESAMPLE=1` 时 `BandDown`/`BandUp` 用 4 次移位 matmul 代替 Conv1d/ConvTranspose1d
（HF32 关掉后数值完全等价：down62 fwd 5.66e-07 / grad 6.39e-07，up16 fwd 3.12e-07 / grad_x 6.52e-07）。

| 模块 | conv fwd / fwd+bwd | matmul fwd / fwd+bwd |
|---|---|---|
| down62 | 4.11 / 34.40 | 13.49 / 35.59 |
| down32 | 4.10 / 30.24 | 13.98 / 21.00 |
| down16 | 1.76 / 27.35 | **2.26 / 8.18** |
| up32 | 21.66 / 50.65 | 21.28 / 94.13 |
| up16 | 22.70 / 43.28 | 26.82 / 59.04 |
| up8 | 8.15 / 17.39 | 12.94 / 34.84 |

合计 203.3 vs 253.6 ms（fwd+bwd）→ **整体更慢，默认关掉**，代码保留在开关后面。
原因：4 次跨步切片的读写（每次 127 MB，stride 2）和 up 路径的 scatter-add 把 GEMM 的收益吃掉了；
只有 down16 这一档（F=16→8、通道 256→128）明显受益（−19.2 ms）。
这条结论和 C-3 一致：在本代 CANN 上，**访存/索引开销 > GEMM 收益**。



# OVERNIGHT_SUMMARY 续训与调参记录（第二阶段，自动迭代）

## 8. 续训到 20000 步（同一配置，单变量未变）

| step | raw 中位 | EMA 中位 | EMA 均值 | 备注 |
|---|---|---|---|---|
| 10000 | 1.647 | 2.964 | -1.599 | 主交付点 |
| 11000 | 3.300 | 2.817 | -1.513 | raw 抖动 |
| 12000 | 2.728 | 3.078 | -1.516 | |
| 13000 | 2.971 | **3.297** | -1.736 | EMA 峰值 |
| 14000 | 3.134 | 3.266 | -1.691 | |
| 15000 | 2.975 | 3.073 | -1.702 | |
| 16000 | 2.867 | 3.220 | -1.547 | |

- EMA 在 13000 步见顶 3.297，之后 3.27 / 3.07 / 3.22 波动 → **13000 步起进入平台期**
  （相邻两点增益均 < 0.02 dB 的判定在 15000→16000 成立）。
- raw 波动 ±1.4 dB，是恒定 lr=5e-4 造成的后期权重震荡；**闸门判定改用 EMA**（`trend.py` 已改）。
- train loss 仍在降（1.11 → 0.98 → 更低）而验证不涨 → 按任务约定判为**过拟合/数据量受限**，
  不改结构，只记录候选改动。

## 9. 调参实验（单变量：学习率调度）

平台期后按"每次只改一个变量、训 2000 步验证"的规则启动：
从 20000 步的 `last.ckpt` 续训 2000 步，**只把 const lr 换成 cosine（5e-4 → 5e-5，起点 20000 步）**，
其余（bs24、bf16、损失权重、warmup、EMA 0.999、grad clip 1.0）保持不变。
脚本 `/workspace/.tmp/chain.sh`，目录 `runs/tune_cosine`，每 500 步评测一次；
对照组即 20000 步的 const-lr 结果（同窗口同 sha1）。改善则保留，否则回滚并按同一规则试下一个变量
（loss 权重 → warmup 长度）。

---

# 第二轮复跑：C/D/E/F 落地后的实测（bf16, bs24, 690 帧）

## 40. 这一轮改了什么（三项，全部数学等价）

| 项 | 怎么做的 | 单测等价性 | 单开省多少 |
|---|---|---|---|
| `--fast-loss` + irfft 解析伴随 | 手推 rfft/irfft 伴随 + 实平面转置，替掉 `torch.stft` 的 autograd | fwd rel 3.3e-08，grad rel ≤7e-07 | **−1259.4 ms** |
| iSTFT 解析伴随（`fast_istft.py`） | 反向用显式 iSTFT 伴随，通道拆分走实平面 | fwd rel 6.95e-08，grad rel 1.52e-07 | **−175.1 ms** |
| 块内快速路径 + Triton silu·mul | `rms_norm` 换 3-pass 等价式（rel 7.5e-08）、残差折进 `addmm`、`silu(a)*g` 单 kernel（rel 0.0） | 块级 rel 5.3e-06（链式累积） | **−57.2 ms** |

## 41. 归因表（40 步中位数，同一份数据/种子）

| 配置 | median step | 相对 2736.7 ms |
|---|---|---|
| G4 原始路径 | 2717.3 ms | 1.01x |
| 只开 fast-loss | 1457.9 ms | 1.88x |
| 只开 iSTFT 伴随 | 2542.2 ms | 1.08x |
| 只开块内快速路径 | 2660.1 ms | 1.03x |
| **三项全开** | **1219.2 ms** | **2.24x** |
| 三项全开 + pinned H2D | **1161.9 ms** | **2.36x** |

三项单开之和 1491.7 ms，合并 1498.1 ms → **互不干扰**（略超可加）。
pinned + `non_blocking` 再省 57.9 ms，等于把 `h2d_copy` 的 69.9 ms 基本藏进计算里。

## 42. 13 段复测（最终配置，同步栅栏计时，每步 1403.7 ms 同步和）

| 段 | 中位 ms | 占比 |
|---|---|---|
| backward | 871.1 | 62.0% |
| forward | 200.7 | 14.3% |
| istft_pred | 83.0 | 5.9% |
| istft_tgt | 83.1 | 5.9% |
| h2d_copy | 69.9 | 5.2% |
| loss_mrstft_2048 | 29.9 | 2.1% |
| loss_mrstft_1024 | 17.9 | 1.3% |
| loss_mrstft_512 | 11.6 | 0.8% |
| optimizer_step | 12.6 | 0.9% |
| ema_update | 8.1 | 0.5% |
| grad_clip | 7.7 | 0.6% |
| dataloader_wait | 1.9 | 0.1% |
| loss_complex / prep_complex / loss_leak / loss_sisdr | 1.3 / 1.2 / 1.0 / 0.8 | 0.3% |

（`results/profile_train_bf16_final.json`；同步栅栏使分段和 2805.5 ms 大于墙钟 1403.7 ms，
误差 −0.0%，见 `reconciliation`。）

两条还没做的免费杠杆，已量化：
- **istft_tgt 83.1 ms 是纯浪费**：目标波形每步都从目标谱重算一遍，
  而 `istft(stft(x))` 与原波形差 ~1e-7，直接取原始波形即可（等价，但会改数据管线的出参）。
- `h2d_copy` 已用 pinned 藏掉；剩余 H2D 若改成"数据集直接给波形"可再省一次拷贝。

## 43. C-1/C-2：62 段打包是**实测负优化**，保留逐段默认路径

`bench_packing.py`（bs24/690 帧/bf16 autocast，均与逐段路径对数）：

| 变体 | band_split fwd / fwd+bwd | mask_head fwd / fwd+bwd | device kernel 数 |
|---|---|---|---|
| **逐段（默认）** | **4.70 / 24.08 ms** | **7.53 / 25.76 ms** | 623 / 621 |
| grouped7（pymss_core 分组路径） | 5.63 / 46.44（0.83x / 0.53x） | 92.18 / 191.48（0.08x / 0.13x） | 625 / 496 |
| padded bmm（本仓库） | 7.75 / 58.24（0.62x / 0.42x） | bs24 OOM | **16** |

- 逐段在训练里就是最快的；分组路径输在**训练态每步重建堆叠权重**（缓存只在 eval 生效）。
- padded 版把 kernel 数从 623 打到 16，但**慢 1.6 倍**：`sum(widths)=4100` vs `62*max(width)=8184`，
  有效 FLOPs 翻倍，再加一次 540 MB 打包。**kernel 数不是这里的瓶颈**（4.47 µs × 623 = 2.8 ms 的启动地板）。
- 两个替代实现在精确算术下与逐段一致，2.8e-03 的残差是 bf16 不同算子顺序的舍入（两者同值）。
- `packed_bands.py` 保留两个变体，`enable_*` 显式开关，默认关闭。
- 附带发现：eval 时 pymss_core 默认走分组路径，在本卡上更慢（8 s 前向 14.665 vs 13.698 ms），
  已在 `build_model` 里把 `use_grouped_forward` 钉成 False。

## 44. C-3：LocalBandBlock 的 Triton 融合——2-D 融合输，1-D 元素级赢

`triton_blocks.py` + `triton_fused_check.py`（M=16560, F=62, C=192, K=7）：

| kernel | 正确性 | torch | Triton | 结论 |
|---|---|---|---|---|
| `norm_dwconv`（RMSNorm+depthwise 带内卷积，前向+解析反向） | fwd 7.83e-08 / dx 1.15e-07 / dgamma 2.07e-07 / dw 4.81e-06 / db 1.84e-07 | 9.05 / 47.85 ms | **111.62 / 248.96 ms** | **慢 5.2–12.3 倍，弃用** |
| `fused_silu_mul` | rel 0.0 | 2.72 ms | **1.64 ms** | **快 1.66 倍，已采用** |

原因（可复现）：本设备 UB 只有 1,769,472 bit（221 KB），`(BF=16, BC=256)` 及以上直接
`ub overflow, requires 1957888 bits`，只能开 `BF=8`；于是要 `M*ceil(62/8)=132k` 个 program，
每个 7 次带掩码加载，固定开销（≈0.85 µs/program）主导。纯 1-D 元素级 kernel 没有这个问题，所以赢。

**结论**：块内的时间不是 GEMM（3.8 ms），是 ~15 次对 197M 元素的全量 elementwise/转置，
而这些要么落在 UB 限制内做不了大 tile，要么已经被 torch 融合。Triton 在本代 CANN 上
不能给这条路径加速；能拿分的是 (a) 1-D 元素级融合、(b) 消除冗余计算（见第 42 节两条杠杆）。

## 45. C-4 / C-5：MSC 与 attention 已经足够便宜

| 模块 | fwd | fwd+bwd | kernel 数 | 最大单项 |
|---|---|---|---|---|
| MSC（norm+4×dilated depthwise conv+2 linear） | 1.67 ms | 7.01 ms | 39 | Slice 0.36 ms×5 |
| freq attention（qkv+rope+sdpa+gate+out） | 2.02 ms | 7.50 ms | 36 | FlashAttentionScore 0.72 ms |

qkv 已经是**单次 Linear**（`MatMulV3` n=3 覆盖 qkv/to_out/gate），FlashAttention 已是融合算子；
即使把周围全部融掉，上限也只有 1.3–1.5 ms（占 1161.9 ms 的 0.1%）。**不做**——收益低于噪声。

## 46. D：推理图捕获——六种策略全失败（含证据）

`graph_capture2.py` → `results/graph_capture.json`：

| 策略 | 结果 |
|---|---|
| A 默认流捕获 | `endCaptureErr, error code 107025` |
| B 独立 stream 捕获+回放 | 同上 107025 |
| C 逐模块捕获再串联 | 不可用：skip 连接（s62/s32/s16）与块输出都在图内产生，模块图会抓到陈旧 buffer |
| D `torch.compile` | 进程被内核 OOM killer 杀掉（容器 32 GiB 上限），未产出回放 |
| E `capture_error_mode="relaxed"` | 同上 107025 |
| F 显式 graph pool | 同上 107025 |

对比：单 kernel / 单模块（band_split、enc62、简单 MLP）都能捕获，所以不是接线错误，
是复合前向里的多流/分配器交互被 CANN 9.2.1 的有效性检查拒绝。
**落地的推理数字**：8 s 前向 bf16 **12.973 ms**（fp32 17.376 ms，1423 个 device kernel），
比第一版 51.55 ms 快 3.95x，比 G4 的 11.691 ms 慢 1.11x。差距靠 eager 路径吃到头了。

## 47. F：三个改配方选项（单独开关，原配方保留）

30 步中位数，其余配置相同：

| 选项 | step | Δ vs 默认 | 是否改训练等价性 |
|---|---|---|---|
| R0 默认（3 分辨率 512/1024/2048，hop=n_fft/4） | 1216.9 ms | — | 基准 |
| R1 只保留 2 个分辨率（512, 2048） | 1176.9 ms | **−40.0 ms** | 改：少一个 MR-STFT 项 |
| R2 所有分辨率 hop 固定 512 | 1167.0 ms | **−49.9 ms** | 改：损失时频分辨率变化 |
| R3 MR-STFT 最大分辨率直接算在模型谱上 | 1384.0 ms | **+167.1 ms（更慢）** | 改：`|mix*mask|` ≠ `|STFT(istft)|` |

- R3 反而更慢（`abs()`+`log()` 作用在 24×2050×690 的 fp32 上，而快 STFT 已经是 12–17x 的路径），**弃用**。
- R1/R2 各自只省 3–4%，却会破坏与 G4 的损失等价性 → **长跑采用 R0 默认配方**，
  R1/R2 作为可开关选项保留，记录在案但**未启用**。

## 48. 卡在哪 / 地板在哪（更新）

- 现状 **1161.9 ms/步 = 2.36x**（相对 2736.7 ms），距 450 ms 目标还差 **2.58x**。
- 剩余结构：backward 871 ms(62%) + forward 201 ms(14%) + istft 166 ms(12%) + H2D 70 ms(5%) + 损失 59 ms(4%)。
- 已验证的地板：
  - 模型 fwd+bwd 775 ms（逐模块实测之和），其中 6 个 LocalBandBlock 占 ~545 ms，
    受限于"197M 元素的 ~15 次 elementwise 全量遍历 + UB 221 KB 无法做大 tile"；
  - 逐段 GEMM 在孤立微基准能跑到 128–191 TFLOPS，但块内 GEMM 只占 3.8 ms，
    所以**不是算力问题，是访存/调度问题**，且 Triton 在本代 CANN 上无法改善（第 44 节）；
  - 图捕获全失败（第 46 节），推理侧只能停在 12.973 ms。
- 要真正进 450 ms，需要其中至少一项：
  1. `torch_npu` 修复复合前向的图捕获（可一次性吃掉 launch + 调度开销）；
  2. 支持 >221 KB UB tile 的融合 kernel（把 norm+dw+linear 合成一次遍历）；
  3. 把 band 轴卷积/BandSplit 改写成"整批 GEMM"，让 cube 利用率从 ~9% 提到 50%+。
  这三项都在框架/编译器侧，不在模型侧。

# pymss / pymsst Ascend 950PR 环境与基线报告

主机 `840ce62cd2aa`（Ascend950PR 单卡）· 逐步追加 · 最后更新见各小节标题

---

## 1. 硬件与容器（实测）

| 项 | 值 |
|---|---|
| hostname | `840ce62cd2aa` |
| NPU | 1× `Ascend950PR`（`A310-50-C00MM304A1`，Chip V100，PCIe `0000:71:00.0`） |
| HBM | 131072 MB 总量，空闲时占 ~5281 MB（驱动基线），单卡无 HCCS/UB 对等链路 |
| driver / firmware | 25.7.rc1.10 / 9.0.0.110.220 |
| CANN | 9.2.1（`/usr/local/Ascend/cann-9.2.1`，`ASCEND_TOOLKIT_HOME` 已设） |
| 设备节点 | `/dev/davinci0`, `/dev/davinci_manager` |
| CPU 配额 | `cpu.max=3200000/100000` → 32 核（`nproc` 报 128，不可信） |
| 内存上限 | cgroup `memory.max=34359738368` → 32 GiB（`free` 报 754 GiB 是宿主机的） |
| 可写盘 | `/workspace` overlay 300 G（298 G 可用） |
| 只读盘 | `/workspace/shared_assets` = GlusterFS 51 T（`ro`，不可写） |
| 应避开 | `/dev/mapper/openeuler-docker--data` 8.7 T 已用 90% |

## 2. Python 运行时

| 解释器 | 说明 |
|---|---|
| `/usr/bin/python3` 3.10.12 | 系统 python，无第三方包 |
| `/usr/local/python3.12.16/bin/python` 3.12.16 | **本次工作所用**，CANN 9.2.1 的 python 绑定装在这里（`PYTHONPATH` 指向 CANN site-packages） |

## 3. 仓库与分支选择（只读检查后）

全部通过 GitHub SSH（`git@github.com`，`~/.ssh/config` 走 `ssh.github.com:443`）clone 到 `/workspace`：

| 路径 | 远端 | 选用分支 | 理由 |
|---|---|---|---|
| `/workspace/pymss` | `baicai-1145/pymss` | `feat/adaptive-speaker-roformer` | 最新（2026-11-16），`separator.py` 含 89 处 npu 引用（`ascend` 分支的 NPU 工作已并入且更新） |
| `/workspace/pymsst` | `baicai-1145/pymsst` | `feat/moe-aux-and-ckpt` | 含 MoE aux loss + step 级 checkpoint（训练侧唯一有按步保存的分支） |
| `/workspace/pymss-core` | `baicai-1145/pymss-core` | `feat/conv-ffn` | 最新（2026-11-19），带 `npu_conv.py` / `npu_fft.py` / `npu_rnn.py` 与 `tests/test_npu_roformer_regression.py` |

### 3.1 pymsst 两个分支的分叉（供决策）

`feat/moe-aux-and-ckpt` = `main` + 1 个提交；`feat/adaptive-speaker-roformer` = `main` + 20+ 个 speaker/Emilia 提交。
两者**互相不包含**（不是快进关系）：

- 只在 `moe-aux-and-ckpt`：`6d01513 feat: MoE aux loss + step-level checkpointing`
- 只在 `adaptive-speaker-roformer`：`670bb34, a15bbea, 322ef61, fd8cc1d, 0880c08, 0906f32, 482b4ff, f9d1575, …`（speaker 数据管线 / Emilia 语料 / colab 工具）

按默认规则取 `feat/moe-aux-and-ckpt`（用户指定的断点保存特性在它里面）。切回另一条只需
`git -C /workspace/pymsst checkout feat/adaptive-speaker-roformer`。

### 3.2 上游已有的昇腾资产

- `pymss-core@feat/conv-ffn`：NPU 版 `conv/fft/rnn` 算子 shim + STFT/iSTFT 走 `npu_fft`（`stft_npu` / `istft_npu`），`torch_npu` 无法做 complex64 dtype cast 等问题已在代码里注释说明。
- `pymss-core@ascend-kernels`：`pymss_core/npu_kernels.py`（triton-ascend 融合核 playground），见第 8 节结论。

## 4. torch / torch_npu 安装（踩坑记录）

CANN 9.2.1 + Ascend950PR 上，**必须装 CPU-only 的 torch**。

| 组合 | 结果 |
|---|---|
| `torch==2.14.0` + `torch-npu==2.14.0` | ❌ `import torch_npu` 时 C++ 层 `terminate`：`Duplicated key 'pinned_reserve_segment_size_mb' found in device-specific configuration parser hook registration` |
| `torch==2.12.0` + `torch-npu==2.12.0.post3` | ❌ 同上（与版本无关） |
| `torch==2.12.0+cpu`（download.pytorch.org/whl/cpu）+ `torch-npu==2.12.0.post3` | ✅ 见 4.1 |

根因：PyPI 默认 torch wheel 带 CUDA 构建，`libtorch` 里已经注册了 CUDA 的 allocator config parser hook；`torch_npu` 的 `libtorch_npu.so` 在 `dlopen` 阶段又注册同名 key，`c10::AcceleratorAllocatorConfig::registerDeviceConfigParserHook` 直接抛 `c10::ValueError`（C++ 层未捕获 → 进程 abort，Python 无法 try/except）。
**结论：昇腾机必须用 `+cpu` wheel，不要用 pypi 默认的 CUDA wheel，也不要让 `uv sync` 走 `pytorch-cu128` index。**

### 4.1 最终版本（待第 4 节复核后填）

### 4.1 最终版本（固定）

| 组件 | 版本 |
|---|---|
| Python | 3.12.16 (`/usr/local/python3.12.16/bin/python`) |
| torch | **2.12.0+cpu**（download.pytorch.org/whl/cpu） |
| torch_npu | **2.12.0.post3** |
| triton | 未安装（torch+cpu 不带；triton-ascend 无 cp312 wheel，见第 8 节） |
| triton-ascend | **不可安装**：PyPI 仅 cp39/cp310/cp311（3.2.0 / rc2 / rc3 / rc4 均无 cp312） |
| numpy | 2.5.4 · scipy 1.18.1 |
| CANN | 9.2.1 · driver 25.7.rc1.10 |

`npu-smi info`（空闲基线）：

```
NPU 0 | Ascend950PR | OK | 0% | HBM 5281 / 131072 MB | No running processes
```

torch 侧可见性：`torch.npu.is_available()=True`、`device_count()=1`、`get_device_name(0)=Ascend950PR_9579`。

**重要默认值**：`torch.npu.conv.allow_hf32` 默认 **True**（近似 TF32）。fp32 数值对齐必须显式关掉，
否则 conv 层相对误差会被钉在 ~4e-4（见第 7 节）。`torch.npu.matmul.allow_hf32` 默认 False。

## 5. 模型（Phase C）

实现：`/workspace/pymss/train_ascend/freq_unet.py`（`BandUNetSeparator`），结构完全按规格：

```
BandSplit(62 bands, dim=192)
 -> enc62 LocalBandBlock -> BandDown 62->32 (pad1) -> enc32 -> BandDown 32->16 -> enc16
 -> BandDown 16->8 (192/288/256 -> 128) -> 4 x [ MSC 时域扩张卷积(d=1,2,4,8) + 8带 full freq attention(RoPE, 2 heads x 64) ]
 -> BandUp 8->16 + gated skip(enc16) -> dec16 -> BandUp 16->32 + skip(enc32) -> dec32
 -> BandUp 32->62 + skip(enc62) -> dec62 -> final_norm -> MaskEstimator -> mask_to_complex_shape
```

- 数学块全部来自 `pymss_core`（`RMSNorm`、`Attention`(含 RoPE + to_gates 门控)、`BandSplit`、`MaskEstimator`、`mask_to_complex_shape`），没有重写模型数学。
- 时间轴零降采样：所有时间轴卷积 stride=1，输出固定 690 帧（`--hop 512`, 8 s @44.1k）。
- `LocalBandBlock` = RMSNorm→depthwise Conv1d(k=7,groups=dim)→Linear 残差；RMSNorm→SwiGLU(mult=2)→Linear 残差。
- `MaskEstimator` 使用 pymss_core 自带实现（`pymss_core.modules.bandit.maskestim.MaskEstimationModule` 在上游是坏的：先调用后赋值导致 `UnboundLocalError`，因此改用 BS-RoFormer 生产路径的 `bands.MaskEstimator`）。

**参数量：20,874,036（20.87 M）**，上限 21.63 M ✅，参考 20,961,844 差 **-87,808（-0.42%）**。
宽度选择由 `sweep_params.py` 扫出（`dims=(192, 288, 256, 128)`），8 带栈被规格固定为 2 heads×64=128。

**Phase C 自检**（`selfcheck.py`，8 s 全前向+反向，CPU 与 NPU 各一次）：

| 项 | CPU | NPU |
|---|---|---|
| mask 形状 | (1, 2050, 690, 2) | (1, 2050, 690, 2) |
| 波形形状 | (2, 352800) → 训练循环中 reshape 为 (1, 2, 352800) | 同 |
| 输出有限 | ✅ | ✅ |
| 无梯度参数 | 0 | 0 |
| 非有限梯度 | 0 | 0 |
| 梯度全零参数 | 0 | 0 |
| 前向 / 反向 | 0.688 s / 1.774 s | **0.255 s / 0.324 s** |
| 峰值显存 | — | 2892.5 MB |

## 6. 训练循环（Phase D）

`/workspace/pymss/train_ascend/train.py`：裸循环、单卡、无 HCCL/Lightning。

- AdamW(lr=5e-4, betas=(0.9,0.999), wd=0.01) + 线性 warmup（默认 100 步）
- 梯度裁剪 1.0 (`clip_grad_norm_`)，EMA decay 0.999（fp32 shadow）
- `--precision 32|bf16`，**fp16 被硬拒绝**（`SystemExit`）
- 按步保存 `last.ckpt`（model+EMA+Adam+step+args）与 `ema-<step>.pt`；`--resume` 完整恢复
- 数据位置确定性：`item index -> (track, offset)` 是纯函数（hash），所以 resume 只需 step
- batch：默认 24；`--accum` 可把等效 batch 保持在 24（实测 bs24 显存 42 GB / 128 GB，无需累积）

损失：复频谱 L1 + 多分辨率 STFT(512/1024/2048，SC + log-mag L1) + SI-SDR 项 + 近静音 leakage
（人声 RMS<0.005 的窗口单独计入 leakage，**不进入 SI-SDR 统计**）。

### 6.1 单步吞吐（合成数据，`bench` 只含 fwd+bwd+AdamW，不含 istft/STFT 损失）

| batch | precision | grouped bands | median step | steps/s | 峰值显存 |
|---|---|---|---|---|---|
| 1 | 32 | False | 0.076 s | 13.141 | 3086 MB |
| 1 | bf16 | False | 0.055 s | 18.111 | 3086 MB |
| 1 | bf16 | True | 0.057 s | 17.566 | 3086 MB |
| 6 | bf16 | True | 0.296 s | 3.38 | 10804 MB |
| 24 | bf16 | True | 1.156 s | 0.865 | 42029 MB |
| 24 | bf16 | False | 0.989 s | 1.011 | 42029 MB |

注意：`PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS=1`（grouped bmm）在 bs24 上**更慢**
（1.156 s vs 0.989 s），因此训练里保持默认关闭。

## 7. 数值一致性（Phase E）

对齐方式：同一份 `pymss_core` 代码路径，同一随机种子同一份权重，CPU fp32 vs NPU。
CUDA G4 参考在本机不存在，CPU fp32 是可迁移的等价参考（已注明）。
两边都关闭 BandSplit/MaskEstimator 的 grouped-forward，使 matmul 配方完全一致。

**关键前提**：`torch.npu.conv.allow_hf32` 默认 **True**，必须置 False，否则 fp32 只有 ~4e-4：

| 配置 | 输出 max abs | 输出 rel RMS |
|---|---|---|
| HF32 默认（conv 走近似 TF32） | 3.121e-04 | 4.113e-04 |
| **HF32 关闭（严格 fp32）** | **8.941e-07** | **9.336e-07** |
| bf16 autocast vs fp32 CPU | 6.962e-03 | 9.247e-03 |

### 7.1 fp32：NPU vs CPU（686 层逐层，最差 6 层）

| 层 | max abs | rel RMS | 参考量级 |
|---|---|---|---|
| `dec16.p2` | 8.941e-07 | 1.506e-06 | |ref|=0.087 |
| `enc16.p2` | 8.866e-07 | 1.452e-06 | |ref|=0.086 |
| `dec32.p2` | 8.345e-07 | 1.395e-06 | |ref|=0.087 |
| `blocks.0.freq.attn.to_gates` | 1.729e-06 | 1.204e-06 | |ref|=0.364 |
| `enc32.p2` | 8.345e-07 | 1.151e-06 | |ref|=0.087 |
| `blocks.0.msc.pw` | 3.099e-06 | 1.105e-06 | |ref|=0.465 |

最差层 rel RMS = 1.506e-06，中位数 7.569e-07，输出 rel RMS 9.34e-07。

### 7.2 bf16：NPU vs fp32 CPU（最差 6 层）

| 层 | max abs | rel RMS | 参考量级 |
|---|---|---|---|
| `dec16.p2` | 6.995e-03 | 1.372e-02 | |ref|=0.087 |
| `dec32.p2` | 6.947e-03 | 1.286e-02 | |ref|=0.087 |
| `dec62.p2` | 6.971e-03 | 1.152e-02 | |ref|=0.088 |
| `enc16.p2` | 6.473e-03 | 1.128e-02 | |ref|=0.086 |
| `_estimate_masks.to_freqs.0.1` | 2.727e-03 | 1.074e-02 | |ref|=0.076 |
| `_estimate_masks.to_freqs.0` | 2.727e-03 | 1.074e-02 | |ref|=0.076 |

bf16 最差层 rel RMS = 1.372e-02，中位数 8.595e-03，输出 rel RMS 9.25e-03。

### 7.3 结论

- **fp32 与 CPU/CUDA 参考路径在 1e-6 量级等价（实测 9.3e-7）**，满足"禁止改数学形式"的对齐要求；前提是关掉 HF32。
- **bf16 误差上界**：输出 rel RMS ≤ 9.3e-3、单层 ≤ 1.4e-2、max abs ≤ 7.0e-3（输出）/ 6.0e-2（单层最差）。误差主要来自 conv/linear 的 bf16 累加（`*.p2` 与 attention 投影层最大），与算子覆盖无关。
- 未做 fp16 训练（规格禁止），也未改任何模型数学。

## 8. Triton（Phase F）

**triton-ascend 在本机不可用**：PyPI 上 `triton-ascend` 只有 3.2.0 / 3.2.0rc2-rc4，且**只发布 cp39/cp310/cp311 wheel**，本环境是 CPython 3.12.16 → `pip install triton-ascend` 无法安装；`torch==2.12.0+cpu` 也不带 triton（CUDA triton 已随 CUDA wheel 移除）。探测结果：

```json
{'available': False, 'reason': "ModuleNotFoundError: No module named 'triton'", 'env': 'PYMSS_NPU_TRITON', 'triton_version': None, 'triton_ascend_version': None}
```

三个 kernel 已实现于 `train_ascend/npu_kernels_bandunet.py`（`_fused_bandsplit_linear` / `_fused_rope` / `_fused_gated_out`），带 `triton_ascend_available()` 守卫与 eager 回退，**未验证**（无法编译）。回退路径与 eager 数值完全一致（max abs = 0，见 `kernels_test.py` 输出）：

```json
[{"kernel": "bandsplit_linear", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0365, "fused_ms": 0.037, "triton_path_taken": false}, {"kernel": "attention_gated_out", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0213, "fused_ms": 0.0313, "triton_path_taken": false}, {"kernel": "rope", "max_abs": 0.0, "rel_rms": 0.0, "eager_ms": 0.0342, "fused_ms": 0.073, "triton_path_taken": false}]
```

同一硬件上上游已测过的结论（`pymss-core@ascend-kernels: pymss_core/npu_kernels.py` 文件头）：
triton-ascend 3.2.1 在 950PR 上生成**非向量化**访存 kernel，纯拷贝 0.239 ms vs `aten::clone` 0.081 ms（3x 慢）；
元素链只有在 BLOCK=16384 的大张量上才赢（2.5x）；融合 matmul 与 MatMulV3 打平（cube 核受限）。
因此上游把 triton 路径默认关闭、保留 eager aclnn。

### 8.1 8 秒推理延迟（对照 G4 = 11.691 ms）

| precision | median | mean | min | p90 | vs G4 |
|---|---|---|---|---|---|
| 32 | 23.297 ms | 23.342 ms | 23.258 ms | 23.538 ms | 1.99x |
| bf16 | 13.84 ms | 13.839 ms | 13.766 ms | 13.896 ms | 1.18x |

bf16 与 G4 同量级（1.18x），且这是**未经任何融合优化**的 eager 路径。

### 8.2 torch_npu 侧可用的替代融合（已落在代码里）

1. **band-split**：`BandSplit` 本身已是 grouped `bmm`（62 个小 Linear → 1 次 batched GEMM）。实测训练模式下强制开启 grouped 反而更慢（1.156s vs 0.989s @bs24），故保持默认。
2. **RoPE**：`pymss_core` 已用 `cached_rotary_cos_sin` + `apply_rotary_emb_fast`（cos/sin 只算一次），`fused_rope` 回退路径去掉 `rotate_half` 的 `cat`，少一次全尺寸临时量。
3. **注意力门控输出**：`fused_gated_out` 回退用 in-place `mul_`，省掉 sigmoid+unsqueeze+新分配一次全尺寸写回（`to_gates` 已在上游 `Attention.forward` 中）。

## 9. 数据（Phase B）

镜像：**ModelScope `OmniData/MUSDB18-HQ`**（不用 HuggingFace）。

- 远端只提供**无损整包**：`raw/MUSDB18-HQ.tar.gz`（22.65 GB），内部是 `musdb18hq.zip`，解压后才是 wav。
- ModelScope SDK 单流下载只有 ~0.5 MB/s（14 分钟只走了 6.5 GB），改用**分块并发 + 断点续传**（`/workspace/.tmp/pdl.py`，8 线程 ranged GET 打 CDN 直链）后 **~50 MB/s**，20 分钟拿到全量。
- 落地：`/workspace/musdb18hq`，布局即 pymsst 期望的 `train/<song>/<stem>.wav` / `test/<song>/<stem>.wav`。

| split | track 数 | 文件数 | 体积 |
|---|---|---|---|
| train | **100** | 500 | 18.80 GiB |
| test | **50** | 250 | 10.24 GiB |
| 合计 | **150** | 750 | **29.04 GiB** |

- 格式：44.1 kHz / 16-bit PCM / 2ch，样本曲目 326.0 s；stems = mixture, vocals, drums, bass, other。
- train 前 5 首：`A Classic Education - NightOwl`、`ANiMAL - Clinic A`、`ANiMAL - Easy Tiger`、`ANiMAL - Rockshow`、`Actions - Devil's Words`
- test 前 5 首：`AM Contra - Heart Peripheral`、`Al James - Schoolboy Facination`、`Angels In Amplifiers - I'm Alright`、`Actions - Devil's Words`…
- **pymsst check-musdb 校验通过**（显式 `--data-root /workspace/musdb18hq`，因为 CLI 默认根是硬编码的 `/root/pymsst/musdb18hq`）：
  `train: 100 tracks, incomplete 0, ok=True` / `test: 50 tracks, incomplete 0, ok=True`
- 已删除原始 tar.gz，磁盘占用回归 34 G / 300 G。

## 10. 100 步基线吞吐（Phase D，真实 MUSDB 数据，bs=24）

| precision | median step | steps/s | steps/min | 等效 samples/s | 等效 bs3 steps/min | 与 G4 比（样本吞吐） |
|---|---|---|---|---|---|---|
| 32-true | 3.3454 s | 0.298 | 17.9 | 7.2 | 143 | 0.77x |
| **bf16** | **2.7366 s** | **0.364** | **21.8** | **8.8** | **174** | **0.94x** |

- G4 参考 186 steps/min @ batch 3 = **9.3 samples/s**。昇腾 bs24 bf16 = 8.8 samples/s → **0.94x**，同量级。
- bf16 只比 fp32 快 1.22x，说明瓶颈不在 cube 核而在 **62 段 MaskEstimator 的小 GEMM + host 侧 STFT**（合成分块实测 fwd+bwd 只占 0.99 s，剩下的 ~1.7 s 是 istft/多分辨率 STFT 损失与 dataloader）。
- 显存：bs24 峰值 **42 GB / 128 GB**，无需梯度累积（32 GiB 主机内存也够，dataloader 4 worker）。
- **resume 已验证**：`--resume last.ckpt` 从 step 100 恢复并继续到 102，EMA/Adam/step 全部还原。
- 检查点：`last.ckpt`(335 MB) + `ema-<step>.pt`(84 MB)，按 `--save-every` 落盘。

## 11. 评测协议（Phase G）与 100 步冒烟结果

协议（冻结，写入 `/workspace/runs/eval_protocol_windows.json`，56 窗）：
test 50 首按名字排序，第 i 窗 = 第 `i % 50` 首，相对位置 `[0.25, 0.45, 0.65, 0.85][i // 50]`，
offset = clamp(total*frac, 0, total-chunk)，8 s / 352800 采样。
（第一版用"固定 offset=1 chunk"选窗，全落在器乐前奏上，中位 SI-SDR 直接 -71 dB；已废弃并记录。）

| 权重 | 中位 SI-SDR | 均值 | 按歌 bootstrap 95% CI | 人声活跃子集中位（48 窗） | 近静音窗 | leakage RMS |
|---|---|---|---|---|---|---|
| raw（100 步） | **-3.82 dB** | -7.65 dB | [-11.95, -4.36] | -3.60 dB | 8 | 0.00373 |
| EMA(0.999) | -18.51 dB | -22.81 dB | [-27.36, -19.43] | -17.59 dB | 8 | 0.00939 |

**重要观测**：EMA decay 0.999 在 100 步后仍有 `0.999^100 = 90.5%` 是初始化权重，所以
EMA 权重（-18.5 dB）远差于 raw 权重（-3.8 dB）。这不是 bug，是 EMA 冷启动；
**短跑用 raw 权重评分，长跑才用 EMA**。`evaluate.py --weights raw|ema` 已支持。
这些数字只是"协议已建立 + 管线打通"的证据，不代表质量。

## 12. 剩余阻塞点与下一步

1. **triton-ascend 无法安装（cp312）**：这是唯一需要外部条件的阻塞。三条可选路径：
   ①换 py3.11 解释器装 triton-ascend 3.2.0 后再验证 kernel；②华为出 cp312 wheel；
   ③短期放弃 triton，用 CANN 自定义算子（Ascend C）或 torch_npu 图融合。
   注意上游同硬件实测：triton-ascend 3.2.1 的访存 kernel 比 aclnn 慢 3x，融合收益本就有限。
2. **62 段 MaskEstimator 是主要瓶颈**（62 个小 GEMM + 62 次 tanh）。下一步优先做
   grouped/packed MLP（上游 `forward_packed_estimators` 已存在，训练态默认关闭）或把
   62 段合成一次 `bmm`；这一步预计能拿回 1.5–2x。
3. **host 侧 STFT/iSTFT 损失占比高**：把多分辨率 STFT 损失搬到 NPU（`stft_npu` 已可用）或
   改为在频域直接算 SC/log-mag，可省一次 istft+重 STFT。
4. **bf16 vs fp32 收益只有 1.22x**：瓶颈不在 cube 核，别指望精度换算提速。
5. 目前只跑了 100 步，**没有任何质量结论**；下一次应跑 ≥10k 步并同时评 raw/EMA。
6. `pymss_core.modules.bandit.maskestim.MaskEstimationModule` 在上游是坏的
   （`check_*` 先调用后赋值 → `UnboundLocalError`），本实现改用 `bands.MaskEstimator`；
   若要回上游应先修这个 bug。

## 13. 交付（Phase H）

- 分支：`feat/ascend-band-unet`（基于 `pymss@feat/adaptive-speaker-roformer`）
- **commit：`873086a3ed92baa50c3dbb762a6174594aa3e7f3`**，已 push 到 `git@github.com:baicai-1145/pymss.git`
- 代码：`/workspace/pymss/train_ascend/`（`freq_unet.py` 模型、`train.py` 裸循环、
  `align_check.py` 对齐、`evaluate.py` 协议、`npu_kernels_bandunet.py` triton 槽位、
  `selfcheck.py`、`bench_infer.py`、`sweep_params.py`、`download_musdb.py`、`prepare_dataset.py`）
- 产物：`train_ascend/results/*.json`（对齐表、吞吐、评测、kernel 状态原始数据）
- 本报告同步入库：`docs/ascend_band_unet_report.md`
- 运行产物（checkpoint / 日志）在 `/workspace/runs/`，未入库（`.gitignore`）

### 13.1 关键数字一览

| 维度 | 结果 |
|---|---|
| NPU | Ascend950PR ×1，HBM 128 GB，空闲基线 5281 MB |
| 环境 | py3.12.16 · torch 2.12.0+cpu · torch_npu 2.12.0.post3 · CANN 9.2.1 |
| 模型 | 20,874,036 参数（上限 21.63M ✅，参考 20,961,844，差 -0.42%） |
| 数据 | MUSDB18-HQ：train 100 / test 50 曲，29.04 GiB，check-musdb 通过 |
| 训练吞吐 (bs24) | fp32 3.345 s/step（17.9 steps/min）· bf16 2.737 s/step（21.8 steps/min） |
| 等效样本吞吐 | 7.2 / 8.8 samples/s vs G4 9.3 samples/s → **0.77x / 0.95x** |
| fp32 一致性 | 输出 rel RMS **9.34e-07**，逐层最差 1.51e-06（686 层） |
| bf16 一致性 | 输出 rel RMS 9.25e-03，逐层最差 1.37e-02 |
| 8 s 推理 | fp32 23.30 ms · **bf16 13.84 ms**（G4 11.691 ms → 1.18x） |
| Triton | 不可用（无 cp312 wheel）；kernel 已写、未验证；eager 回退 0 差异 |
| 评测协议 | 56 窗已冻结；100 步 raw 权重中位 SI-SDR -3.82 dB（仅管线证明） |

### 13.2 未解决 / 需要外部条件

1. triton-ascend 需要 cp311 解释器或华为出 cp312 wheel，否则第 8 节三个 kernel 只能停留在"未验证"。
2. 62 段 MaskEstimator 的小 GEMM 是主要瓶颈（预计 1.5–2x 可挖）。
3. 多分辨率 STFT 损失目前在 NPU 上走 istft+重 STFT，可改频域直接计算。
4. 只有 100 步，无质量结论；下一阶段应 ≥10k 步并同时评 raw/EMA。

---

# 第二轮：镜像源修正 + Triton 打通 + 全量重测

## 14. 镜像源修正（上一轮的 Triton 结论是错的）

上一轮判定"triton-ascend 无 cp312 wheel"是**用错了索引**：我只查了通用 PyPI（最高 3.2.0，确实没有 cp312）。
华为**昇腾专用索引**里有 cp312：

```
https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/
  triton_ascend-3.6.0.post2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl   ✅
```

安装命令（只用昇腾索引，不走通用 PyPI 的 3.2.0）：

```bash
python -m pip install "triton-ascend==3.6.0.post2" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 15. 版本矩阵对齐

triton-ascend 3.6 的兼容矩阵是 CANN 9.2.0 + TorchNPU 2.10.0.post7 或 **2.13.0**；
上一轮的 torch_npu 2.12.0.post3 不在矩阵内，因此整体换到 2.13.0：

| 组件 | 上一轮 | **本轮** | 说明 |
|---|---|---|---|
| torch | 2.12.0+cpu | **2.13.0+cpu** | 仍从 `download.pytorch.org/whl/cpu` 取，**绝不装 CUDA 版**（dlopen abort 已复现过） |
| torch_npu | 2.12.0.post3 | **2.13.0** | 通用索引 `mirrors.huaweicloud.com/repository/pypi/simple` |
| triton | 无 | **3.6.0** | triton-ascend 依赖里指定 |
| triton-ascend | 安装失败 | **3.6.0.post2** | 昇腾专用索引 |
| numpy | 2.5.4 | **1.26.4** | triton-ascend 硬 pin `numpy==1.26.4`，随包降级 |
| scipy | 1.18.1 | **1.13.1** | 同上，`python_version < 3.13` 分支 pin |
| CANN / driver | 9.2.1 / 25.7.rc1.10 | 同 | 未动 |

`npu-smi info`（重测时）：`Ascend950PR · OK · HBM 5813 / 131072 MB · 无用户进程`；
`torch.npu.is_available()=True`、`device_count()=1`、`get_device_name(0)=Ascend950PR_9579`。

**升级后无回归**：`selfcheck.py` 的 NPU 8 s 前向+反向、梯度完整性、数值对齐结果与 2.12 完全一致（见 16 节，数字逐位相同）。

## 16. Triton 可用性证据

```
torch      2.13.0+cpu
torch_npu  2.13.0
triton     3.6.0
triton-ascend 3.6.0.post2
npu avail  True  count 1  Ascend950PR_9579
triton target GPUTarget(backend='npu', arch='Ascend950PR_9579', warp_size=0)
```

`train_ascend/triton_check.py` 跑了官方 vector-add 教程 kernel 与一条 `x*y+z` 融合链：

| kernel | dtype | 编译 | 首次调用 | 稳态 | eager | 与 eager 差异 |
|---|---|---|---|---|---|---|
| vector_add | fp32 | ✅ | 830.49 ms | 0.0259 ms | 0.0191 ms | max_abs 0.0（逐位相同） |
| vector_add | fp16 | ✅ | 680.48 ms | 0.0246 ms | 0.0188 ms | max_abs 0.0（逐位相同） |
| fma_chain | fp16 | ✅ | — | — | — | max_abs 3.902e-03 / rel_rms 2.071e-04 |

结论：**triton-ascend 在 950PR 上能真正编译并在 NPU 执行**，首次调用 ~0.7-0.8 s 是编译开销。
纯 elementwise 拷贝/加法仍略慢于 aclnn（0.0259 vs 0.0191 ms，1.36x）——与上游"aclnn 元素级已接近 HBM 带宽上限"的判断一致。

## 17. 三个融合 kernel 结果（fp16，实测）

| kernel | 形状 | 编译 | eager | fused | **加速比** | max_abs | rel_rms |
|---|---|---|---|---|---|---|---|
| bandsplit_norm_scale | rows=3968, n_feat=16 | ✅ | 0.04481 ms | 0.03142 ms | **1.426x** | 7.812e-03 | 4.120e-04 |
| bandsplit_norm_scale | rows=3968, n_feat=32 | ✅ | 0.04518 ms | 0.0308 ms | **1.467x** | 7.812e-03 | 4.574e-04 |
| bandsplit_norm_scale | rows=3968, n_feat=60 | ✅ | 0.04596 ms | 0.03054 ms | **1.505x** | 7.812e-03 | 4.603e-04 |
| fused_rope | rows=5520, D=64 | ✅ | 0.0608 ms | 0.03299 ms | **1.843x** | 9.766e-04 | 3.588e-04 |
| fused_gated_out | [2, 2, 690, 64] | ✅ | 0.02625 ms | 0.03999 ms | **0.656x（更慢）** | 1.953e-03 | 3.520e-04 |

要点：
- **band-split 归一化链 1.43–1.51x**（把 `normalize → *scale → *gamma` 三次访存合成一次读写），fp16 误差是单次舍入（max_abs 恰为 1/128 = 0.0078，rel_rms 4.6e-4）。
- **RoPE 1.84x**（交织读、`[even|odd]` 连续写，一次读写完成 q/k 旋转）。
- **门控输出 0.66x——慢了**。该形状只有 176k 元素，eager 的 `sigmoid+mul` 已是一条 aclnn kernel、接近带宽上限；2-D tile 版省下的广播materialize 抵不过 kernel 启动与 tile 开销。第一版 1-D 写法（`BLOCK=16384` + 每元素整除）还直接 **UB overflow 编译失败**（需要 2621440 bits，UB 只有 1769472），改成 2-D tile 才能编译。这一条保持 eager。

## 18. 换代后全量重测（torch_npu 2.13.0 + triton-ascend 3.6.0.post2）

### 18.1 默认值复核

`torch.npu.conv.allow_hf32` **仍为 True**，`torch.npu.matmul.allow_hf32` 仍为 False —— 与 2.12 一致，
所以 fp32 路径**依旧必须显式关掉 HF32**，否则只有 ~4e-4。

### 18.2 fp32 逐层对齐（NPU vs CPU，686 层）

| 指标 | 数值 |
|---|---|
| 输出 max abs | **8.941e-07** |
| 输出 rel RMS | **9.336e-07** |
| 逐层最差 rel RMS | 1.506e-06 |
| 逐层最差 max abs | 6.676e-06 |
| 逐层中位 rel RMS | 7.569e-07 |

| 层 | max abs | rel RMS |
|---|---|---|
| `dec16.p2` | 8.941e-07 | 1.506e-06 |
| `enc16.p2` | 8.866e-07 | 1.452e-06 |
| `dec32.p2` | 8.345e-07 | 1.395e-06 |
| `blocks.0.freq.attn.to_gates` | 1.729e-06 | 1.204e-06 |
| `enc32.p2` | 8.345e-07 | 1.151e-06 |
| `blocks.0.msc.pw` | 3.099e-06 | 1.105e-06 |

**仍在 1e-6 量级（9.34e-07），与 torch_npu 2.12 逐位相同** → 换代没有引入数值回归。

### 18.3 bf16 对齐（NPU bf16 vs CPU fp32）

| 指标 | 数值 |
|---|---|
| 输出 max abs | 6.962e-03 |
| 输出 rel RMS | **9.247e-03** |
| 逐层最差 rel RMS | 1.372e-02 |
| 逐层最差 max abs | 6.021e-02 |
| 逐层中位 rel RMS | 8.595e-03 |

| 层 | max abs | rel RMS |
|---|---|---|
| `dec16.p2` | 6.995e-03 | 1.372e-02 |
| `dec32.p2` | 6.947e-03 | 1.286e-02 |
| `dec62.p2` | 6.971e-03 | 1.152e-02 |
| `enc16.p2` | 6.473e-03 | 1.128e-02 |
| `_estimate_masks.to_freqs.0.1` | 2.727e-03 | 1.074e-02 |
| `_estimate_masks.to_freqs.0` | 2.727e-03 | 1.074e-02 |

误差上界不变：输出 ≤9.3e-3、单层 ≤1.4e-2；最大误差仍在 `*.p2`（SwiGLU 输出)与注意力投影。

### 18.4 100 步吞吐重测（sb=24，真实 MUSDB）

| precision | 本轮 median step | steps/min | samples/s | 上一轮 | 变化 |
|---|---|---|---|---|---|
| 32-true | 3.3646 s | 17.8 | 7.1 | 17.9 steps/min | **-0.6%（噪声内）** |
| bf16 | 2.7367 s | 21.8 | 8.8 | 21.8 steps/min | **持平** |

**结论：换代（torch_npu 2.12→2.13 + triton-ascend）没有改变训练吞吐。**
（numpy 1.26.4 / scipy 1.13.1 降级也未影响训练。）

### 18.5 8 秒推理延迟与瓶颈定位（bf16 / fp32）

Stage 分解（NPU event 计时，bf16，device total 13.063 ms）：

| stage | ms | 占比 |
|---|---|---|
| band_split | 3.0903 | 23.7% |
| _estimate_masks | 2.7085 | 20.7% |
| enc32 | 0.7897 | 6.0% |
| enc62 | 0.7621 | 5.8% |
| dec62 | 0.7519 | 5.8% |
| up32 | 0.6461 | 4.9% |
| dec32 | 0.6198 | 4.7% |
| enc16 | 0.4021 | 3.1% |
| up16 | 0.3499 | 2.7% |
| dec16 | 0.3323 | 2.5% |
| up8 | 0.3242 | 2.5% |
| down62 | 0.1037 | 0.8% |
| final_norm | 0.0856 | 0.7% |
| down32 | 0.0827 | 0.6% |
| down16 | 0.0423 | 0.3% |
`blocks`（4 层 MSC+注意力）因挂在 ModuleList 上没有 forward hook，计入 `unaccounted 1.972 ms`（约 15%）。

**实测占比：`band_split` 3.0903 ms（23.7%）才是第一瓶颈，
MaskEstimator 2.7085 ms（20.7%）第二** —— 两者合计约 44%，
共同点都是"62 段小算子"。上一轮把 MaskEstimator 当唯一下瓶颈的说法需要修正为"62 段算子链"。

#### 关键发现：`grouped_forward` 在 NPU 上是负优化

`pymss_core` 为 CUDA 写的"把 62 段合成一次 grouped bmm"在昇腾上反而更慢（单算子实测）：

| 模块 | grouped bmm | 逐段 Linear | 加速比 |
|---|---|---|---|
| BandSplit | 3.947 ms | **2.709 ms** | 1.46x |
| MaskEstimator | 2.996 ms | **2.041 ms** | 1.47x |

整模型（3 次独立重复取中位，进程内波动 <0.5%）：

| precision | A 基线(grouped) | B 逐段 | C 逐段 + triton RoPE | 总加速 | vs G4 11.691 |
|---|---|---|---|---|---|
| fp32 | 24.695 ms | **18.993 ms** | 19.489 ms | 1.267x | **1.62x**（最好） |
| bf16 | 13.938 ms | 13.556 ms | **13.505 ms** | 1.032x | **1.16x** |

- 关掉 grouped forward 纯属"算子选择"改动，数学等价（grouped bmm 与逐段 Linear 只差累加顺序）。
- RoPE 融合在 bf16 上再省 0.05–0.11 ms（<1%），fp32 上反而略慢 → 只在 bf16 路径启用。
- 进程间方差约 5-8%（同一进程内 3 次重复 <0.5%），所以只看同批次相对差值。

### 18.6 剩余瓶颈（按可挖空间排序）

1. **band_split（23.7%）+ mask head（20.7%）= 62 段小算子链**：单段 GEMM 的 M 维极小（每段 4-16 个频点），
   cube 核吃不满。下一步应当做**单次融合 kernel 完成 62 段 normalize+GEMM**（我的 triton band-split kernel
   目前只融合了 normalize+scale，1.45x；把 GEMM 也 tl.dot 进去是下一步），或改成分组按 16 的整数倍打包。
2. **4 层 MSC + 注意力（~15%）**：MSC 的 4 个 dilation 是 4 条 depthwise conv，可融合为一条 kernel。
3. **多分辨率 STFT 损失（训练侧）**：训练 step 里 istft+重 STFT 仍占约一半时间（合成分块 0.99 s vs 实测 2.74 s/step）。
4. **triton 元素级 kernel 在当前形状下大多不划算**：门控输出 0.66x、vector-add 1.36x 慢于 aclnn，
   只有"省一次全尺寸读写"的链条（normalize、RoPE 交织重排）才有 1.4–2.0x。
5. bf16 相对 fp32 只有 1.2x 训练提速、推理 1.4x，说明瓶颈是算子粒度/发射而非算力。

---

# 第三轮：训练 step 的 phase profile（纯测量，不改配方）

工具：`train_ascend/profile_train.py`，真实 MUSDB 数据、bs24、100 步（前 3 步丢弃作 warmup），
**每段用 `torch.npu.synchronize()` 边界单独计时**。落盘：
`results/profile_train_bf16.json`、`results/profile_train_fp32.json`。

## 19. 总计与对账

| precision | 每步 wall（sync 免测） | 分段求和 | 对账误差 |
|---|---|---|---|
| bf16 | **2732.24 ms** | 265045.69 ms | **-0.0%** |
| fp32 | **3350.82 ms** | 325282.02 ms | **-0.0%** |

对账误差 ≈ 0（分段计时器自身的 16 次 sync 开销已含在段内），**没有漏计**。
dataloader 配置：`num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=False`。

## 20. bf16 分段明细（bs24，中位 2732.24 ms）

| 段 | sum_ms | mean_ms | median_ms | share |
|---|---|---|---|---|
| `dataloader_wait` | 87.45 | 0.9 | 0.91 | 0.03% |
| `h2d_copy` | 7550.91 | 77.84 | 77.59 | 2.85% |
| `forward` | 19679.18 | 202.88 | 202.85 | 7.42% |
| `prep_complex` | 115.97 | 1.2 | 1.19 | 0.04% |
| `loss_complex` | 129.5 | 1.34 | 1.33 | 0.05% |
| `loss_mrstft_512` | 941.68 | 9.71 | 9.71 | 0.36% |
| `loss_mrstft_1024` | 1553.62 | 16.02 | 16.01 | 0.59% |
| `loss_mrstft_2048` | 2722.78 | 28.07 | 28.05 | 1.03% |
| `istft_pred` | 8030.96 | 82.79 | 82.77 | 3.03% |
| `istft_tgt` | 8033.18 | 82.82 | 82.84 | 3.03% |
| `loss_sisdr` | 78.54 | 0.81 | 0.81 | 0.03% |
| `loss_leak` | 92.44 | 0.95 | 0.93 | 0.03% |
| `backward` | 212860.5 | 2194.44 | 2194.74 | 80.31% |
| `grad_clip` | 746.41 | 7.69 | 7.56 | 0.28% |
| `grad_norm_to_host` | 6.29 | 0.06 | 0.06 | 0.0% |
| `optimizer_step` | 1654.46 | 17.06 | 14.78 | 0.62% |
| `ema_update` | 761.82 | 7.85 | 7.79 | 0.29% |
| **total** | **265047.18** | **2732.45** | **2732.24** | **100%** |

fp32 同样结构：`backward` 2623.58 ms（78.25%）、
`forward` 385.54 ms、`istft_pred`+`istft_tgt` 166.20999999999998 ms。

**注意：`backward` 这一格包含了损失图的反向**，所以它一个数就吃掉 80%。下面的对照实验把它拆开。

## 21. 把 backward 拆开（关键）

| 配置 | step ms | backward ms | 说明 |
|---|---|---|---|
| `mask_scalar`（只有模型 fwd+bwd，无损失） | **871.66** | — | 模型 fwd+bwd ≈ 763 ms，其中**反向 ≈ 560 ms** |
| `spec_only`（模型 + 复频谱 L1） | 1263.6 | 784.61 | 复频谱 L1 反向 ≈ 225 ms（784.6 − 560） |
| 全量减 MR-STFT（`no_mrstft`） | 1340.3 | 866.01 | istft+SI-SDR+leak 反向 ≈ 469 ms |
| **全量（默认配方）** | **2732.24** | 2194.74 | — |

归因（bf16，bs24）：

| 组成 | 时间 | 占 step |
|---|---|---|
| **多分辨率 STFT 损失**（前向 54 + 反向 ~1338） | **1392 ms** | **50.9%** |
| ISTFT×2 + SI-SDR + leakage（前向 167 + 反向 ~304） | 469 ms | 17.2% |
| 模型 forward + backward | 763 ms | 27.9% |
| H2D 拷贝（543 MB/step） | 78 ms | 2.8% |
| clip + AdamW + EMA | 30 ms | 1.1% |
| dataloader 等待 | 0.91 ms | ~0% |

**损失相关合计 1861 ms = 68.1%，模型 fwd+bwd 只有 27.9%，数据 0.03%。**

## 22. 每个分辨率单独计时

| n_fft | 动作 | step ms | 相对基线 | MR-STFT 前向合计 |
|---|---|---|---|---|
| 256 | 加入 | 3164.23 | +432.0 ms | 1228.13 ms |
| 512 | 移除 | 2280.98 | -451.3 ms | 886.38 ms |
| 1024 | 移除 | 2247.86 | -484.4 ms | 755.06 ms |
| 2048 | 移除 | 2250.68 | -481.6 ms | 514.11 ms |
| 4096 | 加入 | 3237.0 | +504.8 ms | 2111.12 ms |

单个分辨率的前向只要 **9.7–28.1 ms**，
但它让每步多花 **430–505 ms** → **约 95% 的成本在反向**。
微基准（48×352800 单次 STFT 前向+反向）：512→505.6 ms、
1024→520.7 ms、2048→530.4 ms。
**一次 `torch.stft` 的反向约 480 ms**，这是整个 step 里最贵的单个算子。

## 23. hop 与 worker 对照（仅测速）

| 配置 | step ms | MR-STFT 前向 ms | 备注 |
|---|---|---|---|
| 默认（hop = n_fft//4） | 2732.24 | 53.8 | 参考 |
| 全部固定 hop=147 | 4463.47 | 2618.6 | 帧数 ×4，**慢了 63%**；损失数值改变 |
| 全部固定 hop=512 | 2141.26 | 781.0 | 更快但重叠度变了，损失数值改变 |

| num_workers | step ms | dataloader_wait | 与 4 相比 |
|---|---|---|---|
| 0 | 2878.44 | 136.99 ms | +150.9 ms |
| 4 | 2727.52 | 1.01 ms | +0.0 ms |
| 8 | 2719.34 | 0.87 ms | -8.2 ms |
| 16 | 2722.85 | 0.13 ms | -4.7 ms |

数据根本不是瓶颈：`num_workers=4` 已把等待压到 1.01 ms，8/16 worker 只再省 8/5 ms。

## 24. FFT 与 I/O 实测

hop 用 512（默认），8 s = 352800 点，实测帧数 **690**（不是 2400 —— 2400 帧对应 hop 147，那是慢 41% 的配置）：

| 分辨率 | hop | 帧数 | 频点 | 每信号元素 | 每步 FFT 数 |
|---|---|---|---|---|---|
| 512 | 128 | 2757 | 257 | 708,549 | 264,672 |
| 1024 | 256 | 1379 | 513 | 707,427 | 132,384 |
| 2048 | 512 | 690 | 1025 | 707,250 | 66,240 |
| 合计 | | | | **~68 M 元素/方向** | **463,296 次实 FFT** |

外加 ISTFT：每步 **66,240 次实 iFFT**（690 帧 × 2 声道 × 24 batch × 2 路）。

I/O（实测，`worker_costs` 字段）：
- 每个样本解码 **5.64 MB**（mixture+vocals，float32 2×2×352800）→ 每步 **135.5 MB** 逻辑读；命中 page cache，物理读 ≈ 0。
- 每样本解码 **1.39 ms** + STFT **1.34 ms**，**都在 DataLoader 工作进程里做**（`__getitem__` 内 wave 解码 + `torch.stft`）。
- H2D 每步搬 **543 MB**（两份 STFT 谱 float32），实测 78 ms ≈ 7 GB/s。

`clip_grad_norm_` 的设备同步次数：**0**（patch 了 `Tensor.item` 与 `torch.npu.synchronize` 计数）。
真正的同步发生在记录时 `float(grad_norm)`：0.06 ms（此时设备已同步，实际免费）。

## 25. 三个候选实现改动的实测（bf16，23 步）

| 改动 | step ms | 相对基线 | 反向 ms | 数学是否等价 |
|---|---|---|---|---|
| 基线 | 2732.24 | — | 2194.74 | — |
| 损失改用 `pymss_core.npu_fft.stft_npu` | 2098.4 | **-633.8** | 1570.74 | 是，max_abs ≤2.91e-04 / rel_rms ≤8.3e-07 |
| STFT 走 real-view 取幅度 `sqrt(re²+im²+1e-12)` | 2083.55 | **-648.7** | 1528.52 | 是（1e-12 兜底仅防 sqrt(0) 梯度 inf） |
| 损失整体放进 bf16 autocast | 2723.0 | -9.2 | 2193.35 | 否（数值变），且**无收益** |

`realview` 初版（不带 eps）出现 **loss=NaN**：`sqrt(0)` 的梯度是无穷，和之前 `leakage_loss` 同一个坑；加 1e-12 后 loss 恢复
4.650（同批 bf16 运行区间 4.55–4.68）。

## 26. 结论

**(a) 训练慢的主因是"损失"，不是数据、也不主要是模型反向。**
bf16 2732.24 ms 里：损失图合计 **1861 ms（68%）**
（其中多分辨率 STFT 1392 ms）；模型 fwd+bwd **763 ms（28%）**；
H2D 78 ms（3%）；dataloader **0.91 ms（0.03%）**。
根源是**昇腾上没有原生复数 STFT，`torch.stft` 的反向一次约 480 ms**（微基准实测），而损失每步要做 3 次。

**(b) 只改一个地方，收益最大的是：把损失的 `torch.stft` 换掉。**
- `stft_npu`：**2732.24 → 2098.4 ms**（-23%）
- real-view 幅度：**2732.24 → 2083.55 ms**（-24%）
实测量级：**预计 ~2.08–2.10 s/step（bf16）、fp32 同比例约 2.5 s/step**。两者数学等价（rel_rms ≤8.3e-07），
属于纯实现优化；但升更换了算子路径，必须重跑一次 bf16 对齐表再上训练。
若允许改配方，最省事的单点是**去掉一个分辨率**（−450…−484 ms），代价是与 G4 的损失定义不再等价。

**(c) 配方 vs 实现**
- **会改训练配方（影响与 G4 等价性，不能默认采用）**：分辨率增减、hop 变更、把 MR-STFT 改为直接在模型复数谱上算（免 istft→stft 往返）、损失放进 autocast。
- **纯实现优化（数学等价，可默认采用）**：`stft_npu` 后端、real-view 幅度（需 1e-12 兜底）、`num_workers` 4→8（−8 ms）、H2D 用 pinned memory（未测，预计可省 39 ms 量级）。

---

# 第四轮：950PR 峰值标定（算力 / 带宽 / 启动开销）

工具：`train_ascend/calibrate_peak.py`，全部数据在 `train_ascend/results/peak_calib.json`。
设备：`Ascend950PR_9579`，HBM 126112 MB 可见，torch 2.13.0+cpu / torch_npu 2.13.0。
标称（华为 950 系列规格表，按本机 28 核档）：BF16/FP16 **425 TFLOPS**（32 核档 486）、TF32 212 TFLOPS、HBM **1.6 TB/s**。

## 27. GEMM 峰值

每个形状 warmup 10 + 计时 30（超大形状 2/5 次），`torch.npu.synchronize()` 边界，取中位数，
GFLOPS = 2·M·K·N / t。

### 27.1 bf16（13 个形状）

| 形状 | GFLOP | ms | TFLOPS |
|---|---|---|---|
| `(512,512)@(512,512)` | 0.3 | 0.019 | **14.13** |
| `(1024,1024)@(1024,1024)` | 2.1 | 0.025 | **85.9** |
| `(2048,2048)@(2048,2048)` | 17.2 | 0.0678 | **253.39** |
| `(4096,4096)@(4096,4096)` | 137.4 | 0.3878 | **354.41** |
| `(8192,8192)@(8192,8192)` | 1099.5 | 2.9681 | **370.44** |
| `(16384,16384)@(16384,16384)` | 8796.1 | 23.3007 | **377.5** **← 峰值** |
| `(4096,4096)@(4096,1024)` | 34.4 | 0.1174 | **292.67** |
| `(1024,4096)@(4096,4096)` | 34.4 | 0.1146 | **299.82** |
| `(8192,8192)@(8192,2048)` | 274.9 | 0.7934 | **346.46** |
| `(2048,8192)@(8192,8192)` | 274.9 | 0.7986 | **344.2** |
| `(16384,4096)@(4096,4096)` | 549.8 | 1.493 | **368.22** |
| `(4096,16384)@(16384,4096)` | 549.8 | 1.4806 | **371.31** |
| `(16384,8192)@(8192,8192)` | 2199.0 | 6.2206 | **353.51** |

**实测 bf16 峰值 = 377.5 TFLOPS**，出现在 `(16384,16384)@(16384,16384)`。
= 标称 425 TFLOPS 的 **88.8%**（32 核档 486 的 77.7%）。

形状效率曲线（bf16）：512³ 只有 14.13 TFLOPS（3.7% 峰值）、
1024³ 85.9、2048³ 253.39、4096³ 354.41、8192³ 370.44，
**要到 4096 才接近峰值**；长方形状不掉太多（≥292.67 TFLOPS），说明 cube 核吃形状吃在 M/K/N 都小的时候。

### 27.2 fp32

| 形状 | GFLOP | ms | TFLOPS |
|---|---|---|---|
| `(512,512)@(512,512)` | 0.3 | 0.0311 | **8.63** |
| `(1024,1024)@(1024,1024)` | 2.1 | 0.1189 | **18.06** |
| `(2048,2048)@(2048,2048)` | 17.2 | 0.7648 | **22.46** |
| `(4096,4096)@(4096,4096)` | 137.4 | 5.8382 | **23.54** |
| `(8192,8192)@(8192,8192)` | 1099.5 | 46.6939 | **23.55** |
| `(16384,16384)@(16384,16384)` | 8796.1 | 372.1378 | **23.64** |
| `(4096,4096)@(4096,1024)` | 34.4 | 1.5068 | **22.8** |
| `(1024,4096)@(4096,4096)` | 34.4 | 1.508 | **22.78** |
| `(8192,8192)@(8192,2048)` | 274.9 | 11.6488 | **23.6** |
| `(2048,8192)@(8192,8192)` | 274.9 | 11.6498 | **23.6** |
| `(16384,4096)@(4096,4096)` | 549.8 | 23.3658 | **23.53** |
| `(4096,16384)@(16384,4096)` | 549.8 | 23.2746 | **23.62** |
| `(16384,8192)@(8192,8192)` | 2199.0 | 93.0133 | **23.64** **← 峰值** |

**实测 fp32 峰值 = 23.64 TFLOPS**（`(16384,8192)@(8192,8192)`）。
fp32 没有原生 cube 通路，只有 **bf16 的 1/16.0**；
相对 TF32 标称 212 TFLOPS 只有 11.2%（因为我们关掉了 HF32，走的是真 fp32）。
**这条数字解释了为什么 fp32 训练慢 2.4 倍**：不是调度问题，是 fp32 在这张卡上本就没有 tensor 通路。

### 27.3 fp16（仅供参考，不用于训练）

峰值 **370.59 TFLOPS**（`(8192,8192)@(8192,8192)`），与 bf16 基本一致（98.2%）。

## 28. 带宽标定

| op | 大小 | ms | 有效 GB/s |
|---|---|---|---|
| d2d copy_ (copy 1-pass) | 1.0 GiB | 1.4989 | 1432.7 |
| add_(1) read+write | 1.0 GiB | 1.5309 | 1402.8 |
| mul out= (read+write) | 1.0 GiB | 1.533 | 1400.8 |
| d2d copy_ (copy 1-pass) | 2.0 GiB | 2.9842 | 1439.2 |
| add_(1) read+write | 2.0 GiB | 3.007 | 1428.3 |
| mul out= (read+write) | 2.0 GiB | 2.993 | 1435.0 |

**实测 d2d 带宽 = 1439.2 GB/s = 1.439 TB/s**，= 标称 1.6 TB/s 的 **90.0%**。
拷贝、读-改-写、out= 三种写法的带宽基本一致（1399–1439 GB/s），说明已经贴在 HBM 上限上。

## 29. 逐元素 / 访存算子（真实激活形状 (690,62,192) = 8,213,760 元素）

| 算子 | dtype | ms | 有效 GB/s | 占实测带宽 | 单次启动占比 | 判定 |
|---|---|---|---|---|---|---|
| gelu(tanh) | bf16 | 0.0533 | 616.4 | 42.8% | 54.1 | ALU/延迟受限 |
| silu | bf16 | 0.0254 | 1293.5 | 89.9% | 54.1 | 贴带宽上限 |
| sigmoid | bf16 | 0.025 | 1314.2 | 91.3% | 54.1 | 贴带宽上限 |
| rmsnorm(dim=-1) | bf16 | 0.0527 | 623.4 | 43.3% | 54.1 | ALU/延迟受限 |
| add_(1) in-place | bf16 | 0.0227 | 1447.4 | 100.6% | 54.1 | 贴带宽上限 |
| gelu(tanh) | fp32 | 0.0624 | 1053.0 | 73.2% | — | ALU/延迟受限 |
| silu | fp32 | 0.0253 | 2597.2 | 180.5% | — | 贴带宽上限 |
| sigmoid | fp32 | 0.0263 | 2498.5 | 173.6% | — | 贴带宽上限 |
| rmsnorm(dim=-1) | fp32 | 0.0579 | 1134.9 | 78.9% | — | ALU/延迟受限 |
| add_(1) in-place | fp32 | 0.0245 | 2682.0 | 186.4% | — | 贴带宽上限 |

结论：
- **SiLU / sigmoid 已经接近带宽上限**（bf16 1293.5 / 1314.2 GB/s = 实测带宽的 89.9% / 91.3%），继续优化空间很小。
- **GELU(tanh) 与 RMSNorm 只有 616.4 / 623.4 GB/s（42.8% / 43.3%）**，是 ALU/延迟受限，不是带宽受限 —— 这两个还有 ~2.3x 的空间（换近似/查表或与相邻算子融合）。
- 注意：这个形状只有 16 MB(bf16)/33 MB(fp32)，部分常驻片上，所以有效 GB/s 可能**超过** HBM 实测（fp32 行超过 2.7 TB/s），
  因此"占实测带宽"是保守下界，不影响"贴顶 / 没贴顶"的判断。

## 30. 小算子启动开销

| 指标 | 值 |
|---|---|
| 1 元素算子（纯 dispatch+launch，1000 次平均） | **4.47 µs** |
| (690,62,192) elementwise 摊薄后单次（bf16 / fp32） | 8.27 / 9.91 µs |
| (690,62,192) 单次同步计时（bf16 / fp32） | 0.0222 / 0.0239 ms |

一个 8.2M 元素的 elementwise 操作本身只要 ~4 µs 的算，却要 **4.47 µs 的启动** —— 串流里约一半墙钟是启动。

### 30.1 band_split 的 62 段小 GEMM（真实形状）

| dtype | 62 次逐个调用 | 单次 | 其中纯启动 | packed bmm | 加速比 | 填充浪费 |
|---|---|---|---|---|---|---|
| bf16 | 0.2977 ms | 4.91 µs | **91.1%** | 0.0297 ms | **10.024x** | 1.996x |
| fp32 | 0.2988 ms | 4.88 µs | **91.6%** | 0.1197 ms | **2.496x** | 1.996x |

**单次小算子启动开销 = 4.47 µs。**
62 段逐调用的总时间 0.2977 ms 里，**0.027 ms 是纯启动（91.1%）**，
真正做数学只有 ~20.6 µs。
打包成一次 bmm 后 **10.024x**（代价是 1.996x 的填充浪费）。

模型里 BandSplit 两条路径实测（同一输入，含归一化）：grouped `1.8849 ms` vs 逐段 `2.6912 ms`。

## 31. 当前模型 vs 这把尺子

| 指标 | 值 |
|---|---|
| 参数 | 20,874,036 |
| 前向 FLOPs（`FlopCounterMode` 实测，bf16, 690 帧） | **187.48 GFLOP**（addmm 109.6 + conv 36.1 + 其余） |
| 前向延迟（bf16，30 次中位） | **13.789 ms** |
| 实测吞吐 | **13.6 TFLOPS** |
| 占实测 bf16 峰值 | **3.6%** |
| 占标称 425 TFLOPS | 3.2% |
| aten 调用数 / 设备 kernel 数（每次前向） | 5670 / **1910** |
| 平均每 kernel 墙钟 | 7.22 µs（启动开销 4.47 µs） |

用你给的 116 GFLOP / 8.45 ms = 13.7 TFLOPS 口径：占实测峰值 **3.63%**。
本机实测同一模型是 **187.48 GFLOP / 13.789 ms**（帧数、口径差异，我按实测报）。

**判定：不是算力受限，也不是带宽受限，是启动/粒度受限。**
- 算力上限给的时间 0.497 ms，实测 13.789 ms → 算力只用了峰值的 3.6%。
- 带宽：前向若按激活 10 GB 量级算也远低于 1.44 TB/s（每 kernel 平均 7.22 µs 里大部分是发射）。
- **1910 个设备 kernel × 4.47 µs = 8.538 ms = 实测 13.789 ms 的 62%** —— 这是硬地板。

## 32. 结论

**(a) 实测上限**

| 项目 | 实测 | 标称 | 占比 |
|---|---|---|---|
| bf16 GEMM 峰值 | **377.5 TFLOPS** | 425 TFLOPS | **88.8%** |
| fp32 GEMM 峰值 | **23.64 TFLOPS** | —（TF32 212） | 11.2% of TF32 |
| fp16 GEMM 峰值 | 370.59 TFLOPS | 425 TFLOPS | 87.2% |
| d2d 带宽 | **1439.2 GB/s** | 1.6 TB/s | **90.0%** |
| 单次 kernel 启动 | **4.47 µs** | — | — |

**(b) 利用率**：当前 13.6 TFLOPS（你给的口径 13.7）占实测峰值 **3.6% / 3.63%**。

**(c) 不改模型规模时的理论加速上限**
- 按算力约束：187.48 GFLOP / 377.5 TFLOPS = **0.497 ms** → **27.76x**（前提是 kernel 数不减也能满载，实际做不到）。
- 按启动约束：1910 kernel × 4.47 µs = **8.538 ms** → 只有 **1.62x**。
- **两个约束取大者定上限 → 现实上限约 1.62x，瓶颈是 kernel 数不是算力。**
  要突破就得把 kernel 数压下来（融合、大 kernel、packed 算子），这与第三轮"62 段小算子链"的结论完全一致。

**(d) 要做到 10x**
- 目标 1.38 ms。算力预算 = 1.38 ms × 377.5 TFLOPS = **520.5 GFLOP**，而模型只要 187.48 GFLOP
  → **算力上完全够，模型规模不需要砍**（只需跑到 36.0% 峰值）。
- 真正要砍的是 kernel 数：1.38 ms / 4.47 µs = **最多 308 个 kernel**，当前 1910 个 → **要减 6.2x**。
- 如果坚持不融合、只靠缩模型（kernel 数按参数线性缩）：参数量要从 20,874,036 砍到 **~3,371,551（约 1/6.2）**，
  也就是把 20.9M 砍到 3.4M 量级 —— 那已经不是同一个模型了。**所以 10x 的路径是融合，不是缩小。**

**给后续优化的分母**：bf16 算是 377.5 TFLOPS、带宽 1.44 TB/s、启动 4.47 µs。
任何优化提案都应该回答"它减少了多少 kernel 数 / 多少 ALU 受限算子"，而不是"它提高了多少 TFLOPS"。

---

# 第五轮：训练提速攻坚（目标 390–450 ms，实测未达成）

工具：`train_ascend/fast_loss.py`（自研 STFT autograd）、`graph_capture.py`、`train.py --fast-loss`。
数据：`train_ascend/results/speedup_final.json`、`graph_capture.json`。

## 33. 结果先说

| 指标 | 改前 | 改后 | 加速 | 目标 | 差距 |
|---|---|---|---|---|---|
| bf16 bs24 | 2736.7 ms | **1992.5 ms** | **1.37x** | 390–450 ms | 差 1542.5 ms |
| fp32 bs24 | 3364.6 ms | **2600.1 ms** | **1.29x** | — | — |

**6–7x 没达到。** 真正落地且数学等价的只有一项：自研 STFT autograd（打掉 aten::stft 反向），
它把 backward 从 2194.7 ms 压到 1443.8 ms（−751 ms）。图形捕获在本模型上抓不住（见 35 节），
分组打包虽测出 10x 但没有默认启用（理由见 36 节）。

## 34. 自定义 STFT autograd（唯一落地项，全部数学等价）

`fast_loss.py`：前向 `torch.stft` + 由 real/imag 求模；反向**不调用** aten::stft 的 backward，而是
①解析式 `d|z|/dz = g·z/|z|`（与 autograd 对比 6.4e-17）→ ②rfft 的伴随（用 PyTorch 自己的 rfft backward，
一次 FFT）→ ③乘窗 → ④`F.fold` 做 overlap-add → ⑤reflect-pad 的伴随（pad 自己的 backward）。

（`unfold` 的 backward 在昇腾未实现，所以分帧用 `as_strided` 视图 + 自写 backward。）

等价性（bs48、8 s、随机权重做 JVP 对比）：

| n_fft | 前向 rel_rms | 梯度 rel_rms | 参考实现耗时 | 本实现耗时 | 加速 |
|---|---|---|---|---|---|
| 512 | 3.29e-08 | 2.23e-07 | 438.4 ms | 182.8 ms | **2.4x** |
| 1024 | 3.29e-08 | 1.31e-07 | 459.5 ms | 211.8 ms | **2.17x** |
| 2048 | 3.29e-08 | 6.92e-07 | 450.1 ms | 210.4 ms | **2.14x** |

前向 3.3e-08、梯度 ≤6.9e-07，**都满足 ≤1e-6**。整步收益：`backward` 2194.74 → 1443.79 ms。

途中踩到的两个昇腾坑（都写进代码注释）：`sqrt(0)` 的梯度是 inf → NaN；直接对**大张量加 Python float**
会触发 vector core 异常（MTE invalid GM），必须用 0 维张量当 eps。

## 35. NPU Graph 捕获

| 项目 | 结果 |
|---|---|
| API | `torch.npu.NPUGraph` + `torch.npu.graph(...)`（**没有** `torch.npu.CUDAGraph`） |
| 简单 MLP（Linear+GELU） | **可捕获**，replay 0.023 ms，与 eager 逐位相同 |
| `band_split` / `enc62` 单独 | 可捕获 |
| **整模型前向** | **失败**：`endCaptureErr 107025`（cross-stream），`global`/`relaxed` 两种模式均失败 |
| 整训练步 | 失败（同一错误） |

**但这不改变结论**：按第四轮标定，bs24 每步只有 1910 个 kernel × 4.47 µs = **8.54 ms 的启动地板**，
占 1987 ms 的 **0.4%**。也就是说**图捕获在 bs24 上最多省不到 1%**，它不是 6x 的那根杠杆
（它在 bs1 推理 13.79 ms 的场景里才重要，那里启动占 62%）。所以即便抓图成功也到不了目标。

## 36. 分组打包 / 融合

- 62 段 band-split 的打包实测：逐个 0.2977 ms → 一次 padded bmm 0.0297 ms（**10.0x**），但填充浪费 2.0x。
- **没有默认启用**：在 bs24 下整段 band_split GEMM 只占 202 ms 前向里的一小部分，而填充会让有效 FLOPs 翻倍；
  真正吃时间的是 62 段小 GEMM 的**碎片化**与 depthwise conv，需要单 kernel 融合（未完成）。
- mask_head 同构，同样未启用。
- LocalBandBlock / MSC 的 Triton 融合**未实现**（时间不够），这是到 450 ms 的主要缺口。

## 37. 13 段复测（bf16，bs24，含 fast-loss）

对账：分段求和 / wall 误差 **-0.0%**，每步 wall **1987.38 ms**。

| 段 | 改前 ms | 改后 ms | Δ | 占比 |
|---|---|---|---|---|
| `total` | 2732.2 | 1987.4 | -744.9 | 100.0% |
| `backward` | 2194.7 | 1443.8 | -750.9 | 72.67% |
| `forward` | 202.8 | 202.9 | +0.0 | 10.21% |
| `istft_tgt` | 82.8 | 82.9 | +0.1 | 4.17% |
| `istft_pred` | 82.8 | 82.9 | +0.1 | 4.17% |
| `h2d_copy` | 77.6 | 79.2 | +1.6 | 3.99% |
| `loss_mrstft_2048` | 28.1 | 29.9 | +1.9 | 1.51% |
| `loss_mrstft_1024` | 16.0 | 17.9 | +1.8 | 0.9% |
| `optimizer_step` | 14.8 | 14.6 | -0.2 | 0.71% |
| `loss_mrstft_512` | 9.7 | 11.6 | +1.9 | 0.58% |
| `ema_update` | 7.8 | 8.1 | +0.3 | 0.4% |
| `grad_clip` | 7.6 | 7.7 | +0.1 | 0.4% |
| `loss_complex` | 1.3 | 1.3 | +0.0 | 0.07% |
| `prep_complex` | 1.2 | 1.2 | +0.0 | 0.06% |
| `loss_leak` | 0.9 | 1.0 | +0.1 | 0.05% |
| `dataloader_wait` | 0.9 | 0.9 | -0.0 | 0.04% |
| `loss_sisdr` | 0.8 | 0.8 | +0.0 | 0.04% |
| `grad_norm_to_host` | 0.1 | 0.1 | +0.0 | 0.0% |

被压缩的只有 `backward`（−751 ms）；其余段完全不变。`loss_mrstft_*` 前向略涨（+5.6 ms，多算了一次模）。

## 38. 卡在哪 / 地板在哪 / 6x 还差什么

| 段 | 现在 | 实测地板 | 为什么不能再降（本轮） |
|---|---|---|---|
| `backward` | 1443.79 ms | ≈ 560（模型）+ 300（istft/SI-SDR）+ ~120（MR-STFT 伴随） | 模型反向 560 ms 需要 kernel 融合；istft/SI-SDR 的 300 ms 需要把 istft 也做成自研伴随；MR-STFT 仍有 120 ms 是因为我的伴随里**重算了一次 torch.stft**（把谱存下来可再省 ~100–150 ms） |
| `forward` | 202.9 ms | 4500 GFLOP / 377 TFLOPS ≈ 12 ms（算力），实际 22 TFLOPS = 5.9% 峰值 | 62 段小 GEMM + depthwise conv 的碎片化，需融合 |
| `istft_pred`+`istft_tgt` | 165.8 ms | ~83 ms/次（vendor FFT 单次即 83 ms） | istft 是 vendor 路径，要更快只能自己写伴随 |
| `h2d_copy` | 79.2 ms | ~40 ms（pinned memory 未接） | 543 MB/步，现在是 pageable |
| 其余 | ≈ 46 ms | — | — |

**6x 还差 1536.0 ms**（目标 456 ms，现状 1992 ms）。
缺口唯一主要来源：**模型 fwd+bwd（763 ms）与 istft/SI-SDR 图（469 ms）这两块我本轮没有动**，
只打了 STFT 损失。要进 450 ms，模型必须做到 ~150 ms（需要 62 段打包 + depthwise/attention 融合），
istft 路径必须做到 ~50 ms（需要自研 istft 伴随）。

## 39. 哪些能默认采用

- **纯实现、数学等价、可默认采用**：`--fast-loss`（前向 3.3e-8、梯度 6.9e-7）。
- **纯实现但未启用**：packed bmm（有效 FLOPs 翻倍，需配合真正的变宽融合 kernel）、graph 捕获（抓不住）。
- **会改配方、不可默认采用**：减少 MR-STFT 分辨率、改 hop、把 MR-STFT 直接算在模型谱上（免 istft→stft）、损失进 bf16 autocast。
- 未使用 fp16 训练；模型结构、损失定义、数据配方均未改动（`align_check` 的 9.34e-07 / 9.25e-03 仍然成立）。
