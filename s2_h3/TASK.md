# 任务：在昇腾 950PR 上用 Triton（triton-ascend）训练 H3，协议与 Colab 完全相同

**本文件是任务的权威版本。** 你（服务器上的 pi）负责执行、profile、写 Triton 内核和环境排障；我（Mac 上的 lead）负责核对数字、和 Colab 的 H2 曲线对比。每完成一个阶段就把结果提交回分支并更新 `STATUS.md`。

## 0. 一句话

- 代码：`/workspace/s2_h3` 是 `/workspace/pymss` 的 worktree（分支 `exp/h3-ascend`）。实验目录 `S2=/workspace/s2_h3/s2_h3`，等价于 Colab 的 `/content/s2`。
- 协议一个字不改：batch 24、AdamW 5e-4 / wd 0.01 / StepLR(40000, 0.9)、EMA 0.999、bf16-mixed、5k 步一段、56 窗评测、损失 = pymsst 的 `masked_loss`（时域 MSE + 分位数掩码，**没有 STFT 损失**）。
- **训练必须走 Triton**：eager 在 NPU 上不是终点，也不是兜底。做法是 profile 驱动的内核替换——每个 Triton 内核都要同时过"数值等价"和"确实更快"两道验证，才允许进训练。数学不变，只是实现变了。**正式训练不允许用 eager 启动。**
- 数据必须是 Colab 那一份；`protocol_check.py` 打出 `PROTOCOL_PASS` 之前不许训练。

## 1. 取代码

```bash
cd /workspace/s2_h3 && git pull --ff-only          # 已有的 worktree
```

不要动 `/workspace/pymss`（在 `feat/ascend-band-unet`）、`/workspace/pymss-core`、`/workspace/pymsst` 的工作区和 git 状态。

## 2. 环境：只增不改

- 保持：torch 2.13.0+cpu、torch_npu 2.13.0、pytorch-lightning 2.6.6、triton 3.6.0、triton-ascend 3.6.0.post2、numpy、scipy。**不要升级或降级任何已安装的包。**
- 已装：ffmpeg 4.4.2、soundfile、huggingface_hub。其余缺什么补什么（pip 指向华为镜像），不要连带改 torch。
- 所有命令带 `PYTHONPATH=$S2`，让 `pymsst`、`pymss_core`、`entry`、`h3` 都解析到快照目录。

## 3. 数据（进行中）

```bash
git clone git@github.com:baicai-1145/s2-h3-ref.git /workspace/s2_h3_ref      # 私有库：56 窗逐样本参照（不在公开仓库里）
cd $S2 && S2_REF_DIR=/workspace/s2_h3_ref/ref PYTHONPATH=$S2 python protocol_check.py | tee protocol_check.log
```

`protocol_check.py` 用 Colab 导出的 56 窗参照逐窗比对：偏移相同 = 曲目长度相同；再逐样本比较（`stage.score()` 喂给模型的 float16 分辨率）。**通过条件：56 窗全部到位，且每窗要么逐位相同，要么同时满足——没有任何采样点相差超过 16 个 PCM16 LSB（评测自身的 float16 分辨率），且差异能量 ≤ 窗能量的 1e-9（−90 dB）。** 打出 `PROTOCOL_PASS` 才能训练。

已知情况：偏移全对、能量对到 1e-6、CRC 全不同，差异只在约 0.1% 的采样点上——这是解码/量化最后一位的差异。用逐样本比较量出实际幅度即可，**不必再追最后一位的确切规则**（除非顺手能定位）；把你已查到的现象和结论记进 `NPU_LOG.md`。测试集已经过闸门的话，训练集解码完只需重跑一次。

## 4. 预检（已完成：`PREFLIGHT_PASS`，10 个 arm，零改动）

`entry.py` 的 `install_npu_trainer()` 给 Lightning 注册 `npu` 加速器 + `MixedPrecision('bf16-mixed', 'npu')`；`stage.py` 在 NPU 上不传 `--pin-memory`、fused backend 用 `off`。Ascend 的已知坑见同仓库 `train_ascend/README.md`；`pymss_core` 自带 Ascend 设备档（`complex_fft=False` → `npu_fft`）。

## 5. Profile（现在就做：只需要 NPU，不需要训练数据）

写 `s2_h3/profile_step_npu.py`：用快照里的真实路径（`entry.build('H3')`、pymsst 的 `training_step` + `masked_loss` + AdamW + EMA 回调的等价步骤），batch 24 × 131584 的随机输入，bf16-mixed autocast（`torch.autocast('npu', dtype=torch.bfloat16)`）。

1. 整步速度：预热 10 步后计时 30 步，报 steps/s 与 ms/step。
2. 分段（每段前后 `torch.npu.synchronize()`）：h2d、模型前向再拆成 STFT 前端 / band split / 时间轴（MSC + PooledKV）/ 频率支路（每栈分别）/ final norm / mask head / ISTFT，损失（`masked_loss`，含 `torch.quantile`），反向，优化器，EMA。
3. 算子表：`torch.profiler`（NPU）取 self 时间前 30 的算子，标出落在 AI_CPU / host 的（名字里带 `aicpu`、`Cpu`、`Host` 的，或 device 时间为 0 的）。
4. 结果写 `s2_h3/profile/step_eager.json` + `PROFILE.md`（两张表），提交。这张表决定 Triton 的目标顺序——不要凭感觉写内核。

参照：T4 上 H2 是 2.58 s/step（bs24）。

## 6. Triton 内核（triton-ascend）

放在 `s2_h3/triton_npu.py`：内核 + `torch.autograd.Function`（融合了就写显式 backward）+ `install(model, names)`。只在训练进程（`s2_aw_entry.py`）里按环境变量 `S2_TRITON=<逗号分隔的内核名|all>` 安装；**`stage.py` 的 56 窗评测永远走 eager**，这样评测数字与实现无关。不设 `S2_TRITON` 时行为与现在逐字相同。

候选目标，顺序以第 5 节的 profile 为准：
1. 模型的 STFT / ISTFT 前后端（分帧、加窗、overlap-add 及其伴随；FFT 本体先用 torch 的，除非 profile 说 FFT 是瓶颈）。`train_ascend/fast_istft.py` 是现成的显式伴随写法。
2. 时间轴 MSC 块：RMSNorm → u·SiLU(v) → DWConv(k=31, d=1) → Linear。`train_ascend/triton_blocks.py` 的 `fused_silu_mul`、`norm_dwconv` 就是为同一种块写的，先看能不能直接用。
3. H3 模块里的逐元素链：块均值池化 + RMSNorm；门控 sigmoid×mul + to_out；`BlockFeedForward` 的 SiLU-mul；`StaticBandMixer` 的逐头带混合。
4. `masked_loss` 里的 `torch.quantile`（NPU 上若是回退实现就重写成等价形式，线性插值的定义不能变）。
5. band split / mask head 的归一化链、RoPE：`train_ascend/npu_kernels_bandunet.py` 有现成的，能直接用就用。

每个内核的准入标准（缺一不可，结果写进 `TRITON.md` 一张表）：
- **数值**：在 NPU 上、训练的真实形状下，fp32：输出和梯度的 `rel_rms ≤ 1e-5`（对 eager 自动微分）；bf16 autocast：相对误差不超过 bf16 自身精度（≤ 1e-2）。
- **速度**：训练形状下 eager vs Triton，ABBA 各 20 次取中位，Triton 必须更快才启用；把两个毫秒数都记下来。
- **整步**：同一 batch、同一种子下，单步 loss：fp32 相对差 ≤ 1e-4，bf16 相对差 ≤ 1e-2。

参考与坑：`train_ascend/npu_kernels_bandunet.py`、`triton_blocks.py`、`triton_check.py` 在这台机器上能编译能跑（`backend='npu'`，首次 JIT 约 0.8 s）；UB 容量有限，一维 tile 放不下就改二维；`sqrt(0)` 会产生无穷梯度；Python float 加大 tensor 会触发 vector-core 异常（eps 用 0 维 tensor）。bs24 的整步**不是** launch-bound，收益来自合并访存受限的逐元素链、绕开回退到 AI_CPU 的算子，而不是减少 kernel 数量本身。

## 7. 参照与启动（数据 PASS 之后）

1. **验证参照（只用于验证，不是训练）**：用真实的训练入口、不开 Triton 跑 200 步，`--max-steps 200 --loss-history-interval 1 --train-log-interval 1`，结果放 `runs/H3-ref-eager/`，把 loss 曲线提交。它的用途只有一个：给 Triton 路径做等价对照。
2. **Triton 参照**：同样 200 步，带 `S2_TRITON=...`，放 `runs/H3-ref-triton/`。同一 `--seed 0` 下两条 loss 曲线逐步对比：`mean(|Δloss|/loss) ≤ 1%` 才算通过；同时报两个 steps/s。
3. **正式训练（通过后，全程 Triton）**：

```bash
cd $S2 && S2_DEVICE=npu S2_ROOT=$S2 S2_SKIP_PIP=1 HF_ENDPOINT=https://hf-mirror.com \
  S2_TRITON=<全部已准入的内核> H2_ARM=H3 H2_STEPS=5000,10000,15000,20000,25000,30000,35000,40000,45000,50000 \
  setsid python run_h2.py > run_h2.log 2>&1 < /dev/null &
```

必须脱离 ssh 会话（`setsid` + 重定向，或 tmux）。头 200 步内看一次：loss 有限；`runs/H3/train-5000.log` 里有 `ADAMW_START_VERIFIED H3 0 0.0005`；`npu-smi info` 的 HBM 占用。跑到 50k：每段 STAGE_OK 推上来就继续下一段，不用等我（没有 15k 决策点）；**不要启动其它 arm**。训练期间不要在 NPU 上跑别的任务。启动的先后顺序以第 10 节为准：**先把速度压到最快，再从第 0 步开始。**

**训练必须是 Triton 的，没有例外。** eager 只出现在验证里（单算子参照、200 步参照曲线），**不允许用 eager 启动正式训练**——0.856 steps/s 没法做实验迭代。因此：

- profile 排第 1 的热点（时间轴 DWConv 的权重梯度，496 ms）**必须解决**：两次不过准入就换方案再来（换 tile、换公式——dW 可以写成 31 个移位点积的归约，也可以 unfold 成 (c, k, b·T) 后做小 GEMM），直到过准入为止。启动的先决条件就是这个内核通过并启用。
- 第 2、3 个热点（逐元素链、cast）尽力而为：按同样的准入标准做，不过就记录原因跳过。目标是整步明显快于 eager——每过一批准入，就报一次整步 ms 和相对 eager 的倍数。
- 启动之后的速度工作按第 10 节。

## 8. 回传

每个阶段结束、以及任何失败时：

```bash
cd /workspace/s2_h3 && git add s2_h3/out/H3/*.json s2_h3/out/H3/train-*.log s2_h3/runs/H3/time-*.json \
  s2_h3/runs/H3/train-*.log s2_h3/runs/H3-ref-*/ s2_h3/profile/ s2_h3/*.md s2_h3/*.log s2_h3/*.py s2_h3/*.json
git commit -m "H3@npu <阶段>: <一句话结果>" && git push origin exp/h3-ascend
```

`.gitignore` 已排除 `*.pt`、`*.ckpt`、`data/`、`mp3/`；**不要把权重和数据推上去**（仓库是公开的）。

`STATUS.md` 第一行写当前状态，例如：
`STATUS: triton dev (2/3 passed: silu_mul, gate_out) | eager 1.9 steps/s | updated 2026-11-27 03:10`
我只读这一行就知道进度。`NPU_LOG.md` 记互操作绕法，`TRITON.md` 记每个内核的数值/速度/是否启用。

## 9. 不允许

- 改超参、损失定义、数据管线、评测窗口、优化器、EMA、学习率计划、段长。
- 用 `train_ascend/` 的裸训练循环代替：那份配方和 Colab 的不等价。
- 没过第 6 节准入标准的内核进训练。
- 用 eager 启动正式训练，或训练中途退回 eager。
- 训练时在 NPU 上并发跑别的东西。

对照（Colab 的 H2，56 窗中位，EMA）：5k 5.066 / 10k 6.818 / 15k 7.742 / 20k 7.972 dB，全程曲线在 `FINDINGS.md`。H3 直接跑满 50k，逐段和 H2 比；看趋势，不看单点。

## 10. 先把速度压到最快，再从第 0 步跑 50k（用户的指令，2026-11-25；覆盖 §7 的启动顺序）

用户的指令：**先停掉训练，把训练速度压到最快，再从第 0 步开始训练 50k。** 起点就是第一轮的结果：`dwconv_dw` 过准（2.03×），`silu_mul` 和卷积前向/dX 在这台机器上跑不过 aclnn，被拒；整步 844.6 ms。

1. **现在就停**：结束 `run_h2.py` 和正在跑的 5000 段进程（整个进程组），`npu-smi info` 确认 NPU 空闲。这次 5k 段的数据作废，不保留。`STATUS.md` 第一行改成 `SPEED ROUND`。
2. **速度冲刺**（独占 NPU；每采纳一个内核就重新 profile）：
   - **轮次 1：`dwconv_dw` v2。** 现在的内核每个 conv 仍要 62.7 ms，4 个 conv ≈ 250 ms/步，是剩余最大项。它读的数据量（X 的 31 个移位视图 ≈ 9.4 GB + dy 0.3 GB）在 1.4 TB/s 上只要约 7 ms，所以瓶颈是并行度和每轮迭代的延迟：程序数只有 (C=384) × (NS=4) = 1536，每个程序串行 372 行 × 3 个时间片 = 1116 次小迭代。两条路，都先做数值再做计时：
     - (a) 把时间片和行块也放进 grid（程序数 ≥ 4608，`tl.atomic_add` 累加），行方向一次迭代处理多行（BR=4 时 tile 约 64 KB，UB 放得下），减少迭代次数。
     - (b) 换公式走 cube core：`dW[c] = Σ_n DY[n,c,:] @ unfold(X[n,c,:])ᵀ`，即按通道的批量 GEMM（c=384, K=N·T, N=31），unfold 按行块分块物化（每块约 190 MB）再 `bmm`。数值上它比现在的版本更准（求和顺序不同），准入标准不变。
     目标：≤ 10 ms/conv。
   - **轮次 2：h2d 59 ms/步。** NPU 上没有 pinned memory。看 `torch_npu` 有没有可用的 pinned host / 预取手段，把拷贝和计算重叠。不改数据：dataloader 的采样顺序、batch 内容必须逐位不变（同一 `--seed 0` 下 200 步 loss 对照通过即可证明）。
   - **轮次 3 起：按新 profile 处理剩下的单项 ≥ 25 ms 的热点**，从大到小。逐元素链只减算子数量，不写替换 aclnn 逐元素算子的 Triton 内核（已证明更慢）；候选包括 cast、卷积 dX、attention 路径、dropout、优化器 + EMA。
   - 准入标准不变（§6）。每轮结束报：整步 ms、相对 eager（1169 ms）的倍数、新的 profile 表；写进 `profile/step_roundN.json` 和 `PROFILE.md`。
   - **停止条件**（满足即停）：新 profile 里不再有单项 ≥ 25 ms 的可处理热点，或连续 3 个尝试没过准。硬上限：从停训起 6 小时——到点用届时最快的已准入配置启动，没做完的写进 `TRITON.md`，留到段边界继续。
   - 参照：eager 1169 ms，现在 845 ms。我估计的合理目标是 ≤ 450 ms/步（≥ 2.2 steps/s：dW v2 省约 210 ms，h2d 重叠省 45–59 ms，其余逐项）；达到或更好就停。
3. **从第 0 步启动 50k**：先用最终的内核列表跑 200 步参照（`runs/H3-ref-triton2/`），和已有的 eager 200 步曲线逐步对照（`mean(|Δloss|/loss) ≤ 1%`），报 steps/s；通过后按 §7.3 的命令启动队列（不带 resume），并在 `TRITON.md` 写清启用的内核列表和整步 ms。
4. **跑到 50k**：每段 STAGE_OK 推上来就继续下一段，不用等我。段边界可以切换新内核（过同样的准入 + 200 步对照；新段第一句写清内核列表，steps/s 和 loss 曲线的连续性记进 `TRITON.md`）。

## 12. 名字（2026-11-26）

从此在文档、提交信息、`STATUS.md`、报告里用名字，代码里两个写法都认（`entry.ALIASES` / `entry.resolve`）：

| 名字 | 代码键 | 是什么 | 参数量 | 50k 中位 |
|---|---|---|---:|---:|
| **Stock** | A | 原始 BS-RoFormer（baseline） | 24.59M | 9.295 |
| **Pedal** | H2 | 频率轴注意力在 0、4 层换成池化 K/V——审计结论是"逐帧路由只是围绕慢变均值的噪声采样"，把 K/V 压成那个均值 | 21.58M | **9.406** |
| **Largo** | H3 | Pedal 再走一步：0–2 层不要注意力，换门控静态带混合；3–5 层每 4 帧才路由一次 + 窄 v + 每 4 帧一个 FFN | 20.27M | 9.103 |
| Pedal-M / Pedal-MG / Pedal-N128 | H2M / H2MG / H2N128 | Pedal 的变体 | 19.40M / 19.40M / 10.44M | — |

已有的产物（`runs/H2`、`out/H3`、权重文件名、HF 仓库路径）不改名——都是归档，改名只会让链接和历史断掉。新跑的东西用名字当 `ARM`（例如 `ARM=Largo`），目录就会叫 `out/Largo`。
