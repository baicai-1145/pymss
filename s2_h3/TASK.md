# 任务：在昇腾 950PR 上，用 Colab 的同一套协议训练 H3

**本文件是任务的权威版本。** 你（服务器上的 pi）负责执行和环境排障；我（Mac 上的 lead）负责核对数字、和 Colab 的 H2 曲线对比。每完成一个阶段就把结果提交回分支并更新 `STATUS.md`。

## 0. 一句话

- 代码：`/workspace/s2_h3` 是 `/workspace/pymss` 的一个 worktree（分支 `exp/h3-ascend`）。实验目录 `S2=/workspace/s2_h3/s2_h3`，等价于 Colab 的 `/content/s2`。
- 协议一个字不改：batch 24、AdamW 5e-4 / wd 0.01 / StepLR(40000, 0.9)、EMA 0.999、bf16-mixed、5k 步一段、56 窗评测。**只允许改"设备互操作"这一层。**
- 数据必须是 Colab 那一份（HF `Viol2000/musdb18hq` 的 MP3 解码件）。`protocol_check.py` 打出 `PROTOCOL_PASS` 之前不许开始训练。

## 1. 取代码

```bash
cd /workspace/pymss && git fetch origin exp/h3-ascend
git worktree add /workspace/s2_h3 exp/h3-ascend        # 已存在则: git -C /workspace/s2_h3 pull --ff-only
```

不要动 `/workspace/pymss`（在 `feat/ascend-band-unet`）、`/workspace/pymss-core`、`/workspace/pymsst` 的工作区和 git 状态。

## 2. 环境：只增不改

- 保持现有版本：torch 2.13.0+cpu、torch_npu 2.13.0、pytorch-lightning 2.6.6、triton-ascend 3.6.0.post2、numpy、scipy。**不要升级或降级任何已安装的包。**
- `apt-get install -y ffmpeg`（`prepare.py` 用 subprocess 调 `ffmpeg`；Ubuntu 22.04 的 4.4.2 和 Colab 是同一个包）。
- `python -m pip install soundfile huggingface_hub`，以及 pymsst 缺的运行依赖（`einops beartype ml_collections lazy_loader rotary_embedding_torch einja` 等，以实际 import 报错为准；pip 已指向华为镜像）。pip 命令不要连带改动 torch，必要时加 `--no-deps`。
- 代码解析路径：所有命令都带 `PYTHONPATH=$S2`，让 `pymsst`、`pymss_core`、`entry` 都解析到快照目录（`pymss_core` 与 `/workspace/pymss-core@2365094` 内容相同；`pymsst` 是 Colab 实际使用的版本，要优先于 `/workspace/pymsst` 的 editable 安装）。确认：
  `PYTHONPATH=$S2 python -c "import pymsst, pymss_core, entry; print(pymsst.__file__, pymss_core.__file__, entry.__file__)"`

## 3. 数据

```bash
cd $S2
HF_ENDPOINT=https://hf-mirror.com S2_ROOT=$S2 PYTHONPATH=$S2 python prepare.py 2>&1 | tee prepare.log
PYTHONPATH=$S2 python protocol_check.py | tee protocol_check.log      # 必须 PROTOCOL_PASS
```

- `prepare.py` 下载 `train.tar` / `test.tar`（约 10 GB）、解出 MP3、解码成 PCM16 WAV（约 45 GB）。磁盘有 251 GB 空闲。`hf-mirror.com` 从这台机器可达（已测 200）；HuggingFace 本站不可达。
- `protocol_check.py` 拿 Colab 导出的 56 窗参照（每窗的偏移、两种音轨 float16 PCM 的 CRC32、能量）逐窗比对：偏移相同 = 曲目长度相同；CRC 相同 = 解码样本逐位相同。不 PASS 就停下，把输出发给我，不要训练。

## 4. 预检

```bash
cd $S2 && S2_DEVICE=npu PYTHONPATH=$S2 python entry.py check 2>&1 | tail -20     # 要看到 PREFLIGHT_PASS
```

- NPU 路径在 `entry.py` 的 `install_npu_trainer()`：给 Lightning 注册 `npu` 加速器（单设备）+ `MixedPrecision('bf16-mixed', 'npu')`。`stage.py` 在 NPU 上不传 `--pin-memory`，fused backend 用 `off`。CUDA 上的行为逐字不变。
- Ascend 上已知的坑见同一仓库的 `train_ascend/README.md`（`sqrt(0)` 梯度、Python float 加大 tensor、复数 STFT）。`pymss_core` 自带 Ascend 设备档（`complex_fft=False` → `npu_fft`）。
- 遇到不支持的算子/后端：只找**不改数学**的实现层绕法（换 SDPA 后端、关某个融合、把某步搬运放 CPU 等）。每一条都记进 `NPU_LOG.md`（现象、原因、改法、为什么等价）。对 `s2_h3/*.py` 的任何改动都要留在 diff 里让我能看到。

## 5. 启动

```bash
cd $S2 && S2_DEVICE=npu S2_ROOT=$S2 S2_SKIP_PIP=1 HF_ENDPOINT=https://hf-mirror.com \
  H2_ARM=H3 H2_STEPS=5000,10000,15000,20000,25000 \
  setsid python run_h2.py > run_h2.log 2>&1 < /dev/null &
```

- 必须脱离 ssh 会话（`setsid` + 重定向，或 tmux）：断线不能中断训练。
- 头 200 步内看一次：loss 有限；`runs/H3/train-5000.log` 里有 `ADAMW_START_VERIFIED H3 0 0.0005`；`npu-smi info` 的 HBM 占用。之后每段结束 `run_h2.log` 里出现 `STAGE_OK H3 <step> <median_dB>`。
- 15k 是决策点：跑完 15k 段、把结果推上来之后继续 20k/25k，不用等我；但**不要启动其它 arm**（H2 对照要不要在 NPU 上重训，等我看过 5k/10k 再定）。
- 训练期间不要在 NPU 上跑别的任务（会污染速度数据）。

## 6. 回传

每段 `STAGE_OK` 之后，以及任何失败时：

```bash
cd /workspace/s2_h3 && git add s2_h3/out/H3/*.json s2_h3/out/H3/train-*.log s2_h3/runs/H3/time-*.json \
  s2_h3/runs/H3/train-*.log s2_h3/*.log s2_h3/NPU_LOG.md s2_h3/STATUS.md s2_h3/*.py
git commit -m "H3@npu step <step>: <median_dB> dB" && git push origin exp/h3-ascend
```

`.gitignore` 已排除 `*.pt`、`*.ckpt`、`data/`、`mp3/`；**不要把权重和数据推上去**（仓库是公开的）。

`STATUS.md` 第一行写当前状态，例如：
`STATUS: training H3 3721/5000 | done: 5k=5.1 dB | updated 2026-11-27 03:10`
我只读这一行就知道进度。

## 7. 不允许

- 改超参、损失、数据管线、评测窗口、优化器、EMA、学习率计划、段长。
- 用 `train_ascend/` 的裸训练循环代替：那份配方和 Colab 的不等价（BandUNet 16k 步只有 3.9 dB，Colab 配方下 H2@10k 是 6.8 dB）。
- 报数之外的任何"解释性"调整。

对照（Colab 的 H2，56 窗中位，EMA）：5k 5.066 / 10k 6.818 / 15k 7.742 / 20k 7.972 dB。判据：H3 在 15k 时与 H2@15k 差距 ≤ 0.3 dB 才进 50k；看趋势，不看单点。
