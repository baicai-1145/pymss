# pymss
用于音乐源分离的 Python 包。
[English](./README.md)  [简体中文]
## 安装
使用 pip 安装 `pymss` 包的示例：
```sh
pip install pymss
```
## 用法
这是一个简单的例子。
```python
from pymss import MSSeparator, get_separation_logger
# 初始化
separator = MSSeparator(
    model_type='htdemucs',
    model_path='path/to/model',
    config_path='path/to/config',
    device='cuda',
    device_ids=[0],
    output_format='wav',
    use_tta=True,
    store_dirs={
        "vocals": "./output/vocals",
        "other": None # None 或缺少此音轨将导致不输出此音轨的文件。 此示例将在 ./output/vocals 中输出人声音轨，并忽略其他（乐器）音轨。 确保键与配置文件匹配。
    },
    audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}, # 可以省略
    logger=get_separation_logger(), # 可以省略
    debug=False, # 可以省略
    inference_params={
        "batch_size": 4,
        "overlap_size": 512,
        "chunk_size": 1024,
        "normalize": True
    } # 可以省略
)
# 处理文件夹中的所有音频文件
separator.process_folder('path/to/input_folder')
```
### 参数
- model_type: 模型类型，例如 'htdemucs'。 必须是以下之一
    ['bs_roformer',
    'mel_band_roformer',
    'htdemucs',
    'mdx23c',
    'bandit',
    'bandit_v2',
    'scnet',
    'apollo',
    'vr']
- model_path: 模型文件路径。
- config_path: 配置文件路径。
- device: 设备类型，默认为 'auto'。 必须是以下之一 ['auto', 'cuda', 'mps', 'cpu']
- device_ids: 设备 ID 列表，默认为 [0]。
- output_format: 输出音频格式，默认为 'wav'。 必须是以下之一 ['wav', 'flac', 'mp3']
- use_tta: 是否使用 TTA（测试时增强），默认为 False。 使用 TTA 会使处理时间增加三倍，但质量会略有提高。
- store_dirs: 存储目录，可以是单个文件夹路径或带有乐器键的字典。
- audio_params: 音频参数，包括 wav_bit_depth、flac_bit_depth 和 mp3_bit_rate。 默认为 {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}。
- logger: Logger 实例。 默认为 pymss.get_separation_logger()
- debug: 是否启用调试模式，默认为 False。
- inference_params: 推理参数，包括 batch_size、overlap_size、chunk_size、normalize、`model_compute_dtype`、`cuda_attention_backend` 和 `cuda_triton_backend`。默认值均为 None（意味着参数取配置文件或运行时默认值）。`model_type='vr'` 支持 `batch_size`、`window_size`、`aggression`、`enable_tta`、`enable_post_process`、`post_process_threshold` 和 `high_end_process`。

### CUDA Attention 后端

RoFormer 系列模型在已安装 PyTorch 暴露 cuDNN attention 时默认使用 cuDNN attention，否则使用 PyTorch 默认 SDPA 路径。需要探测式回退时可通过 `inference_params={"cuda_attention_backend": "auto"}` 覆盖。可选值为 `auto`、`default`、`flash`、`cudnn`、`efficient`、`math` 和 `xformers`。`auto` 会优先尝试 cuDNN attention，然后回退到 PyTorch memory-efficient SDPA，再回退到 PyTorch 默认 SDPA。`xformers` 是本地可选安装项，不作为必需依赖。

RoFormer 系列在 CUDA 上还默认使用 `cuda_triton_backend="auto"`。该路径只在本机 Torch/Triton 与当前 tensor shape 支持时启用实测有效的 attention gate/output 融合内核，否则自动回退 Torch 路径。可用 `inference_params={"cuda_triton_backend": "off"}` 或 `"default"` 关闭。诊断值包括 `freq_atomic_out`、`attention_gate` 和 `attention_gate_out`。

启用 CUDA AMP 时，BS-RoFormer 系列默认使用 `model_compute_dtype="auto"`，会让 Linear 权重/偏置和 RMSNorm gamma 在推理中常驻 fp16，避免反复 autocast 权重转换。需要保持 fp32 参数可设置 `inference_params={"model_compute_dtype": "off"}`；CUDA RoFormer 实验可显式使用 `"float16"`。

### RoFormer overlap 速度设置

RoFormer 系列推理可显式设置 `overlap_size=0`，减少 chunk overlap 带来的重复计算，同时保留各模型原始 chunk 长度和 batch size。它会改变 chunk 边界混合方式，所以输出不会和常见 5% overlap 设置逐 bit 一致；但它不是近似 attention，也不改模型 forward。

以下为 `test.m4a`、300 秒输入、RTX 5090、PyTorch 2.9.1+cu128、关闭 TTA、预热 2 次、正式运行 5 次后的实测：

| 模型 | 5% overlap RTFx | 0% overlap RTFx | 0% overlap 处理 1 小时音频 | MUSDB18-HQ test 全曲 SDR 下降 |
|---|---:|---:|---:|---:|
| BS-Roformer-HyperACE_v2_voc | 300.81x | 324.51x | 11.1s | 0.007 dB |
| model_bs_roformer_ep_368_sdr_12.9628 | 138.04x | 146.41x | 24.6s | 0.089 dB |
| logic_bs_roformer | 202.61x | 216.38x | 16.6s | 0.065 dB |
| mel-band-roformer-deux | 209.73x | 217.58x | 16.5s | -0.068 dB |
| Mel-Band-Roformer-big | 229.40x | 244.10x | 14.7s | -0.049 dB |

目前 BS-Roformer-HyperACE_v2_voc 测到的最快精确推理使用 `chunk_size=160000`、`batch_size=5`、5% overlap，并对 `_mask_stft_repr` 使用 CUDA graph capture：**358.39x**，即处理 1 小时音频约 **10.0s**。相对未 capture 路径的 chunk 级最大绝对误差为 `1.74e-7`。

显式开启 0% overlap 的方式：

```python
inference_params={
    "overlap_size": 0,
}
```

### RoFormer 实验性近似时间 attention

RoFormer 系列有两个显式开启、默认关闭的近似时间 attention 开关。它们会改变模型输出，所以只能作为速度/质量实验使用，目标素材上需要重新验证。

`approx_time_kv_stride` 保持 Q、残差、FFN 和输出长度全分辨率，只在选中的 time-attention 层里对 K/V 做 sample 或 avg。对 BS-Roformer-HyperACE_v2_voc 来说，它明显比直接压缩整条 time-token 序列更稳。

```python
inference_params={
    "approx_time_kv_stride": 2,
    "approx_time_kv_stride_start_layer": 4,
    "approx_time_kv_stride_every": 2,
    "approx_time_kv_stride_mode": "sample",
}
```

以下为 `test.m4a`、300 秒输入、RTX 5090、PyTorch 2.9.1+cu128、关闭 TTA、预热 2 次后的实测：

| 模型 | 设置 | RTFx | MUSDB18-HQ test 全曲 SDR 下降 |
|---|---|---:|---:|
| BS-Roformer-HyperACE_v2_voc | 精确 | 301.74x | 0.000 dB |
| BS-Roformer-HyperACE_v2_voc | KV sample `stride=2,start_layer=4,every=2` | 308.44x | 0.467 dB |
| BS-Roformer-HyperACE_v2_voc | KV sample `stride=2,start_layer=4,every=3` | 305.57x | 0.275 dB |

近似模式必须保持显式开启；默认仍是精确推理。

### Apple Silicon MLX 后端

在 `device='mps'` 时，可以通过 `inference_params` 显式启用可选 MLX 完整 forward：

```python
inference_params={
    "mps_model_backend": "mlx_full",
    "mps_model_compute_dtype": "float16",
}
```

该后端需要本地安装 `mlx`，但当前不会作为 `setup.py` 必需依赖安装。默认推理仍使用 Torch 路径；缺少 MLX 或 backend 运行失败时，非 VR 模型会记录 `_pymss_mlx_full_backend_error` 并回退 Torch。

### 模型兼容性

Demucs 仅支持配置为 `model: htdemucs` 且 `htdemucs.cac: true` 的 HTDemucs checkpoint。当前无外部依赖推理路径不支持 classic `model: demucs`、`model: hdemucs` 和 non-CaC Wiener Demucs 配置。

UVR VR 可通过 `model_type='vr'` 使用，支持已适配的 UVR/VR 系列 `.pth` 权重。输出 stem 名称来自内置 VR 模型列表，例如 `Vocals`、`Instrumental`、`No Echo` 或 `Echo`。

```python
separator = MSSeparator(
    model_type='vr',
    model_path='pretrain/VR_Models/1_HP-UVR.pth',
    device='cuda',
    output_format='wav',
    store_dirs={
        "Vocals": "./output/vocals",
        "Instrumental": "./output/instrumental",
    },
    inference_params={
        "batch_size": 2,
        "window_size": 512,
        "aggression": 5,
    },
)
separator.process_folder('path/to/input_folder')
```

### Hugging Face 配置提醒
一些从 Hugging Face 或 MSST-WebUI 下载的模型配置使用 `inference.num_overlap`。当前优化后的 pymss 路径使用 `inference.overlap_size`。如果配置里只有 `num_overlap`，请手动添加 `overlap_size`，或通过 `inference_params` 传入；否则 pymss 会回退到 50% overlap，推理会慢很多。

原始 HyperACE chunk schedule 的推荐快速设置：

```yaml
audio:
  chunk_size: 480000
inference:
  batch_size: 2
  overlap_size: 0
```

### RTX 5090 实测

测试环境为 NVIDIA GeForce RTX 5090、PyTorch 2.9.1+cu128、CUDA 12.8，关闭 TTA。主模型表使用实测最快设置；RoFormer 和 smoke 模型均为预热 2 次、正式运行 5 次后的确认结果。

| 模型 | 类型 | 最快设置 | RTFx | 1 小时音频 |
|---|---|---|---:|---:|
| BS-Roformer-HyperACE_v2_voc | bs_roformer | chunk=160000, overlap=5%, batch=5, CUDA graph | 358.39x | 10.0s |
| model_bs_roformer_ep_368_sdr_12.9628 | bs_roformer | chunk=120000, overlap=0, batch=8 | 154.92x | 23.2s |
| logic_bs_roformer | bs_roformer | chunk=160000, overlap=0, batch=4 | 274.72x | 13.1s |
| mel-band-roformer-deux | mel_band_roformer | chunk=160000, overlap=0, batch=6 | 271.66x | 13.3s |
| Mel-Band-Roformer-big | mel_band_roformer | chunk=160000, overlap=0, batch=4 | 274.04x | 13.1s |
| model_vocals_mdx23c_sdr_10.17 | mdx23c | chunk=130560, overlap=0, batch=2 | 235.77x | 15.3s |
| HTDemucs4 | htdemucs | chunk=646800, overlap=0, batch=12 | 350.85x | 10.3s |
| scnet_checkpoint_musdb18 | scnet | chunk=242550, overlap=0, batch=8 | 599.39x | 6.0s |
| model_bandit_plus_dnr_sdr_11.47 | bandit | chunk=264600, overlap=0, batch=10 | 257.44x | 14.0s |
| checkpoint-multi_state_dict | bandit_v2 | chunk=256000, overlap=0, batch=12 | 230.69x | 15.6s |
| Apollo_LQ_MP3_restoration | apollo | chunk=66150, overlap=0, batch=2 | 110.32x | 32.6s |

VR 模型使用 `window_size=512`、`aggression=5`，关闭 TTA 和后处理，先搜索 batch，再按表中 batch 预热 2 次、正式运行 3 次确认。

| VR 模型 | 最快设置 | RTFx | 1 小时音频 |
|---|---|---:|---:|
| UVR-DeNoise-Lite | batch=10 | 270.00x | 13.3s |
| Harmonic_Noise_Separation_yxlllc | batch=12 | 235.63x | 15.3s |
| MGM_HIGHEND_v4 | batch=6 | 232.91x | 15.5s |
| MGM_LOWEND_A_v4 | batch=6 | 131.64x | 27.3s |
| MGM_MAIN_v4 | batch=6 | 125.51x | 28.7s |
| 10_SP-UVR-2B-32000-1 | batch=10 | 113.87x | 31.6s |
| 11_SP-UVR-2B-32000-2 | batch=6 | 113.06x | 31.8s |
| 12_SP-UVR-3B-44100 | batch=12 | 112.00x | 32.1s |
| MGM_LOWEND_B_v4 | batch=6 | 109.21x | 33.0s |
| 15_SP-UVR-MID-44100-1 | batch=6 | 108.24x | 33.3s |
| 14_SP-UVR-4B-44100-2 | batch=4 | 107.36x | 33.5s |
| 16_SP-UVR-MID-44100-2 | batch=10 | 107.33x | 33.5s |
| 13_SP-UVR-4B-44100-1 | batch=8 | 105.81x | 34.0s |
| 5_HP-Karaoke-UVR | batch=4 | 93.09x | 38.7s |
| 2_HP-UVR | batch=4 | 91.50x | 39.3s |
| UVR-DeNoise | batch=12 | 88.52x | 40.7s |
| UVR-De-Echo-Aggressive | batch=10 | 88.12x | 40.9s |
| UVR-De-Echo-Normal | batch=12 | 87.84x | 41.0s |
| 4_HP-Vocal-UVR | batch=6 | 87.77x | 41.0s |
| 3_HP-Vocal-UVR | batch=6 | 87.35x | 41.2s |
| 1_HP-UVR | batch=2 | 86.09x | 41.8s |
| UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband | batch=12 | 83.88x | 42.9s |
| UVR-DeEcho-DeReverb | batch=6 | 83.34x | 43.2s |
| 17_HP-Wind_Inst-UVR | batch=12 | 82.82x | 43.5s |
| 6_HP-Karaoke-UVR | batch=4 | 82.02x | 43.9s |
| UVR-BVE-4B_SN-44100-1 | batch=8 | 78.57x | 45.8s |
| 8_HP2-UVR | batch=6 | 57.70x | 62.4s |
| 9_HP2-UVR | batch=4 | 57.24x | 62.9s |
| 7_HP2-UVR | batch=8 | 55.96x | 64.3s |

## 贡献
欢迎贡献！
