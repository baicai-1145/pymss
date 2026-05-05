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
    'apollo']
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
- inference_params: 推理参数，包括 batch_size、overlap_size、chunk_size 和 normalize。 默认值均为 None（意味着所有参数都取决于配置文件）。
## 贡献
欢迎贡献！
