# pymss

Python package for music source separation. <br>
[English]   [简体中文](./README_CN.md)

## Install

Example of using pip to install `pymss` package：

```sh
pip install pymss
```

## Usage

Here's a simple example.
```python
from pymss import MSSeparator, get_separation_logger

# init
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
        "other": None # None or missing this stem will result in no output file for this stem. This example will output the vocal's stem in ./output/vocals and ignoring the other(instrumental) stem. Making sure the key(s) match the config file.
    },
    audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}, # Can be omitted
    logger=get_separation_logger(), # Can be omitted
    debug=False, # Can be omitted
    inference_params={
        "batch_size": 4,
        "overlap_size": 512,
        "chunk_size": 1024,
        "normalize": True
    } # Can be omitted
)

# process all audio files in the folder
separator.process_folder('path/to/input_folder')
```

### Parameters

- model_type: The type of model, e.g., 'htdemucs'. Must be one of 
    ['bs_roformer', 
    'mel_band_roformer', 
    'htdemucs', 
    'mdx23c', 
    'bandit', 
    'bandit_v2', 
    'scnet', 
    'apollo']
- model_path: The path to the model file.
- config_path: The path to the configuration file.
- device: The type of device, default is 'auto'. Must be one of ['auto', 'cuda', 'mps', 'cpu']
- device_ids: List of device IDs, default is [0].
- output_format: The output audio format, default is 'wav'. Must be one of ['wav', 'flac', 'mp3']
- use_tta: Whether to use TTA, default is False. Using TTA will triple the processing time with a little bit improvement in quality.
- store_dirs: Storage directories, can be a single folder path or a dictionary with instrument keys.
- audio_params: Audio parameters including wav_bit_depth, flac_bit_depth, and mp3_bit_rate. Default is {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}.
- logger: Logger instance. Default is pymss.get_separation_logger()
- debug: Whether to enable debug mode, default is False.
- inference_params: Inference parameters including batch_size, overlap_size, chunk_size, and normalize. Default is all None (means all params are depended on the config file).

## Contributing
Contributions are welcome! 
