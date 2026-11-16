"""The adaptive API writes exactly one full-length file per detected identity."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from pymss_core.modules.bs_roformer import AdaptiveBSRoformer
from torch import nn

from pymss.speakers import SpeakerSeparator


class ScriptedExistence(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count, self.step = count, 0

    def forward(self, x):
        result = x.new_full((x.shape[0], 1), 10. if self.step < self.count else -10.)
        self.step += 1
        return result


def separator(tmp_path):
    config = {'model_type': 'adaptive_bs_roformer', 'model': {
        'dim': 8, 'depth': 1, 'max_speakers': 5, 'decoder_heads': 1, 'memory_frames': 8,
        'heads': 1, 'dim_head': 8, 'freqs_per_bands': [4, 4, 4, 5],
        'stft_n_fft': 32, 'stft_win_length': 32, 'stft_hop_length': 8,
        'time_transformer_depth': 1, 'freq_transformer_depth': 1}, 'audio': {'sample_rate': 8000, 'duration': 4.}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    weights = tmp_path / 'model.pt'
    torch.save({'state_dict': AdaptiveBSRoformer(**config['model']).state_dict(), 'config': config}, weights)
    return SpeakerSeparator(weights, path, device='cpu')


@pytest.mark.parametrize('count', [0, 3, 4, 5])
def test_output_file_count_and_duration(tmp_path, count):
    runtime = separator(tmp_path)
    runtime.model.existence_head = ScriptedExistence(count)
    source = tmp_path / 'input.wav'
    sf.write(source, np.random.default_rng(1).normal(0, .1, 800).astype(np.float32), 8000)
    result = runtime.separate_file(source, tmp_path / 'output')
    assert result['speakers'] == count and not result['limit_reached']
    assert len(list((tmp_path / 'output').glob('speaker_*.wav'))) == count
    assert json.loads((tmp_path / 'output' / 'separation.json').read_text())['speakers'] == count
    for path in result['tracks']:
        info = sf.info(path)
        assert info.frames == 800 and info.samplerate == 8000
    with pytest.raises(FileExistsError):
        runtime.separate_file(source, tmp_path / 'output')


def test_standard_cli_exposes_adaptive_command():
    from pymss.cli import build_parser
    args = build_parser().parse_args(['separate-speakers', 'input.wav', '--checkpoint', 'best.pt',
                                      '--config', 'config.yaml', '--output-dir', 'output'])
    assert args.command == 'separate-speakers'
    assert args.func.__module__ == 'pymss.speakers'


def test_limit_flag_and_invalid_input(tmp_path):
    runtime = separator(tmp_path)
    runtime.model.existence_head = ScriptedExistence(6)
    result = runtime.separate(np.zeros(128, dtype=np.float32))
    assert result.count == 5 and result.limit_reached
    with pytest.raises(ValueError, match='nonfinite'):
        runtime.separate(np.full(64, np.nan))
    with pytest.raises(ValueError):
        runtime.separate(np.empty(0))
