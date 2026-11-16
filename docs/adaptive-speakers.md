# Separate an unknown number of speakers

Install the matching `feat/adaptive-speaker-roformer` source branches of pymss and pymss-core. Train weights using the companion pymsst branch; no pretrained weights or automatic quality claims are bundled.

```bash
python -m pip install -e ../pymss-core -e .
python -m pymss.speakers conversation.wav \
  --checkpoint /path/to/run/best.pt \
  --config /path/to/run/config.yaml \
  --output-dir /path/to/new-output --device cuda
```

The same arguments are accepted by `pymss separate-speakers`. This dedicated path does not assume instrument names or a fixed number of stems. It writes `speaker_001.wav`, `speaker_002.wav`, etc., one per detected identity, and `separation.json`. Zero detected speakers produces metadata and no WAV files. Existing speaker outputs are rejected rather than overwritten or mixed with stale tracks.

Inputs are decoded with the existing audio I/O, resampled to the model rate, and downmixed for mono models. Outputs use float WAV at the model rate and retain the decoded input length. `separation.json` records the detected count, output paths and whether the safety limit was reached while the model still predicted another speaker. `--threshold` controls the existence decision; calibrate it on validation data. `--speaker-limit` overrides the default budget but does not establish reliable separation beyond trained counts.

```python
from pymss import SpeakerSeparator

separator = SpeakerSeparator("best.pt", "config.yaml", device="cpu")
# Input is [channels, samples] at separator.sample_rate (or [samples] for mono).
result = separator.separate(audio)
print(result.count, result.tracks.shape, result.limit_reached)
```

All audio in one call is processed together. Full-clip temporal attention can exceed memory on long recordings. Automatic chunk matching, streaming and cross-call identity tracking are not implemented, so the interface does not silently concatenate unrelated per-chunk speaker slots. Initial speech weights do not establish performance on singing, chorus, unison, music mixtures or unseen speaker counts.
