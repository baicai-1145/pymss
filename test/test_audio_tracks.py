import shutil
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from pymss.audio_io import list_audio_tracks, load_audio
from test.test_separator_channels import make_separator


@pytest.fixture
def dual_track_media(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required to create the dual-track fixture")
    path = tmp_path / "dual.mka"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.08:sample_rate=8000",
        "-f", "lavfi", "-i", "sine=frequency=900:duration=0.08:sample_rate=8000",
        "-map", "0:a", "-map", "1:a", "-metadata:s:a:0", "language=jpn",
        "-metadata:s:a:0", "title=Japanese", "-metadata:s:a:1", "language=eng",
        "-metadata:s:a:1", "title=English", "-c:a", "flac", str(path),
    ], check=True)
    return path


def test_list_and_load_selected_audio_track(dual_track_media):
    tracks = list_audio_tracks(dual_track_media)
    assert [(track["index"], track["language"], track["title"]) for track in tracks] == [
        (1, "jpn", "Japanese"), (2, "eng", "English"),
    ]
    assert [(track["channels"], track["sampleRate"]) for track in tracks] == [(1, 8000), (1, 8000)]
    first, first_rate = load_audio(dual_track_media, sr=None)
    second, second_rate = load_audio(dual_track_media, sr=None, audio_track=2)
    assert first_rate == second_rate == 8000
    assert first.shape == second.shape
    assert not np.allclose(first, second)
    with pytest.raises(ValueError, match="positive integer"):
        load_audio(dual_track_media, audio_track=True)
    with pytest.raises(RuntimeError, match="Audio track 3"):
        load_audio(dual_track_media, audio_track=3)


def test_non_first_track_does_not_fall_back_to_librosa(monkeypatch, dual_track_media):
    import pymss.audio_io as audio_io

    failing_av = Mock(side_effect=RuntimeError("PyAV failed"))
    librosa = Mock(side_effect=AssertionError("librosa was used"))
    monkeypatch.setattr(audio_io, "_load_audio_av", failing_av)
    monkeypatch.setattr(audio_io, "_load_audio_librosa", librosa)
    audio, rate = load_audio(dual_track_media, sr=None, audio_track=2)
    assert rate == 8000 and audio.size > 0
    failing_av.assert_called_once()
    librosa.assert_not_called()


def test_default_first_track_keeps_loader_call_signature(monkeypatch):
    import pymss.audio_io as audio_io

    loader = Mock(return_value=(np.zeros(4, dtype=np.float32), 8000))
    monkeypatch.setattr(audio_io, "_load_audio_av", loader)
    load_audio("input.wav", sr=8000, mono=True, offset=1.0, duration=2.0, downmix_stereo=True, return_layout=False)
    loader.assert_called_once_with(
        "input.wav", sr=8000, mono=True, offset=1.0, duration=2.0,
        downmix_stereo=True, return_layout=False,
    )


def _configured_separator(tmp_path):
    separator = make_separator(2)
    separator.store_dirs = {"vocals": str(tmp_path / "out")}
    separator.output_format = "wav"
    separator.audio_params = {"wav_bit_depth": "FLOAT"}
    separator.save_as_folder = True
    separator.debug = True
    return separator


def test_process_audio_tracks_reuses_model_and_saves_source_folder(tmp_path, dual_track_media, monkeypatch):
    separator = _configured_separator(tmp_path)
    model = separator.model
    separate = Mock(wraps=separator.separate)
    monkeypatch.setattr(separator, "separate", separate)
    records = separator.process_audio_tracks(dual_track_media, audio_tracks=[2, 1])
    assert [record["index"] for record in records] == [2, 1]
    assert [record["name"] for record in records] == ["dual [track 2 eng]", "dual [track 1 jpn]"]
    assert all(record["success"] for record in records)
    assert separator.model is model
    assert separate.call_count == 2
    assert len(model.calls) > 0
    output_dir = tmp_path / "out" / "dual"
    assert (output_dir / "dual [track 1 jpn]_vocals.wav").is_file()
    assert (output_dir / "dual [track 2 eng]_vocals.wav").is_file()


def test_process_audio_tracks_validates_all_selections_before_processing(tmp_path, dual_track_media, monkeypatch):
    separator = _configured_separator(tmp_path)
    separate = Mock(wraps=separator.separate)
    monkeypatch.setattr(separator, "separate", separate)
    for selection in ([], [True], [3], [1, 3]):
        with pytest.raises(ValueError):
            separator.process_audio_tracks(dual_track_media, audio_tracks=selection)
    separate.assert_not_called()


def test_process_audio_tracks_reports_save_failure_and_continues(tmp_path, dual_track_media, monkeypatch):
    separator = _configured_separator(tmp_path)
    original_save = separator._save_output
    calls = []

    def fail_first_save(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("save failed")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(separator, "_save_output", fail_first_save)
    records = separator.process_audio_tracks(dual_track_media, audio_tracks=[1, 2])
    assert records[0]["success"] is False and records[0]["error"] == "One or more output files could not be saved."
    assert records[1]["success"] is True


def test_single_track_keeps_basename_and_process_folder_is_unchanged(tmp_path, dual_track_media):
    single = tmp_path / "single.mka"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(dual_track_media), "-map", "0:a:0", "-c", "copy", str(single)], check=True)
    separator = _configured_separator(tmp_path)
    records = separator.process_audio_tracks(single)
    assert records[0]["name"] == "single" and records[0]["success"] is True
    assert (tmp_path / "out" / "single" / "single_vocals.wav").is_file()

    ordinary = _configured_separator(tmp_path)
    assert ordinary.process_folder(str(single)) == ["single.mka"]
    assert (tmp_path / "out" / "single" / "single_vocals.wav").is_file()


def test_process_audio_tracks_reports_separation_failure_and_continues(tmp_path, dual_track_media, monkeypatch):
    separator = _configured_separator(tmp_path)
    original = separator.separate
    calls = []

    def fail_first(mix, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first track failed")
        return original(mix, **kwargs)

    monkeypatch.setattr(separator, "separate", fail_first)
    failed = separator.process_audio_tracks(dual_track_media, audio_tracks=[1, 2])
    assert failed[0]["success"] is False and failed[0]["error"] == "first track failed"
    assert failed[1]["success"] is True
