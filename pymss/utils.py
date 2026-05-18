from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict

from .config import load_config

def get_model_from_config(model_type, config_path):
    config = load_config(config_path)

    if model_type == 'mdx23c':
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        return TFC_TDF_net(config), config
    elif model_type == 'htdemucs':
        from .modules.demucs4ht import get_model
        return get_model(config), config
    elif model_type == 'mel_band_roformer':
        from .modules.bs_roformer import MelBandRoformer
        return MelBandRoformer(**dict(config.model)), config
    elif model_type == 'bs_roformer':
        from .modules.bs_roformer import BSRoformer
        return BSRoformer(**dict(config.model)), config
    elif model_type == 'bs_roformer_hyperace':
        from .modules.bs_roformer import BSRoformerHyperACE
        return BSRoformerHyperACE(**dict(config.model)), config
    elif model_type == 'bandit':
        from .modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        return MultiMaskMultiSourceBandSplitRNNSimple(**config.model), config
    elif model_type == 'bandit_v2':
        from .modules.bandit_v2.bandit import Bandit
        return Bandit(**config.kwargs), config
    elif model_type == 'scnet':
        from .modules.scnet import SCNet
        return SCNet(**config.model), config
    elif model_type == 'apollo':
        from .modules.look2hear.apollo import Apollo
        return Apollo(**config.model), config
    elif model_type == 'vr':
        raise ValueError("VR models are loaded directly by MSSeparator and do not use YAML config loading")
    raise ValueError(f"Model type {model_type} not supported")

def _getWindowingArray(window_size, fade_size):
    if fade_size <= 0:
        return torch.ones(window_size)

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def _build_chunk_plan(total_length, chunk_size, step, fade_size):
    starts = list(range(0, total_length, step))
    normal_window = _getWindowingArray(chunk_size, fade_size)

    def window_for(start):
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            return normal_window
        window = normal_window.clone()
        if start == 0:
            window[:fade_size] = 1
        if start + length >= total_length:
            window[max(0, length - fade_size):length] = 1
        return window

    return starts, [window_for(start) for start in starts]


def _get_inference_step(config, chunk_size):
    overlap_size = int(config.inference.get('overlap_size', chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("inference.overlap_size must be >= 0 and < audio.chunk_size")
    return chunk_size - overlap_size


def _complete_chunk_count(total_length, chunk_size, step):
    return 0 if total_length < chunk_size else (total_length - chunk_size) // step + 1


def _fold_windows(counter, windows, step, start_offset=0):
    n_chunks = windows.shape[0]
    if n_chunks == 0:
        return

    chunk_size = windows.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    folded_counter = nn.functional.fold(
        windows.transpose(0, 1).unsqueeze(0),
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    counter[..., start_offset:start_offset + output_length] += folded_counter.view(1, 1, output_length)


def _fold_chunk_batch(result, chunks, windows, step, start_offset=0):
    n_chunks = chunks.shape[0]
    if n_chunks == 0:
        return

    chunk_size = chunks.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    n_sources, n_channels = chunks.shape[1:3]

    folded = nn.functional.fold(
        (chunks * windows[:, None, None, :]).permute(1, 2, 3, 0).reshape(
            1, n_sources * n_channels * chunk_size, n_chunks
        ),
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    result[..., start_offset:start_offset + output_length] += folded.view(n_sources, n_channels, output_length)


def _ensure_source_dim(x, chunk_batch):
    return x.unsqueeze(1) if x.ndim == chunk_batch.ndim else x


def _fit_tensor_length(x, length):
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def _autocast(device, enabled):
    device_type = torch.device(device).type
    if enabled and device_type in ('cuda', 'mps'):
        return torch.amp.autocast(device_type, dtype=torch.float16)
    return nullcontext()


def _use_amp(config):
    value = config.inference.get('use_amp', None)
    return config.training.get('use_amp', True) if value is None else bool(value)


def _source_names(config):
    return config.training.instruments if config.training.target_instrument is None else [config.training.target_instrument]


def _source_count(config):
    return len(_source_names(config))


def _sources_to_dict(config, estimated_sources):
    return {k: v for k, v in zip(_source_names(config), estimated_sources)}


def _prepare_mix_for_chunks(mix, border):
    length_init = mix.shape[-1]
    mix = mix.unsqueeze(0) if mix.ndim == 1 else mix
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
    return mix, length_init


def _init_overlap_buffers(config, mix, device, use_fast_path):
    req_shape = (_source_count(config),) + tuple(mix.shape)
    result_device = device if use_fast_path else 'cpu'
    counter_shape = (1, 1, mix.shape[1]) if use_fast_path else req_shape
    result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
    counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
    return result, counter


def _model_mix(mix, device):
    return mix.to(device) if torch.device(device).type != 'cpu' else mix


def _run_model_chunk(model, arr, chunk_size):
    return _fit_tensor_length(_ensure_source_dim(model(arr), arr).float(), chunk_size)


def _extract_chunk(mix, start, chunk_size):
    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start:start + chunk_size]
    if length == chunk_size:
        return part, length
    if length > chunk_size // 2 + 1:
        part = nn.functional.pad(part, (0, chunk_size - length), mode='reflect')
    else:
        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode='constant', value=0)
    return part, length


def _add_weighted_chunk(result, counter, chunk, window, start, length):
    device = result.device
    window = window.to(device=device, dtype=torch.float32)[:length]
    result[..., start:start + length] += chunk[..., :length].to(device=device, dtype=torch.float32) * window
    counter[..., start:start + length] += window


def _run_complete_chunks(model, mix, windows, result, counter, chunk_size, step, batch_size, progress_bar):
    n_chunks = _complete_chunk_count(mix.shape[1], chunk_size, step)
    if n_chunks == 0:
        return 0

    n_complete = n_chunks
    if len(windows) > n_chunks:
        n_complete -= n_complete % batch_size
    if n_complete == 0:
        return 0

    inputs = mix.unfold(-1, chunk_size, step).permute(1, 0, 2)[:n_complete]
    fold_windows = torch.stack(windows[:n_complete], dim=0).to(device=result.device, dtype=torch.float32)
    _fold_windows(counter, fold_windows, step)

    for batch_start in range(0, n_complete, batch_size):
        batch_end = min(batch_start + batch_size, n_complete)
        chunks = _run_model_chunk(model, inputs[batch_start:batch_end].contiguous(), chunk_size)
        _fold_chunk_batch(
            result,
            chunks,
            fold_windows[batch_start:batch_end],
            step,
            start_offset=batch_start * step,
        )
        if progress_bar:
            progress_bar.update(step * (batch_end - batch_start))

    return n_complete


def _run_tail_chunks(model, mix, starts, windows, result, counter, chunk_size, step, batch_size, first_chunk, progress_bar):
    for batch_start in range(first_chunk, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch = [(_extract_chunk(mix, starts[idx], chunk_size), idx) for idx in batch_indices]
        batch_data = [chunk for (chunk, _), _ in batch]
        chunks = _run_model_chunk(model, torch.stack(batch_data, dim=0), chunk_size)
        for j, ((_, length), idx) in enumerate(batch):
            start = starts[idx]
            _add_weighted_chunk(result, counter, chunks[j], windows[idx], start, length)

        if progress_bar:
            progress_bar.update(step * len(batch_data))


def _finalize_overlap(result, counter, length_init, border):
    estimated_sources = result / counter
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    if estimated_sources.device.type == 'cuda':
        host = torch.empty(estimated_sources.shape, dtype=estimated_sources.dtype, device='cpu', pin_memory=True)
        host.copy_(estimated_sources, non_blocking=True)
        torch.cuda.current_stream(estimated_sources.device).synchronize()
        estimated_sources = host.numpy()
    else:
        estimated_sources = estimated_sources.cpu().numpy()
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    return estimated_sources


def _mlx_reflect_pad_1d(x, left=0, right=0):
    import mlx.core as mx

    parts = []
    if left > 0:
        parts.append(x[..., 1:left + 1][..., ::-1])
    parts.append(x)
    if right > 0:
        parts.append(x[..., -right - 1:-1][..., ::-1])
    return mx.concatenate(parts, axis=-1)


def _mlx_get_windowing_array(window_size, fade_size):
    import mlx.core as mx

    if fade_size <= 0:
        return mx.ones((window_size,), dtype=mx.float32)
    fadein = mx.linspace(0, 1, fade_size)
    fadeout = mx.linspace(1, 0, fade_size)
    window = mx.ones((window_size,), dtype=mx.float32)
    window = window.at[:fade_size].multiply(fadein)
    window = window.at[-fade_size:].multiply(fadeout)
    return window


def _mlx_build_chunk_plan(total_length, chunk_size, step, fade_size):
    starts = list(range(0, total_length, step))
    normal_window = _mlx_get_windowing_array(chunk_size, fade_size)
    windows = []
    for start in starts:
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            windows.append(normal_window)
            continue
        window = normal_window
        if start == 0 and fade_size > 0:
            window = window.at[:fade_size].add(1 - window[:fade_size])
        if start + length >= total_length and fade_size > 0:
            tail = slice(max(0, length - fade_size), length)
            window = window.at[tail].add(1 - window[tail])
        windows.append(window)
    return starts, windows


def _mlx_prepare_mix_for_chunks(mix, border):
    import mlx.core as mx

    length_init = mix.shape[-1]
    mix = mx.array(np.asarray(mix, dtype=np.float32))
    if mix.ndim == 1:
        mix = mix[None, :]
    if length_init > 2 * border and border > 0:
        mix = _mlx_reflect_pad_1d(mix, border, border)
    return mix, length_init


def _mlx_extract_chunk(mix, start, chunk_size):
    import mlx.core as mx

    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start:start + chunk_size]
    if length == chunk_size:
        return part, length
    pad = chunk_size - length
    if length > chunk_size // 2 + 1:
        part = _mlx_reflect_pad_1d(part, right=pad)
    else:
        part = mx.pad(part, [(0, 0), (0, pad)])
    return part, length


def _mlx_fit_length(x, length):
    import mlx.core as mx

    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, length - x.shape[-1])])
    return x


def _mlx_run_model_chunk(model, arr, chunk_size):
    y = model.mlx_forward_mx(arr)
    if y.ndim == arr.ndim:
        y = y[:, None]
    return _mlx_fit_length(y, chunk_size)


def _mlx_add_weighted_chunk(result, counter, chunk, window, start, length):
    window = window[:length].astype(result.dtype)
    weighted = chunk[..., :length].astype(result.dtype) * window
    target = slice(start, start + length)
    for source_idx in range(result.shape[0]):
        for channel_idx in range(result.shape[1]):
            result = result.at[source_idx, channel_idx, target].add(weighted[source_idx, channel_idx])
            counter = counter.at[source_idx, channel_idx, target].add(window)
    return result, counter


def _mlx_finalize_overlap(result, counter, length_init, border):
    import mlx.core as mx

    estimated_sources = result / counter
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    estimated_sources = np.array(estimated_sources, copy=False)
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    return estimated_sources


def _can_demix_mlx_full(model, device):
    return (
        torch.device(device).type == "mps"
        and getattr(model, "mps_model_backend", None) == "mlx_full"
        and hasattr(model, "mps_model_compute_dtype")
        and hasattr(model, "mlx_forward_mx")
    )


def demix_track_mlx_full(config, model, mix, device, pbar=False):
    import mlx.core as mx

    C = config.audio.chunk_size
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size

    mix, length_init = _mlx_prepare_mix_for_chunks(mix, border)
    starts, windows = _mlx_build_chunk_plan(mix.shape[1], C, step, fade_size)
    result = mx.zeros((_source_count(config), mix.shape[0], mix.shape[1]), dtype=mx.float32)
    counter = mx.zeros_like(result)
    progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch = [(_mlx_extract_chunk(mix, starts[idx], C), idx) for idx in batch_indices]
        chunks = _mlx_run_model_chunk(model, mx.stack([chunk for (chunk, _), _ in batch], axis=0), C)
        for j, ((_, length), idx) in enumerate(batch):
            result, counter = _mlx_add_weighted_chunk(result, counter, chunks[j], windows[idx], starts[idx], length)
        mx.eval(result, counter)
        if progress_bar:
            progress_bar.update(step * len(batch))

    if progress_bar:
        progress_bar.close()
    return _sources_to_dict(config, _mlx_finalize_overlap(result, counter, length_init, border))


demix_track_mlx_roformer = demix_track_mlx_full


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size

    mix, length_init = _prepare_mix_for_chunks(mix, border)
    chunk_starts, chunk_windows = _build_chunk_plan(mix.shape[1], C, step, fade_size)
    device_type = torch.device(device).type
    use_complete_fast_path = device_type in ('cuda', 'cpu')
    mix_device = _model_mix(mix, device)

    with _autocast(device, _use_amp(config)):
        with torch.inference_mode():
            result, counter = _init_overlap_buffers(config, mix, device, use_complete_fast_path)
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            complete_chunks = 0
            if use_complete_fast_path:
                complete_chunks = _run_complete_chunks(
                    model, mix_device, chunk_windows, result, counter, C, step, batch_size, progress_bar
                )

            _run_tail_chunks(
                model, mix_device, chunk_starts, chunk_windows, result, counter, C, step, batch_size, complete_chunks, progress_bar
            )


            if progress_bar:
                progress_bar.close()

            estimated_sources = _finalize_overlap(result, counter, length_init, border)

    return _sources_to_dict(config, estimated_sources)


def demix_track_demucs(config, model, mix, device, pbar=False):
    if _can_demix_mlx_full(model, device):
        return demix_track_mlx_full(config, model, mix.cpu().numpy(), device, pbar=pbar)

    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    batch_size = config.inference.batch_size
    step = _get_inference_step(config, C)

    with _autocast(device, _use_amp(config)):
        with torch.inference_mode():
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step


                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    for j, (start, l) in enumerate(batch_locations):
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data, batch_locations = [], []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = (result / counter).cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    return dict(zip(config.training.instruments, estimated_sources)) if S > 1 else estimated_sources

def demix(config, model, mix: NDArray, device, pbar=False, model_type: str = None) -> Dict[str, NDArray]:
    if _can_demix_mlx_full(model, device):
        return demix_track_mlx_full(config, model, mix, device, pbar=pbar)
    mix = torch.tensor(mix, dtype=torch.float32)
    if model_type == 'htdemucs':
        return demix_track_demucs(config, model, mix, device, pbar=pbar)
    return demix_track(config, model, mix, device, pbar=pbar)
