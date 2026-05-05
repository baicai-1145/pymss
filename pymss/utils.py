import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict

from .config import load_config
from .logger import get_separation_logger
logger = get_separation_logger()


def get_model_from_config(model_type, config_path):
    config = load_config(config_path)

    if model_type == 'mdx23c':
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from .modules.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'mel_band_roformer':
        from .modules.bs_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        from .modules.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer_hyperace':
        from .modules.bs_roformer import BSRoformerHyperACE
        model = BSRoformerHyperACE(
            **dict(config.model)
        )
    elif model_type == 'bandit':
        from .modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(
            **config.model
        )
    elif model_type == 'bandit_v2':
        from .modules.bandit_v2.bandit import Bandit
        model = Bandit(
            **config.kwargs
        )
    elif model_type == 'scnet':
        from .modules.scnet import SCNet
        model = SCNet(
            **config.model
        )
    elif model_type == 'apollo':
        from .modules.look2hear.apollo import Apollo
        model = Apollo(**config.model)
    else:
        raise ValueError(f"Model type {model_type} not supported")
        # model = None

    return model, config

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
    windows = []
    normal_window = _getWindowingArray(chunk_size, fade_size)

    for start in starts:
        length = min(chunk_size, total_length - start)
        window = normal_window
        if start == 0 or start + length >= total_length:
            window = normal_window.clone()
            if start == 0:
                window[:fade_size] = 1
            if start + length >= total_length:
                fade_start = max(0, length - fade_size)
                window[fade_start:length] = 1
        windows.append(window)

    return starts, windows


def _get_inference_step(config, chunk_size):
    overlap_size = int(config.inference.get('overlap_size', chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("inference.overlap_size must be >= 0 and < audio.chunk_size")
    return chunk_size - overlap_size


def _complete_chunk_count(total_length, chunk_size, step):
    if total_length < chunk_size:
        return 0
    return (total_length - chunk_size) // step + 1


def _fold_windows(counter, windows, step, start_offset=0):
    n_chunks = windows.shape[0]
    if n_chunks == 0:
        return

    chunk_size = windows.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    counter_columns = windows.transpose(0, 1).unsqueeze(0)
    folded_counter = nn.functional.fold(
        counter_columns,
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

    weighted = chunks * windows[:, None, None, :]
    columns = weighted.permute(1, 2, 3, 0).reshape(1, n_sources * n_channels * chunk_size, n_chunks)
    folded = nn.functional.fold(
        columns,
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    result[..., start_offset:start_offset + output_length] += folded.view(n_sources, n_channels, output_length)


def _ensure_source_dim(x, chunk_batch):
    if x.ndim == chunk_batch.ndim:
        return x.unsqueeze(1)
    return x


def _fit_tensor_length(x, length):
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def _get_compiled_chunk_model(model, sample_batch, mode):
    target = _get_model_target(model)
    cache = target.__dict__.setdefault('_pymss_compiled_chunk_models', {})
    key = (
        tuple(sample_batch.shape),
        sample_batch.device.type,
        sample_batch.device.index,
        sample_batch.dtype,
        mode,
    )
    compiled = cache.get(key)
    if compiled is None:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        cache[key] = compiled
    return compiled


def _run_model_chunk_batch(model, arr, expected_batch_size=None):
    target = _get_model_target(model)
    compile_scope = target.__dict__.get('_pymss_torch_compile_scope')
    compile_enabled = target.__dict__.get('_pymss_torch_compile_enabled', False)
    use_fixed_compile = (
        expected_batch_size is not None
        and arr.shape[0] == expected_batch_size
        and compile_enabled
    )
    if (
        use_fixed_compile
        and compile_scope == 'chunk'
    ):
        compiled_model = _get_compiled_chunk_model(
            model,
            arr,
            target.__dict__.get('_pymss_torch_compile_mode', 'default'),
        )
        return compiled_model(arr)

    if compile_enabled and compile_scope == 'core':
        target.__dict__['_pymss_compile_core_this_call'] = use_fixed_compile
        try:
            return model(arr)
        finally:
            target.__dict__.pop('_pymss_compile_core_this_call', None)

    return model(arr)


def _set_model_inference_option(model, name, value):
    target = model.module if hasattr(model, 'module') else model
    if hasattr(target, name):
        setattr(target, name, value)


def _get_model_target(model):
    return model.module if hasattr(model, 'module') else model


def _set_model_stft_hop_length(model, hop_length):
    target = _get_model_target(model)
    if hop_length is None or not hasattr(target, 'stft_kwargs'):
        return None

    old_hop_length = target.stft_kwargs.get('hop_length', None)
    target.stft_kwargs['hop_length'] = int(hop_length)
    return old_hop_length


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size
    old_stft_hop_length = _set_model_stft_hop_length(
        model,
        config.inference.get('stft_hop_length', None)
    )

    try:
        length_init = mix.shape[-1]

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            if mix.ndim == 1:
                mix = mix.unsqueeze(0)  # [1, length]
            mix = nn.functional.pad(mix, (border, border), mode='reflect')

        # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
        chunk_starts, chunk_windows = _build_chunk_plan(mix.shape[1], C, step, fade_size)

        with torch.cuda.amp.autocast(enabled=config.training.get('use_amp', True)):
            with torch.inference_mode():
                if config.training.target_instrument is not None:
                    req_shape = (1, ) + tuple(mix.shape)
                else:
                    req_shape = (len(config.training.instruments),) + tuple(mix.shape)

                device_type = torch.device(device).type
                is_cuda = device_type == 'cuda'
                use_complete_fast_path = device_type in ('cuda', 'cpu')
                result_device = device if use_complete_fast_path else 'cpu'
                counter_shape = (1, 1, mix.shape[1]) if use_complete_fast_path else req_shape
                result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
                counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
                progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None
                mix_device = mix.to(device) if is_cuda else mix
                _set_model_inference_option(
                    model,
                    'inference_layer_skip',
                    config.inference.get('layer_skip', None)
                )
                _set_model_inference_option(
                    model,
                    'inference_mask_mode',
                    config.inference.get('mask_mode', 'full')
                )
                _set_model_inference_option(
                    model,
                    'inference_time_layer_skip',
                    config.inference.get('time_layer_skip', None)
                )
                _set_model_inference_option(
                    model,
                    'inference_freq_layer_skip',
                    config.inference.get('freq_layer_skip', None)
                )
                _set_model_inference_option(
                    model,
                    'inference_grouped_band_ops',
                    config.inference.get('grouped_band_ops', True)
                )
                _set_model_inference_option(
                    model,
                    'inference_rmsnorm_fp32',
                    config.inference.get('rmsnorm_fp32', True)
                )

                complete_chunks = 0
                if use_complete_fast_path:
                    complete_chunks = _complete_chunk_count(mix.shape[1], C, step)
                    if complete_chunks:
                        full_inputs = mix_device.unfold(-1, C, step).permute(1, 0, 2)[:complete_chunks]
                        full_windows = torch.stack(chunk_windows[:complete_chunks], dim=0).to(
                            device=device,
                            dtype=torch.float32,
                        )
                        _fold_windows(counter, full_windows, step)

                        for batch_start in range(0, complete_chunks, batch_size):
                            batch_end = min(batch_start + batch_size, complete_chunks)
                            arr = full_inputs[batch_start:batch_end].contiguous()
                            x = _ensure_source_dim(_run_model_chunk_batch(model, arr, batch_size), arr).float()
                            x = _fit_tensor_length(x, C)
                            _fold_chunk_batch(
                                result,
                                x,
                                full_windows[batch_start:batch_end],
                                step,
                                start_offset=batch_start * step,
                            )

                            if progress_bar:
                                progress_bar.update(step * (batch_end - batch_start))

                        del full_inputs, full_windows

                for batch_start in range(complete_chunks, len(chunk_starts), batch_size):
                    batch_indices = range(batch_start, min(batch_start + batch_size, len(chunk_starts)))
                    batch_data = []

                    for idx in batch_indices:
                        start = chunk_starts[idx]
                        length = min(C, mix.shape[1] - start)
                        part = mix_device[:, start:start + C]
                        if length < C:
                            if length > C // 2 + 1:
                                part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                            else:
                                part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                        batch_data.append(part)

                    arr = torch.stack(batch_data, dim=0)
                    x = _ensure_source_dim(_run_model_chunk_batch(model, arr), arr)
                    x = _fit_tensor_length(x, C)

                    for j, idx in enumerate(batch_indices):
                        start = chunk_starts[idx]
                        length = min(C, mix.shape[1] - start)
                        if is_cuda:
                            window = chunk_windows[idx].to(device=device, dtype=torch.float32)
                            result[..., start:start+length] += x[j][..., :length].float() * window[..., :length]
                            counter[..., start:start+length] += window[..., :length]
                        else:
                            window = chunk_windows[idx]
                            result[..., start:start+length] += x[j][..., :length].cpu() * window[..., :length]
                            counter[..., start:start+length] += window[..., :length]

                    if progress_bar:
                        progress_bar.update(step * len(batch_data))

                if progress_bar:
                    progress_bar.close()

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    # Remove pad
                    estimated_sources = estimated_sources[..., border:-border]

        if config.training.target_instrument is None:
            return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}
    finally:
        if old_stft_hop_length is not None:
            _get_model_target(model).stft_kwargs['hop_length'] = old_stft_hop_length


def demix_track_demucs(config, model, mix, device, pbar=False):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    batch_size = config.inference.batch_size
    step = _get_inference_step(config, C)
    # logger.info(S, C, step, mix.shape, mix.device)

    with torch.cuda.amp.autocast(enabled=config.training.get('use_amp', True)):
        with torch.inference_mode():
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                # logger.info(i, i + C, mix.shape[1])
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
                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return estimated_sources

def demix(config, model, mix: NDArray, device, pbar=False, model_type: str = None) -> Dict[str, NDArray]:
    mix = torch.tensor(mix, dtype=torch.float32)
    if model_type == 'htdemucs':
        return demix_track_demucs(config, model, mix, device, pbar=pbar)
    else:
        return demix_track(config, model, mix, device, pbar=pbar)
