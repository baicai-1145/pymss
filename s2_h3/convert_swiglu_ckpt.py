"""Convert an H4s training checkpoint to the `npu_swiglu` parameter convention (one-off, offline).

`entry.prepare_swiglu` swaps the two output halves of every `MSC.proj_in` so `torch_npu.npu_swiglu` computes the same
function as the original `u * silu(v)` gate (measured 274.2 -> 234.3 ms/step on the 950PR).  The swap permutes the
parameters, so a run that resumes from a checkpoint written *before* the swap must permute the same three things or it
silently trains a different function:

  1. `state_dict` entries for `*.proj_in.weight` / `*.proj_in.bias` (the model weights),
  2. the Adam first/second moments for those same parameters (`optimizer_states[0]['state']`),
  3. the EMA shadow (`state_dict['ema_model.*']`), which is loaded with `strict=True` a few steps later.

The converted file is tagged `swiglu_permuted: True`; `entry.install_resume_hook` refuses to load a checkpoint whose tag
disagrees with the model, so a mismatched run fails loudly instead of corrupting.  Exported eval weights stay in the
original convention -- `pymsst.export.checkpoint` un-swaps them on the way out.

Usage (on the Ascend box, from `s2_h3/`):

    python convert_swiglu_ckpt.py runs/H4s/resume-5000.ckpt runs/H4s/resume-5000-swiglu.ckpt --arm H4s
    python convert_swiglu_ckpt.py runs/H4s/resume-5000.ckpt --dry-run          # report only

Then resume with `S2_SWIGLU_PREPERMUTED=1` (and `S2_NPU_SWIGLU` left at its default) pointing at the new file.
"""
import argparse
import os
import sys

import torch

PREFIXES = ('ema_model.module.', 'ema_model.', 'model.', 'module.')
MOMENT_KEYS = ('exp_avg', 'exp_avg_sq', 'momentum')


def base_key(key):
    """Strip the wrapper prefixes a Lightning state_dict may carry, so it can be matched against `named_parameters()`."""
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def swap_rows_(tensor):
    """Swap the two halves of dim 0 in place.  Exact: a copy of one half is staged before either write."""
    half = tensor.shape[0] // 2
    staged = tensor[:half].detach().clone()
    with torch.no_grad():
        tensor[:half].copy_(tensor[half:])
        tensor[half:].copy_(staged)
    return tensor


def swiglu_parameter_names(model):
    """The exact `proj_in` parameters `prepare_swiglu` swaps, keyed without the Lightning wrapper prefix."""
    return {name for name, _ in model.named_parameters()
            if name.endswith('.proj_in.weight') or name.endswith('.proj_in.bias')}


def convert(checkpoint, model, dry_run=False):
    names = swiglu_parameter_names(model)
    assert names, 'no MSC proj_in parameters found -- wrong arm?'
    if checkpoint.get('swiglu_permuted'):
        raise SystemExit('already converted (swiglu_permuted=True); refusing to swap twice')

    state = checkpoint['state_dict']
    swapped_state = [k for k in state if base_key(k) in names]
    assert len(swapped_state) == len(names) * 2 or len(swapped_state) >= len(names), \
        (len(swapped_state), len(names), sorted(names)[:4])
    for key in swapped_state:
        swap_rows_(state[key])

    moments = 0
    if checkpoint.get('optimizer_states'):
        opt_state = checkpoint['optimizer_states'][0]
        params = list(model.parameters())
        param_names = [n for n, _ in model.named_parameters()]
        assert len(params) == len(param_names)
        order = opt_state['param_groups'][0]['params']
        assert len(order) == len(params), (len(order), len(params))
        for local, index in enumerate(order):
            name = param_names[local]
            if name not in names:
                continue
            slot = opt_state['state'].get(index)
            if slot is None:
                slot = opt_state['state'].get(str(index))
            assert slot is not None, ('no optimizer state for', name, index)
            for key in MOMENT_KEYS:
                if key in slot and slot[key] is not None:
                    assert slot[key].shape == params[local].shape, (name, slot[key].shape, params[local].shape)
                    swap_rows_(slot[key])
                    moments += 1

    ema = [k for k in state if k.startswith('ema_model.') and base_key(k) in names]
    for key in ema:
        swap_rows_(state[key])

    print('SWIGLU_CONVERT', 'weights', len(swapped_state), 'moments', moments, 'ema', len(ema),
          'step', checkpoint.get('global_step'), flush=True)
    checkpoint['swiglu_permuted'] = True
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst', nargs='?')
    parser.add_argument('--arm', default=os.environ.get('ARM', 'H4s'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    from entry import build
    model = build(args.arm)
    checkpoint = torch.load(args.src, map_location='cpu', weights_only=False)
    path = sys.argv[0]
    convert(checkpoint, model, dry_run=args.dry_run)
    if args.dry_run:
        print('DRY_RUN done for', args.src, flush=True)
        return 0
    dst = args.dst or args.src.replace('.ckpt', '-swiglu.ckpt')
    torch.save(checkpoint, dst)
    print('SWIGLU_CONVERT_SAVED', dst, os.path.getsize(dst), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
