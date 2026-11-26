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
    # Exactly one swap per key.  The model-side and EMA-side names both match `names` after prefix stripping, so they
    # must be partitioned here: an earlier version swapped the EMA keys in both passes and the double swap silently
    # cancelled, leaving the shadow in the original order while the weights were permuted.
    model_keys = [k for k in state if not k.startswith('ema_model.') and base_key(k) in names]
    ema_keys = [k for k in state if k.startswith('ema_model.') and base_key(k) in names]
    assert set(model_keys).isdisjoint(ema_keys)
    assert len(model_keys) == len(names), ('model proj_in keys', len(model_keys), len(names))
    assert len(ema_keys) == len(names), ('ema proj_in keys', len(ema_keys), len(names), sorted(ema_keys)[:2])

    # Model and EMA only diverge by the 5000 steps of averaging, so their distance is a free self-check: a double swap
    # on either side multiplies it by ~10, which is how the EMA bug above was caught.
    before = _pair_gap(state, model_keys, ema_keys)
    for key in model_keys + ema_keys:
        swap_rows_(state[key])
    after = _pair_gap(state, model_keys, ema_keys)
    assert after <= 3 * before + 1e-6, ('EMA/model gap exploded after conversion -- a key was swapped twice',
                                        before, after)

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

    print('SWIGLU_CONVERT', 'weights', len(model_keys), 'moments', moments, 'ema', len(ema_keys),
          'ema_gap', f'{before:.4f}->{after:.4f}', 'step', checkpoint.get('global_step'), flush=True)
    checkpoint['swiglu_permuted'] = True
    return checkpoint


def _pair_gap(state, model_keys, ema_keys):
    """max |model - ema| over the proj_in weights: ~0.04 at step 5000, ~0.35 if one side was swapped twice."""
    by_base = {base_key(k): k for k in ema_keys}
    gaps = []
    for m_key in model_keys:
        e_key = by_base.get(base_key(m_key))
        if e_key is None:
            continue
        a, b = state[m_key], state[e_key]
        if a.shape != b.shape:
            continue
        gaps.append((a.float() - b.float()).abs().max().item())
    return max(gaps) if gaps else 0.0


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
