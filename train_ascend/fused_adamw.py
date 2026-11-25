"""Fused AdamW on NPU.

`torch_npu.npu_apply_adam_w` implements decoupled weight decay + bias correction + eps in one
accepted aclnn op (Addcdiv/Mul/Add per parameter disappear). It must reproduce
`torch.optim.AdamW` exactly: same weight-decay rule (decay applied to the parameter *before*
the Adam step, not folded into the gradient), same bias correction via beta1_power/beta2_power,
same eps placement (added to sqrt(v_hat) as sqrt(v)/sqrt(bc2) + eps).

    python fused_adamw.py --check      # numerics vs torch.optim.AdamW
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


class FusedAdamW(torch.optim.Optimizer):
    """AdamW with the whole update batched through ``torch._foreach_*``.

    torch.optim.AdamW's own step leaves one ``Addcdiv`` per parameter tensor on this device
    (measured: 60 tensors -> 60 Addcdiv kernels, 78 kernels total), i.e. ~585 kernels in our
    run. Every element of the update is expressible as a list op, so the same arithmetic runs
    as ~7 batched kernels. The maths is the torch.optim.AdamW eager schedule, unchanged:

        p   *= (1 - lr*wd)
        m    = b1*m + (1-b1)*g
        v    = b2*v + (1-b2)*g*g
        p   -= (lr/bc1) * m / (sqrt(v)/sqrt(bc2) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 amsgrad=False, maximize=False):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        if not 0.0 <= eps or not 0.0 <= weight_decay:
            raise ValueError("eps / weight_decay must be non-negative")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                    "amsgrad": amsgrad, "maximize": maximize}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            params, grads, m_list, v_list = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                if group["maximize"]:
                    p.grad = p.grad.neg()
                params.append(p)
                grads.append(p.grad)
                m_list.append(state["exp_avg"])
                v_list.append(state["exp_avg_sq"])
            if not params:
                continue
            step = self.state[params[0]]["step"]
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            # decoupled weight decay, exactly torch.optim.AdamW's `p *= (1 - lr*wd)`
            torch._foreach_mul_(params, 1.0 - lr * wd)
            # m, v
            torch._foreach_mul_(m_list, b1)
            torch._foreach_add_(m_list, grads, alpha=1.0 - b1)
            torch._foreach_mul_(v_list, b2)
            torch._foreach_addcmul_(v_list, grads, grads, value=1.0 - b2)
            # denominator = sqrt(v)/sqrt(bc2) + eps
            denom = torch._foreach_sqrt(v_list)
            torch._foreach_div_(denom, bc2 ** 0.5)
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(params, m_list, denom, value=-lr / bc1)
        return loss


def _make_problem(n_params: int, seed: int = 0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    params = []
    for i in range(n_params):
        shape = (192, 192) if i % 3 else (384, 192)
        params.append(torch.randn(*shape, generator=g, dtype=torch.float32))
    return params


def check(steps: int = 5) -> dict:
    import copy

    import torch_npu  # noqa: F401

    dev = torch.device("npu:0")
    base = _make_problem(24)
    a = [t.to(dev).requires_grad_(True) for t in base]
    b = [t.to(dev).requires_grad_(True) for t in base]
    opt_eager = torch.optim.AdamW(a, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    opt_fused = FusedAdamW(b, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    g = torch.Generator(device="cpu").manual_seed(7)
    rel = []
    for step in range(steps):
        for i, (pa, pb) in enumerate(zip(a, b)):
            grad = torch.randn(pa.shape, generator=g).to(dev) * (0.1 + 0.05 * i)
            pa.grad = grad.clone()
            pb.grad = grad.clone()
        opt_eager.step()
        opt_fused.step()
        d = torch.cat([(pb.detach() - pa.detach()).flatten() for pa, pb in zip(a, b)])
        ref = torch.cat([pa.detach().flatten() for pa in a])
        rel.append(float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()))
    return {"steps": steps, "param_rel_rms_per_step": rel,
            "worst_rel_rms": max(rel), "ok": max(rel) <= 1e-6}


def bench() -> dict:
    import statistics
    import time

    import torch_npu  # noqa: F401

    from graph_capture3 import kernel_count

    dev = torch.device("npu:0")
    out = {}
    for name in ("eager", "foreach", "fused"):
        base = _make_problem(24)
        ps = [t.to(dev).requires_grad_(True) for t in base]
        g = torch.Generator(device="cpu").manual_seed(1)
        for i, p in enumerate(ps):
            p.grad = torch.randn(p.shape, generator=g).to(dev)
        if name == "eager":
            opt = torch.optim.AdamW(ps, lr=5e-4, weight_decay=0.01)
        elif name == "foreach":
            opt = torch.optim.AdamW(ps, lr=5e-4, weight_decay=0.01, foreach=True)
        else:
            opt = FusedAdamW(ps, lr=5e-4, weight_decay=0.01)

        def run(opt=opt):
            opt.step()

        for _ in range(3):
            run()
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            run()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        k, _, top = kernel_count(run, n=3)
        out[name] = {"median_ms": round(statistics.median(ts), 3), "device_kernels": k,
                     "top": list(top)[:3]}
    out["speedup_vs_eager"] = round(out["eager"]["median_ms"] / out["fused"]["median_ms"], 2)
    out["kernel_reduction"] = round(out["eager"]["device_kernels"] / max(out["fused"]["device_kernels"], 1), 2)
    return out


def make_optimizer(model, kind: str = "foreach", lr: float = 5e-4, weight_decay: float = 0.01):
    """Optimizer factory used by train.py / train_long.py.

    ``fused`` = FusedAdamW (torch._foreach_* batched step, exact torch.optim.AdamW maths).
    ``foreach`` = torch.optim.AdamW(foreach=True) - still one Addcdiv per tensor on this device.
    ``eager`` = torch.optim.AdamW default.
    The aclnn route (torch_npu.npu_apply_adam_w) was measured slower (1.729 vs 0.63 ms) and not
    bit-compatible with torch.optim.AdamW (rel 1.4e-04 per step at wd=0), so it is not used.
    """
    if kind in ("fused", "foreach_fused"):
        return FusedAdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=weight_decay)
    if kind == "eager":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                 weight_decay=weight_decay, foreach=False)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                             weight_decay=weight_decay, foreach=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/fused_adamw.json"))
    args = parser.parse_args()
    payload = {}
    if args.check or not args.bench:
        payload["numerics"] = check()
        print(json.dumps(payload["numerics"], indent=1))
    if args.bench or not args.check:
        payload["bench"] = bench()
        print(json.dumps(payload["bench"], indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
