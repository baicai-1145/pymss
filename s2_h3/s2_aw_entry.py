"""AdamW training entry (identical optimizer/scheduler/EMA protocol to the archived s_aw_entry.py)."""
import math
import os
import runpy
import sys

import torch
import pymsst.training.module as tm
from entry import factory, install_resume_hook, ARMS, structure_ok, DEVICE, install_npu_trainer
from pymsst.training.callbacks import ExponentialMovingAverage as EMA


def configure(module):
    assert module.learning_rate == 5e-4 and module.config.training.optimizer == 'adamw'
    opt = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=.01, **tm._optimizer_backend_kwargs(module.optimizer_backend))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40000, gamma=.9)
    return dict(optimizer=opt, lr_scheduler=dict(scheduler=sched, interval='step', frequency=1))


if __name__ == '__main__':
    arm = os.environ['ARM']; assert arm in ARMS, arm
    if DEVICE.type == 'npu':
        install_npu_trainer()

    def factory_with_kernels(*args, **kwargs):
        """Build the model, then hand it to the kernels named in S2_TRITON (TASK §6).  Unset = byte-identical eager path."""
        model, cfg = factory(*args, **kwargs)
        names = [n for n in os.environ.get('S2_TRITON', '').replace(' ', '').split(',') if n]
        if names:
            from triton_npu import install
            print('TRITON_INSTALLED', install(model, names), flush=True)
        return model, cfg

    tm.get_model_from_config = factory_with_kernels
    tm.SeparationLightningModule.configure_optimizers = configure
    if '--resume-checkpoint' in sys.argv:
        install_resume_hook()
    original = EMA.on_train_start

    def check(cb, trainer, module):
        original(cb, trainer, module)
        step = trainer.global_step; opt, = trainer.optimizers; s = trainer.lr_scheduler_configs[0].scheduler
        model = module.model
        assert model.arm == arm and structure_ok(model), ('structure', arm)
        assert isinstance(opt, torch.optim.AdamW) and s.last_epoch == step and int(cb.ema_model.n_averaged) == step
        assert math.isclose(opt.param_groups[0]['lr'], 5e-4 * .9 ** (step // 40000)) and opt.param_groups[0]['weight_decay'] == .01
        print('ADAMW_START_VERIFIED', arm, step, opt.param_groups[0]['lr'], flush=True)

    EMA.on_train_start = check
    runpy.run_module('pymsst', run_name='__main__', alter_sys=True)
