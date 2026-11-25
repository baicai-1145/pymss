import torch
from torch import nn
from torch.nn.modules import rnn as _rnn
class TimeFrequencyModellingModule(nn.Module):
    pass
class ResidualRNN(nn.Module):
    def __init__(self, emb_dim, rnn_dim, bidirectional=True, rnn_type="LSTM", use_batch_trick=True, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm, self.use_batch_trick = use_layer_norm, use_batch_trick
        self.norm = nn.LayerNorm(emb_dim) if use_layer_norm else nn.GroupNorm(num_groups=emb_dim, num_channels=emb_dim)
        self.rnn = _rnn.__dict__[rnn_type](input_size=emb_dim, hidden_size=rnn_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(rnn_dim * (2 if bidirectional else 1), emb_dim)
    def forward(self, z0):
        z = self.norm(z0) if self.use_layer_norm else self.norm(z0.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        b, n_uncrossed, n_across, emb_dim = z.shape
        if self.use_batch_trick: z = self.rnn(z.reshape(b * n_uncrossed, n_across, emb_dim))[0].reshape(b, n_uncrossed, n_across, -1)
        else: z = torch.stack([self.rnn(z[:, i, :, :])[0] for i in range(n_uncrossed)], dim=1)
        return self.fc(z) + z0
class Transpose(nn.Module):
    def __init__(self, dim0, dim1): super().__init__(); self.dim0, self.dim1 = dim0, dim1
    def forward(self, z): return z.transpose(self.dim0, self.dim1)
class SeqBandModellingModule(TimeFrequencyModellingModule):
    # three layouts: parallel (t/f ModuleList pairs), Sequential(rrn, Transpose)*n, or plain ModuleList with transpose
    def __init__(self, n_modules=12, emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="LSTM", parallel_mode=False, sequential_transpose=False, checkpoint_segments=None):
        super().__init__()
        self.n_modules, self.parallel_mode, self.checkpoint_segments = n_modules, parallel_mode, checkpoint_segments
        rrn = lambda: ResidualRNN(emb_dim, rnn_dim, bidirectional, rnn_type)
        if parallel_mode:
            self.seqband = nn.ModuleList([nn.ModuleList([rrn(), rrn()]) for _ in range(n_modules)])
        elif sequential_transpose: self.seqband = nn.Sequential(*[m for _ in range(2 * n_modules) for m in (rrn(), Transpose(1, 2))])
        else: self.seqband = nn.ModuleList([rrn() for _ in range(2 * n_modules)])
    def forward(self, z):
        from torch.utils.checkpoint import checkpoint_sequential
        if self.parallel_mode:
            for sbm_t, sbm_f in self.seqband: z = sbm_t(z) + sbm_f(z.transpose(1, 2)).transpose(1, 2)
            return z
        if isinstance(self.seqband, nn.Sequential):
            if self.checkpoint_segments and self.training: return checkpoint_sequential(self.seqband, self.checkpoint_segments, z, use_reentrant=False)
            return self.seqband(z)
        for sbm in self.seqband: z = sbm(z).transpose(1, 2)
        return z
class _SeqBandModellingPreset(SeqBandModellingModule):
    def __init__(self, n_modules=12, emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="LSTM", parallel_mode=False): super().__init__(n_modules=n_modules, emb_dim=emb_dim, rnn_dim=rnn_dim, bidirectional=bidirectional, rnn_type=rnn_type, parallel_mode=parallel_mode, **self._preset_runtime_options(n_modules, parallel_mode))
    @staticmethod
    def _preset_runtime_options(n_modules, parallel_mode): return {"sequential_transpose": False, "checkpoint_segments": None}