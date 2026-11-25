import torch
import torch.nn.functional as F

from pymss_core.npu_conv import replicate_pad_bins
from torch import nn

from . import layers_new as layers


class BaseNet(nn.Module):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super().__init__()
        e = lambda i, o: layers.Encoder(i, o, 3, 2, 1)
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2, self.enc3, self.enc4, self.enc5 = e(nout, nout * 2), e(nout * 2, nout * 4), e(nout * 4, nout * 6), e(nout * 6, nout * 8)
        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout, 3, 1, 1)
    def forward(self, input_tensor):
        e1 = self.enc1(input_tensor); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        bottleneck = self.dec4(self.aspp(e5), e4)
        bottleneck = self.dec3(bottleneck, e3)
        bottleneck = self.dec2(bottleneck, e2)
        return self.dec1(torch.cat([bottleneck, self.lstm_dec2(bottleneck)], dim=1), e1)
class CascadedNet(nn.Module):
    def __init__(self, n_fft, nn_arch_size=51000, nout=32, nout_lstm=128):
        super().__init__()
        self.max_bin, self.output_bin = n_fft // 2, n_fft // 2 + 1
        self.nin_lstm, self.offset = self.max_bin // 2, 64
        nout = 64 if nn_arch_size == 218409 else nout
        bn = lambda nin, nout_, nl: BaseNet(nin, nout_, self.nin_lstm // 2, nl)
        self.stg1_low_band_net = nn.Sequential(bn(2, nout // 2, nout_lstm), layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0))
        self.stg1_high_band_net = bn(2, nout // 4, nout_lstm // 2)
        self.stg2_low_band_net = nn.Sequential(bn(nout // 4 + 2, nout, nout_lstm), layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0))
        self.stg2_high_band_net = bn(nout // 4 + 2, nout // 2, nout_lstm // 2)
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)
    def forward(self, input_tensor):
        input_tensor = input_tensor[:, :, : self.max_bin]
        bandw = input_tensor.size()[2] // 2
        l1_in, h1_in = input_tensor[:, :, :bandw], input_tensor[:, :, bandw:]
        l1, h1 = self.stg1_low_band_net(l1_in), self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)
        l2, h2 = self.stg2_low_band_net(torch.cat([l1_in, l1], dim=1)), self.stg2_high_band_net(torch.cat([h1_in, h1], dim=1))
        aux2 = torch.cat([l2, h2], dim=2)
        f3 = self.stg3_full_band_net(torch.cat([input_tensor, aux1, aux2], dim=1))
        mask = torch.sigmoid(self.out(f3))
        # F.pad(mode="replicate") is aten::replication_pad2d, which SUPA fails on.
        mask = replicate_pad_bins(mask, self.output_bin)
        if self.training: aux = torch.sigmoid(self.aux_out(torch.cat([aux1, aux2], dim=1))); aux = F.pad(aux, (0, 0, 0, self.output_bin - aux.shape[2]), mode='replicate'); return (mask, aux)
        return mask
    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)
        if self.offset > 0:
            mask = mask[:, :, :, self.offset:-self.offset]
            assert mask.size()[3] > 0
        return mask
    def predict(self, input_tensor):
        pred_mag = input_tensor * self.forward(input_tensor)
        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset:-self.offset]
            assert pred_mag.size()[3] > 0
        return pred_mag
