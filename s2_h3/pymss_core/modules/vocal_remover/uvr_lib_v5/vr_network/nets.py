import torch
import torch.nn.functional as F

from pymss_core.npu_conv import replicate_pad_bins
from torch import nn
from . import layers
class BaseASPPNet(nn.Module):
    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super().__init__()
        self.nn_architecture = nn_architecture
        enc = lambda i, o: layers.Encoder(i, o, 3, 2, 1)
        self.enc1, self.enc2, self.enc3, self.enc4 = enc(nin, ch), enc(ch, ch * 2), enc(ch * 2, ch * 4), enc(ch * 4, ch * 8)
        if nn_architecture == 129605:
            self.enc5 = enc(ch * 8, ch * 16)
            self.aspp = layers.ASPPModule(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = layers.Decoder(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = layers.ASPPModule(nn_architecture, ch * 8, ch * 16, dilations)
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)
    def forward(self, input_tensor):
        hidden_state, e1 = self.enc1(input_tensor)
        hidden_state, e2 = self.enc2(hidden_state)
        hidden_state, e3 = self.enc3(hidden_state)
        hidden_state, e4 = self.enc4(hidden_state)
        if self.nn_architecture == 129605: hidden_state, e5 = self.enc5(hidden_state); hidden_state = self.dec5(self.aspp(hidden_state), e5)
        else: hidden_state = self.aspp(hidden_state)
        hidden_state = self.dec4(hidden_state, e4)
        hidden_state = self.dec3(hidden_state, e3)
        hidden_state = self.dec2(hidden_state, e2)
        return self.dec1(hidden_state, e1)
def determine_model_capacity(n_fft_bins, nn_architecture): ch = {31191: 16, 33966: 16, 123821: 32, 123812: 32, 537238: 64, 537227: 64}[nn_architecture]; caps = [(2, ch), (2, ch), (ch + 2, ch // 2, 1, 1, 0), (ch // 2, ch), (2 * ch + 2, ch, 1, 1, 0), (ch, 2 * ch), (2 * ch, 2, 1), (ch, 2, 1), (ch, 2, 1)]; return CascadedASPPNet(n_fft_bins, caps, nn_architecture)
class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super().__init__()
        m = model_capacity_data
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *m[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *m[1])
        self.stg2_bridge = layers.Conv2DBNActiv(*m[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *m[3])
        self.stg3_bridge = layers.Conv2DBNActiv(*m[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *m[5])
        self.out, self.aux1_out, self.aux2_out = nn.Conv2d(*m[6], bias=False), nn.Conv2d(*m[7], bias=False), nn.Conv2d(*m[8], bias=False)
        self.max_bin, self.output_bin, self.offset = n_fft // 2, n_fft // 2 + 1, 128
    def forward(self, input_tensor):
        mix = input_tensor.detach()
        input_tensor = input_tensor[:, :, : self.max_bin]
        bandwidth = input_tensor.shape[2] // 2
        aux1 = torch.cat([self.stg1_low_band_net(input_tensor[:, :, :bandwidth]), self.stg1_high_band_net(input_tensor[:, :, bandwidth:])], dim=2)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(torch.cat([input_tensor, aux1], dim=1)))
        hidden_state = self.stg3_full_band_net(self.stg3_bridge(torch.cat([input_tensor, aux1, aux2], dim=1)))
        mask = torch.sigmoid(self.out(hidden_state))
        # F.pad(mode="replicate") is aten::replication_pad2d, which SUPA fails on.
        mask = replicate_pad_bins(mask, self.output_bin)
        if self.training: pad = lambda t: F.pad(t, (0, 0, 0, self.output_bin - t.shape[2]), mode='replicate'); return (mask * mix, pad(torch.sigmoid(self.aux1_out(aux1))) * mix, pad(torch.sigmoid(self.aux2_out(aux2))) * mix)
        return mask
    def predict_mask(self, input_tensor): mask = self.forward(input_tensor); return mask[:, :, :, self.offset:-self.offset] if self.offset > 0 else mask