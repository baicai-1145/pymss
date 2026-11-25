import torch
import torch.nn.functional as F
from torch import nn
from .layers import Conv2DBNActiv, crop_center
class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU): super().__init__(); self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ); self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)
    def forward(self, input_tensor): return self.conv2(self.conv1(input_tensor))
class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False): super().__init__(); self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ); self.dropout = nn.Dropout2d(0.1) if dropout else None
    def forward(self, input_tensor, skip=None):
        x = F.interpolate(input_tensor, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None: x = torch.cat([x, crop_center(skip, x)], dim=1)
        x = self.conv1(x)
        return self.dropout(x) if self.dropout is not None else x
class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super().__init__()
        conv = lambda k, p, d: Conv2DBNActiv(nin, nout, k, 1, p, d, activ=activ)
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2, self.conv3, self.conv4, self.conv5 = conv(1, 0, 1), conv(3, dilations[0], dilations[0]), conv(3, dilations[1], dilations[1]), conv(3, dilations[2], dilations[2])
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None
    def forward(self, input_tensor): _, _, h, w = input_tensor.size(); x = self.bottleneck(torch.cat((self.conv1(input_tensor).expand(-1, -1, h, -1), self.conv2(input_tensor), self.conv3(input_tensor), self.conv4(input_tensor), self.conv5(input_tensor)), dim=1)); return self.dropout(x) if self.dropout is not None else x
class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm): super().__init__(); self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0); self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True); self.dense = nn.Sequential(nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU())
    def forward(self, input_tensor): N, _, nbins, nframes = input_tensor.size(); hidden, _ = self.lstm(self.conv(input_tensor)[:, 0].permute(2, 0, 1)); return self.dense(hidden.reshape(-1, hidden.size(-1))).reshape(nframes, N, 1, nbins).permute(1, 2, 3, 0)