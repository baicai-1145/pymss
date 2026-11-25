import torch
import torch.nn.functional as F
from torch import nn
def crop_center(h1, h2):
    if h1.size(3) == h2.size(3): return h1
    if h1.size(3) < h2.size(3): raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s = (h1.size(3) - h2.size(3)) // 2
    return h1[:, :, :, s:s + h2.size(3)]
class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU): super().__init__(); self.conv = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False), nn.BatchNorm2d(nout), activ())
    def forward(self, input_tensor): return self.conv(input_tensor)
class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU): super().__init__(); self.conv = nn.Sequential(nn.Conv2d(nin, nin, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=nin, bias=False), nn.Conv2d(nin, nout, kernel_size=1, bias=False), nn.BatchNorm2d(nout), activ())
    def forward(self, input_tensor): return self.conv(input_tensor)
class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU): super().__init__(); self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ); self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)
    def forward(self, input_tensor): skip = self.conv1(input_tensor); return self.conv2(skip), skip
class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False): super().__init__(); self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ); self.dropout = nn.Dropout2d(0.1) if dropout else None
    def forward(self, input_tensor, skip=None):
        x = F.interpolate(input_tensor, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None: x = torch.cat([x, crop_center(skip, x)], dim=1)
        x = self.conv(x)
        return x if self.dropout is None else self.dropout(x)
class ASPPModule(nn.Module):
    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super().__init__()
        sep = lambda d: SeperableConv2DBNActiv(nin, nin, 3, 1, d, d, activ=activ)
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ))
        self.nn_architecture = nn_architecture
        self.six_layer, self.seven_layer = [129605], [537238, 537227, 33966]
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3, self.conv4, self.conv5 = sep(dilations[0]), sep(dilations[1]), sep(dilations[2])
        if self.nn_architecture in self.six_layer:
            self.conv6 = sep(dilations[2]); nin_x = 6
        elif self.nn_architecture in self.seven_layer: self.conv6 = self.conv7 = sep(dilations[2]); nin_x = 7
        else: nin_x = 5
        self.bottleneck = nn.Sequential(Conv2DBNActiv(nin * nin_x, nout, 1, 1, 0, activ=activ), nn.Dropout2d(0.1))
    def forward(self, input_tensor):
        _, _, h, w = input_tensor.size()
        features = [self.conv1(input_tensor).expand(-1, -1, h, -1), self.conv2(input_tensor), self.conv3(input_tensor), self.conv4(input_tensor), self.conv5(input_tensor)]
        if self.nn_architecture in self.six_layer: features.append(self.conv6(input_tensor))
        elif self.nn_architecture in self.seven_layer: features.extend([self.conv6(input_tensor), self.conv7(input_tensor)])
        return self.bottleneck(torch.cat(features, dim=1))