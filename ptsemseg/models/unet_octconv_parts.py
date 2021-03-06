# reference:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import OctConv2d, OctConvMaxPool2d, OctConvUpsample, OctConvBatchNorm2d, OctConvReLU


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, alphas1, alphas2):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            OctConv2d(in_ch, out_ch, 3, bias=False, alphas=alphas1),
            OctConvBatchNorm2d(out_ch, alphas1[1]),
            OctConvReLU(out_ch, alphas1[1]),
            OctConv2d(out_ch, out_ch, 3, bias=False, alphas=alphas2),
            OctConvBatchNorm2d(out_ch, alphas2[1]),
            OctConvReLU(out_ch, alphas2[1])
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(inconv, self).__init__()
        alphas1 = (0.0, alpha)
        alphas2 = (alpha, alpha)
        self.conv = double_conv(in_ch, out_ch, alphas1, alphas2)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(down, self).__init__()
        alphas = (alpha, alpha)
        self.mpconv = nn.Sequential(
            OctConvMaxPool2d(in_ch, kernel_size=2, stride=2, alpha=alpha),
            double_conv(in_ch, out_ch, alphas, alphas)
        )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(up, self).__init__()
        self.up = OctConvUpsample(in_ch // 2, scale_factor=2, mode='bilinear', alpha=alpha)
        alphas = (alpha, alpha)
        self.conv = double_conv(in_ch, out_ch, alphas, alphas)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # XXX: error occurs when alpha = 0 or 1
        hf = torch.cat([x2[0], x1[0]], dim=1)
        lf = torch.cat([x2[1], x1[1]], dim=1)
        return self.conv((hf, lf))


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(outconv, self).__init__()
        alphas = (alpha, 0.0)
        self.conv = OctConv2d(in_ch, out_ch, 1, bias=True, alphas=alphas)

    def forward(self, x):
        return self.conv(x)
