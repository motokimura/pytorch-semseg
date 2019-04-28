# reference:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import OctConv2d, OctConvPool2d, OctConvUpsample, OctConvBatchNorm2d


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, alphas1, alphas2):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            OctConv2d(in_ch, out_ch, 3, alphas=alphas1),
            OctConvBatchNorm2d(out_ch, alpha),
            nn.ReLU(inplace=True),
            OctConv2d(out_ch, out_ch, 3, alphas=alphas2),
            OctConvBatchNorm2d(out_ch, alpha),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(inconv, self).__init__()
        alphas1 = (0.0, alpha)
        alphas2 = (alpha, alpha)
        self.conv = double_conv(in_ch, out_ch, alphas1, alphas2)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(down, self).__init__()
        alphas = (alpha, alpha)
        self.mpconv = nn.Sequential(
            OctConvPool2d(in_ch, kernel_size=2, stride=2, mode='max'),
            double_conv(in_ch, out_ch, alphas, alphas)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(up, self).__init__()
        self.up = OctConvUpsample(in_ch // 2, scale_factor=2, alpha=alpha)
        alphas = (alpha, alpha)
        self.conv = double_conv(in_ch, out_ch, alphas, alphas)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha):
        super(outconv, self).__init__()
        alphas = (alpha, 0.0)
        self.conv = OctConv2d(in_ch, out_ch, 1, alphas=alphas)

    def forward(self, x):
        x = self.conv(x)
        return x
