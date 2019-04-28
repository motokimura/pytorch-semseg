# an unofficial implementation for OctConv. 
# reference:
# https://github.com/iacolippo/octconv-pytorch/blob/master/octconv.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OctConv2d(nn.Module):
    """OctConv proposed in:
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.
    paper link: https://arxiv.org/abs/1904.05049
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alphas=(0.5, 0.5)):
        super(OctConv2d, self).__init__()

        if stride != 1:
            raise NotImplementedError()
            # FIXME: correctly implement strided OctConv.
            # references:
            # https://github.com/iacolippo/octconv-pytorch/issues/3
            # https://github.com/terrychenism/OctaveConv/issues/4

        alpha_in, alpha_out = alphas
        assert (0 <= alpha_in <= 1) and (0 <= alpha_in <= 1), "Alphas must be in interval [0, 1]"
        # input channels
        self.ch_in_lf = int(alpha_in * in_channels)
        self.ch_in_hf = in_channels - self.ch_in_lf
        # output channels
        self.ch_out_lf = int(alpha_out * out_channels)
        self.ch_out_hf = out_channels - self.ch_out_lf

        # filters
        self.wHtoH, self.wHtoL, self.wLtoH, self.wLtoL = None, None, None, None
        if not (self.ch_out_hf == 0 or self.ch_in_hf == 0):
            self.wHtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_hf, kernel_size, kernel_size))
        if not (self.ch_out_lf == 0 or self.ch_in_hf == 0):
            self.wHtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_hf, kernel_size, kernel_size))
        if not (self.ch_out_hf == 0 or self.ch_in_lf == 0):
            self.wLtoH = nn.Parameter(torch.randn(self.ch_out_hf, self.ch_in_lf, kernel_size, kernel_size))
        if not (self.ch_out_lf == 0 or self.ch_in_lf == 0):
            self.wLtoL = nn.Parameter(torch.randn(self.ch_out_lf, self.ch_in_lf, kernel_size, kernel_size))

        # PADDING: (H - F + 2P)/S + 1 = 2 * [(0.5 H - F + 2P)/S +1] -> P = (F-S)/2
        self.padding = (kernel_size - stride) // 2

    def forward(self, x):
        # logic to handle input tensors:
        # if ch_in_lf = 0., we assume to be at the first layer, with only high freq repr
        if self.ch_in_lf == 0:
            hf_input = x
            lf_input = None
        elif self.ch_in_hf == 0:
            lf_input = x
            hf_input = None
        else:
            fmap_height, fmap_width = x.shape[-2], x.shape[-1]
            hf_input = x[:, :self.ch_in_hf * 4, ...].reshape(-1, self.ch_in_hf, fmap_height * 2, fmap_width * 2)
            lf_input = x[:, self.ch_in_hf * 4:, ...]

        # apply convolutions
        HtoH = HtoL = LtoL = LtoH = 0.
        if self.wHtoH is not None:
            HtoH = F.conv2d(hf_input, self.wHtoH, padding=self.padding)
        if self.wHtoL is not None:
            HtoL = F.conv2d(F.avg_pool2d(hf_input, 2), self.wHtoL, padding=self.padding)
        if self.wLtoH is not None:
            LtoH = F.interpolate(
                F.conv2d(lf_input, self.wLtoH, padding=self.padding),
                scale_factor=2, mode='nearest'
            )
        if self.wLtoL is not None:
            LtoL = F.conv2d(lf_input, self.wLtoL, padding=self.padding)

        # compute output tensors
        hf_output = HtoH + LtoH
        lf_output = LtoL + HtoL

        # logic to handle output tensors:
        # if ch_out_lf = 0., we assume to be at the last layer, with only high freq repr
        if self.ch_out_lf == 0:
            output = hf_output
        elif self.ch_out_hf == 0:
            output = lf_output
        else:
            # if alpha in (0, 1)
            fmap_height, fmap_width = hf_output.shape[-2] // 2, hf_output.shape[-1] // 2
            hf_output = hf_output.reshape(-1, 4 * self.ch_out_hf, fmap_height, fmap_width)
            output = torch.cat([hf_output, lf_output], dim=1)  # cat over channel dim
        return output


class OctConvPool2d(nn.Module):
    """Pooling module for 2d features represented by OctConv way.
    """
    def __init__(self, channels, kernel_size, stride=None, mode='max', alpha=0.5):
        super(OctConvPool2d, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        # prepare pooling layer to be applied to both lf and hf features
        if mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride)
        elif mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride)
        else:
            raise NotImplementedError()

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.ch_hf == 0 or self.ch_lf == 0:
            return self.pool(x)
        
        # decompose features into high/low repr
        fmap_height, fmap_width = x.shape[-2], x.shape[-1]
        hf_input = x[:, :self.ch_hf * 4, ...].reshape(-1, self.ch_hf, fmap_height * 2, fmap_width * 2)
        lf_input = x[:, self.ch_hf * 4:, ...]

        # apply pooling
        hf_pool = self.pool(hf_input)
        lf_pool = self.pool(lf_input)

        # compose high/low features into a tensor
        fmap_height, fmap_width = hf_pool.shape[-2] // 2, hf_pool.shape[-1] // 2
        hf_pool = hf_pool.reshape(-1, 4 * self.ch_hf, fmap_height, fmap_width)
        output = torch.cat([hf_pool, lf_pool], dim=1)  # cat over channel dim
        return output


class OctConvUpsample(nn.Module):
    def __init__(self, channels, scale_factor, mode='bilinear', align_corners=True, alpha=0.5):
        super(OctConvUpsample, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        # prepare upsample layer to be applied to both lf and hf features
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.ch_hf == 0 or self.ch_lf == 0:
            return self.upsample(x)
        
        # decompose features into high/low repr
        fmap_height, fmap_width = x.shape[-2], x.shape[-1]
        hf_input = x[:, :self.ch_hf * 4, ...].reshape(-1, self.ch_hf, fmap_height * 2, fmap_width * 2)
        lf_input = x[:, self.ch_hf * 4:, ...]

        # apply upsampling
        hf_up = self.upsample(hf_input)
        lf_up = self.upsample(lf_input)

        # compose high/low features into a tensor
        fmap_height, fmap_width = hf_up.shape[-2] // 2, hf_up.shape[-1] // 2
        hf_up = hf_up.reshape(-1, 4 * self.ch_hf, fmap_height, fmap_width)
        output = torch.cat([hf_up, lf_up], dim=1)  # cat over channel dim
        return output


class OctConvBatchNorm2d(nn.Module):
    def __init__(self, channels, alpha=0.5):
        super(OctConvBatchNorm2d, self).__init__()

        assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
        # input channels
        self.ch_lf = int(alpha * channels)
        self.ch_hf = channels - self.ch_lf

        # prepare batchnorm layers for lf and hf features
        self.bn_lf = nn.BatchNorm2d(self.ch_lf) if self.ch_lf > 0 else None
        self.bn_hf = nn.BatchNorm2d(self.ch_hf) if self.ch_hf > 0 else None

    def forward(self, x):
        # case in which either of low- or high-freq repr is given
        if self.bn_lf is None:
            return self.bn_hf(x)
        if self.bn_hf is None:
            return self.bn_lf(x)
        
        # decompose features into high/low repr
        fmap_height, fmap_width = x.shape[-2], x.shape[-1]
        hf_input = x[:, :self.ch_hf * 4, ...].reshape(-1, self.ch_hf, fmap_height * 2, fmap_width * 2)
        lf_input = x[:, self.ch_hf * 4:, ...]

        # apply batch normalization
        hf_bn = self.bn_hf(hf_input)
        lf_bn = self.bn_lf(lf_input)

        # compose high/low features into a tensor
        fmap_height, fmap_width = hf_bn.shape[-2] // 2, hf_bn.shape[-1] // 2
        hf_bn = hf_bn.reshape(-1, 4 * self.ch_hf, fmap_height, fmap_width)
        output = torch.cat([hf_bn, lf_bn], dim=1)  # cat over channel dim
        return output
