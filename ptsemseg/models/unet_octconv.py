# reference:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_octconv_parts import *

class unet_octconv(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, alpha=0.25):
        super(unet_octconv, self).__init__()
        self.inc = inconv(in_channels, 64, alpha)
        self.down1 = down(64, 128, alpha)
        self.down2 = down(128, 256, alpha)
        self.down3 = down(256, 512, alpha)
        self.down4 = down(512, 512, alpha)
        self.up1 = up(1024, 256, alpha)
        self.up2 = up(512, 128, alpha)
        self.up3 = up(256, 64, alpha)
        self.up4 = up(128, 64, alpha)
        self.outc = outconv(64, n_classes, alpha)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


# test purpose only
unet = unet_octconv()
data = torch.randn(2, 3, 128, 64)
out = unet(data)
print(out.shape)
