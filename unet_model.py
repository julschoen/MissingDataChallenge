""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=4, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 3))

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
        out = self.outc(x)
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(4, 64, firstLayer=True)
        self.downStep2 = downStep(64, 128)
        self.downStep3 = downStep(128, 256)
        self.downStep4 = downStep(256, 512, keep_div=True)
        self.downStep5 = downStep(512, 1024)
        
        self.upStep1 = upStep(1024, 512)
        self.upStep2 = upStep(512, 256, keep_div=True)
        self.upStep3 = upStep(256, 128)
        self.upStep4 = upStep(128, 64)

        self.conv = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # todo
        x1 = self.downStep1(x)
        print('x1', x1.shape)
        x2 = self.downStep2(x1)
        print('x2', x2.shape)
        x3 = self.downStep3(x2)
        print('x3', x3.shape)
        x4 = self.downStep4(x3)
        print('x4', x4.shape)
        x5 = self.downStep5(x4) 
        print('x5', x5.shape)

        x = self.upStep1(x5, x4)
        print(x.shape)
        x = self.upStep2(x, x3)
        print(x.shape)
        x = self.upStep3(x, x2)
        print(x.shape)
        x = self.upStep4(x, x1)
        print(x.shape)

        x = self.conv(x)
        print(x.shape)
        x = self.tanh(x)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, firstLayer=False, keep_div=False):
        super(downStep, self).__init__()
        # todo
        self.firstLayer = firstLayer
        kernel = 2 if keep_div else 3
        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel, padding=1),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=1),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # todo
        if not self.firstLayer:
            x = self.maxpool(x)

        x = self.conv(x)

        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, keep_div=False):
        super(upStep, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3, padding=1),
            nn.ReLU())
        kernel = 2 if keep_div else 2
        pad = 1 if keep_div else 0
        self.upsampling = nn.ConvTranspose2d(inC, outC, kernel, stride=2, padding=pad)

    def forward(self, x, x_down):
        # todo
        x = self.upsampling(x)

        x = torch.cat([x_down, x], dim=1)

        x = self.conv(x)
        
        return x