import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(4, 64, firstLayer=True)
        self.downStep2 = downStep(64, 128)
        self.downStep3 = downStep(128, 256)
        self.downStep4 = downStep(256, 512)
        self.downStep5 = downStep(512, 1024)
        
        self.upStep1 = upStep(1024, 512)
        self.upStep2 = upStep(512, 256)
        self.upStep3 = upStep(256, 128)
        self.upStep4 = upStep(128, 64)

        self.conv = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = F.interpolate(x, size=384)
        x1 = self.downStep1(x)
        x2 = self.downStep2(x1)
        x3 = self.downStep3(x2)
        x4 = self.downStep4(x3)
        x5 = self.downStep5(x4) 

        x = self.upStep1(x5, x4)
        x = self.upStep2(x, x3)
        x = self.upStep3(x, x2)
        x = self.upStep4(x, x1)

        x = self.conv(x)
        x = F.interpolate(x, size=360)
        x = torch.tanh(x)
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
        pad = 1 if keep_div else 0
        self.upsampling = nn.ConvTranspose2d(inC, outC, 2, stride=2, padding=pad)

    def forward(self, x, x_down):
        # todo
        x = self.upsampling(x)

        x = torch.cat([x_down, x], dim=1)

        x = self.conv(x)
        
        return x