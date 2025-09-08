""" Full assembly of the parts to form the complete network (original from from https://github.com/milesial/Pytorch-UNet)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class mlpDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_shape, out_channels):
        super().__init__()
        in_shape = (in_shape[0], in_shape[1] // 2, in_shape[2] // 2)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            mlpSpatialMixing(in_shape, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class mlpUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_shape, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = mlpSpatialMixing(in_shape, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class mlpSpatialMixing(nn.Module):

    def __init__(self, input_shape, out_channels):
        super().__init__()

        #input_shape = (c,h,w)
        s = input_shape[1] * input_shape[2]
        k = input_shape[0]

        #k = mid_channels
        #s = w*h
        self.c1 = nn.Conv2d(k, out_channels * 2, kernel_size=3, padding=1)  #(n,k,w,h)
        self.c2 = nn.Conv2d(out_channels, k, kernel_size=3, padding=1)  #(n,k,w,h)
        self.c3 = nn.Conv2d(k, out_channels, kernel_size=3, padding=1)  #(n,k,w,h)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.spatial_mix_weights = nn.Parameter(torch.zeros(s, s))
        self.spatial_mix_bias = nn.Parameter(torch.ones(out_channels, 1))

        self.out_channels = out_channels
        self.input_shape = input_shape

    def forward(self, x):
        # print('first',x.shape)

        y = F.relu(self.c1(x))

        y = self.bn1(y)

        x1, x2 = y[:, :y.shape[1] // 2], y[:, y.shape[1] // 2:]

        y = x1.view((x2.shape[0], x2.shape[1], -1)) * (torch.matmul(x2.view((x2.shape[0], x2.shape[1], -1)),
                                                                    self.spatial_mix_weights) + self.spatial_mix_bias)

        y = self.c2(y.view((y.shape[0], y.shape[1], self.input_shape[1], self.input_shape[2])))

        x = F.relu(self.c3(x + y))

        x = self.bn2(x)

        return x


class mlpSpatUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(mlpSpatUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        sz = 96
        ch = 16
        sc = 2

        self.inc = mlpSpatialMixing((1, sz, sz), ch)
        self.down1 = mlpDown((ch, sz, sz), ch * sc ** 1)
        self.down2 = mlpDown((ch * sc ** 1, sz // 2, sz // 2), ch * sc ** 2)
        self.down3 = mlpDown((ch * sc ** 2, sz // 4, sz // 4), ch * sc ** 3)
        self.down4 = mlpDown((ch * sc ** 3, sz // 8, sz // 8), ch * sc ** 3)

        self.up1 = mlpUp((ch * sc ** 4, sz // 8, sz // 8), ch * sc ** 2, bilinear)
        self.up2 = mlpUp((ch * sc ** 3, sz // 4, sz // 4), ch * sc ** 1, bilinear)
        self.up3 = mlpUp((ch * sc ** 2, sz // 2, sz // 2), ch, bilinear)
        self.up4 = mlpUp((ch * sc ** 1, sz, sz), ch, bilinear)
        self.outc = OutConv(ch, n_classes)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print('here', x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        pred = torch.sigmoid(logits)
        return pred


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.c1_1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.c1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # self.bn1 = nn.InstanceNorm2d(mid_channels)
        # self.rel1 = nn.ReLU(inplace=True)

        self.c2_1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.c2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # self.bn2 = nn.InstanceNorm2d(out_channels)
    # self.rel2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)  #, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        pred = torch.sigmoid(logits)
        return pred
