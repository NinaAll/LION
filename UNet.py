import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from LION.models import LIONmodel
import pathlib

# from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
import tomosipo as ts

# from ts_algorithms import fdk
from torch.nn.functional import relu
import torch.nn.functional as F

# from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x2, x1):
        # print("hi")
        # print(x2.shape)
        x2 = self.up(x2)
        diffY = x1.size()[2] - x2.size()[2]
        # print(diffY)

        diffX = x1.size()[3] - x2.size()[3]
        # print(diffX)
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x1, x2], dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)
        return x


class UNet(LIONmodel.LIONmodel):
    def __init__(
        self, in_channels, num_classes, model_parameters: LIONParameter = None
    ):
        super().__init__(model_parameters)
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.down_convolution_5 = DownSample(512, 1024)

        self.bottle_neck = DoubleConv(1024, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        l1 = self.down_convolution_1(x)
        l2 = self.down_convolution_2(l1)
        l3 = self.down_convolution_3(l2)
        l4 = self.down_convolution_4(l3)
        l5 = self.down_convolution_5(l4)
        # print("l1")
        # print(l1.shape)
        # print(l2.shape)
        # print(l3.shape)
        # print(l4.shape)
        bl = self.bottle_neck(l5)
        # print(bl.shape)

        u1 = self.up_convolution_1(l5, l4)
        # print(u1.shape)

        u2 = self.up_convolution_2(u1, l3)
        # print(u2.shape)

        u3 = self.up_convolution_3(u2, l2)
        print(u3.shape)

        x = self.up_convolution_4(u3, l1)
        print(x.shape)

        out = self.out(x)
        return out

    @staticmethod
    def default_parameters():
        UNet_params = LIONParameter()

        return UNet_params


model = UNet(in_channels=1, num_classes=1)
input_tensor = torch.randn(4, 1, 180, 450)
output = model(input_tensor)
print(output.shape)
