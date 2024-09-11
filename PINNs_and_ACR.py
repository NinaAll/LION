import torch
import torch.nn as nn
import config
from LION.models import LIONmodel
import LION.CTtools.ct_geometry as ct
from LION.utils.parameter import LIONParameter
from torch.utils.data import DataLoader
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
import torch.nn.utils.parametrize as P
from ts_algorithms import fdk
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from LION.utils.math import power_method

# import wandb
from LION.utils.math import power_method
import pathlib
import LION.CTtools.ct_geometry as ct
import LION.experiments.ct_experiments as ct_experiments

from typing import Tuple
import tomosipo as ts
from LION.CTtools.ct_geometry import Geometry
from ts_algorithms import fdk as ts_fdk

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


class convexnet(nn.Module):
    def __init__(self, n_channels=16, kernel_size=5, n_layers=5, convex=True, n_chan=1):
        super().__init__()
        # self.args=args
        # self.convex = args.wclip
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.smooth_length = 0
        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList(
            [
                nn.Conv2d(
                    n_chan,
                    n_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=2,
                    bias=True,
                )
                for _ in range(self.n_layers + 1)
            ]
        )

        # these layers should have non-negative weights
        self.wzs = nn.ModuleList(
            [
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=2,
                    bias=False,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_conv2d = nn.Conv2d(
            n_channels, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False
        )

        self.initialize_weights()

        # #FoE kernels
        # self.n_kernels = 10
        # ker_size=5
        # self.conv = nn.ModuleList([nn.Conv2d(n_chan, 32, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False)\
        #                            for i in range(self.n_kernels)])

    def initialize_weights(self, min_val=0, max_val=1e-3):
        for layer in range(self.n_layers):
            self.wzs[layer].weight.data = min_val + (
                max_val - min_val
            ) * torch.rand_like(self.wzs[layer].weight.data)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand_like(
            self.final_conv2d.weight.data
        )

    def clamp_weights(self):
        for i in range(self.smooth_length, self.n_layers):
            self.wzs[i].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)

    def wei_dec(self):
        rate = 10  # 500
        # for i in range(self.n_kernels):
        # self.conv[i].weight.data=(1-2*rate*self.args.lr)*self.conv[i].weight.data

    def forward(self, x, grady=False):
        # for layer in range(self.n_layers):
        #     print((self.wzs[layer].weight.data<0).sum())
        # if self.convex:
        self.clamp_weights()  # makes sure that it is convex

        z = self.leaky_relu(self.wxs[0](x))
        for layer_idx in range(self.n_layers):
            z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx + 1](x))
        z = self.final_conv2d(z)
        net_output = z.view(z.shape[0], -1).mean(dim=1, keepdim=True)
        # assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape}"
        # print(net_output.shape)
        # print(net_output.mean().item(),foe_out.mean().item(),l2_out.mean().item())
        return net_output
