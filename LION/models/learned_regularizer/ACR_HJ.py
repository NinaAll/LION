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

device = torch.device("cuda:1")
torch.cuda.set_device(device)


def my_ssim(x: torch.tensor, y: torch.tensor):
    if x.shape[0] == 1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return ssim(x, y, data_range=x.max() - x.min())
    else:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(ssim(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


def my_psnr(x: torch.tensor, y: torch.tensor):
    if x.shape[0] == 1:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        return psnr(x, y, data_range=x.max() - x.min())
    else:
        x = x.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(psnr(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


class Positive(nn.Module):
    def forward(self, X):
        return torch.clip(X, min=0.0)


class ICNN_layer(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1, relu_type="LeakyReLU"):
        super().__init__()

        # The paper diagram is in color, channels are described by "blue" and "orange"
        self.blue = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.blue, "weight", Positive())

        self.blue2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.blue2, "weight", Positive())

        self.orange = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=True,
        )

        self.orange2 = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=True,
        )

        self.orange_quadratic = nn.Conv2d(
            1, channels, kernel_size, stride=1, padding="same", bias=False
        )
        self.orange_quadratic2 = nn.Conv2d(
            1, channels, kernel_size, stride=1, padding="same", bias=False
        )

        if relu_type == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

    def forward(self, z, x0):

        res = (
            self.blue(z)
            + self.orange(x0)
            + self.orange_quadratic(x0) ** 2
            + t * (self.blue2(z) + self.orange2(x0) + self.orange_quadratic2(x0) ** 2)
        )
        res = self.activation(res)
        return res


class ACR_HJ(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):

        super().__init__(model_parameters, geometry_parameters)
        self._make_operator()
        # First Conv
        self.first_layer = nn.Conv2d(
            in_channels=1,
            out_channels=model_parameters.channels,
            kernel_size=model_parameters.kernel_size,
            stride=model_parameters.stride,
            padding="same",
            bias=True,
        )

        if model_parameters.relu_type == "LeakyReLU":
            self.first_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

        for i in range(model_parameters.layers):
            self.add_module(
                f"ICNN_layer_{i}",
                ICNN_layer(
                    channels=model_parameters.channels,
                    kernel_size=model_parameters.kernel_size,
                    stride=model_parameters.stride,
                    relu_type=model_parameters.relu_type,
                ),
            )

        self.last_layer = nn.Conv2d(
            in_channels=model_parameters.channels,
            out_channels=1,
            kernel_size=model_parameters.kernel_size,
            stride=model_parameters.stride,
            padding="same",
            bias=False,
        )
        P.register_parametrization(self.last_layer, "weight", Positive())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.initialize_weights()
        self.estimate_lambda()
        self.op_norm = power_method(self.op)
        self.model_parameters.step_size = 1 / (self.op_norm) ** 2

    # a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=1e-3):
        device = torch.cuda.current_device()
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            block.blue.weight.data = min_val + (max_val - min_val) * torch.rand_like(
                block.blue.weight.data
            )
        self.last_layer.weight.data = min_val + (max_val - min_val) * torch.rand_like(
            self.last_layer.weight.data
        )
        return self

    def improved_initialize_weights(self, min_val=0.0, max_val=0.001):
        ###
        ### This is based on a recent paper https://openreview.net/pdf?id=pWZ97hUQtQ
        ###
        # convex_init = ConvexInitialiser()
        # w1, b1 = icnn[1].parameters()
        # convex_init(w1, b1)
        # assert torch.all(w1 >= 0) and b1.var() > 0
        device = torch.cuda.current_device()
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            block.blue.weight.data = min_val + (max_val - min_val) * torch.rand(
                self.model_parameters.channels,
                self.model_parameters.channels,
                self.model_parameters.kernel_size,
                self.model_parameters.kernel_size,
            ).to(device)
        self.last_layer.weight.data = min_val + (max_val - min_val) * torch.rand_like(
            self.last_layer.weight.data
        )
        return self

    def forward(self, x):
        # x = fdk(self.op, x)
        x = self.normalise(x)
        z = self.first_layer(x)
        z = self.first_activation(z)
        for i in range(self.model_parameters.layers):
            layer = primal_module = getattr(self, f"ICNN_layer_{i}")
            z = layer(z, x)

        z = self.last_layer(z)
        # print(self.pool(z).mean(),self.L2(z).mean())
        return self.pool(z).reshape(-1, 1)

    def estimate_lambda(self, dataset=None):
        self.lamb = 1.0
        if dataset is None:
            self.lamb = 1.0
        else:
            residual = 0.0
            for index, (data, target) in enumerate(dataset):
                residual += torch.norm(
                    self.AT(self.A(target) - data), dim=(2, 3)
                ).mean()
            self.lamb = residual.mean() / len(dataset)
        print("Estimated lambda: " + str(self.lamb))

    # def output(self, x):
    # return self.AT(x)

    def var_energy(self, x, y):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        return 0.5 * ((self.A(x) - y) ** 2).sum() + self.lamb * self.forward(x).sum()

    ### What is the difference between .sum() and .mean()??? idfk but PSNR is lower when I do .sum

    def output(self, y, truth=None):
        x0 = []
        device = torch.cuda.current_device()
        for i in range(y.shape[0]):
            x0.append(fdk(self.op, y[i]))
        x = torch.stack(x0)
        # print(x.shape)
        # print(x.min(),x.max())
        # print(my_psnr(truth.detach().to(device),x.detach()).mean(),my_ssim(truth.detach().to(device),x.detach()).mean())
        x = torch.nn.Parameter(x)  # .requires_grad_(True)

        optimizer = torch.optim.SGD(
            [x],
            lr=self.model_parameters.step_size,
            momentum=self.model_parameters.momentum,
        )
        lr = self.model_parameters.step_size
        prevpsn = 0
        curpsn = 0
        for j in range(self.model_parameters.no_steps):
            # print(x.min(),x.max())
            # data_misfit=self.A(x)-y
            # data_misfit_grad = self.AT(data_misfit)

            optimizer.zero_grad()
            # reg_func=self.lamb * self.forward(x).mean()
            # reg_func.backward()
            # print(x.requires_grad, reg_func.requires_grad)
            energy = self.var_energy(x, y)
            energy.backward()
            while (
                self.var_energy(x - x.grad * lr, y)
                > energy - 0.5 * lr * (x.grad.norm(dim=(2, 3)) ** 2).mean()
            ):
                lr = self.model_parameters.beta_rate * lr
            for g in optimizer.param_groups:
                g["lr"] = lr
            # x.grad+=data_misfit_grad
            if truth is not None:
                loss = torch.nn.MSELoss()(x.detach(), truth.detach().to(device))
                psnr_val = my_psnr(truth.detach().to(device), x.detach()).mean()
                ssim_val = my_ssim(truth.detach().to(device), x.detach()).mean()
                # wandb.log({'MSE Loss': loss.item(),'SSIM':ssim_val,'PSNR':psnr_val})
                # wandb.log({'MSE Loss'+str(self.model_parameters.step_size): loss.item(),'SSIM'+str(self.model_parameters.step_size):ssim_val,'PSNR'+str(self.model_parameters.step_size):psnr_val})
                print(
                    f"{j}: SSIM: {my_ssim(truth.to(device).detach(),x.detach())}, PSNR: {my_psnr(truth.to(device).detach(),x.detach())}, Energy: {energy.detach().item()}"
                )

            #     if(self.args.outp):
            #         print(j)
            #     prevpsn=curpsn
            #     curpsn=psnr
            #     if(self.args.earlystop is True and curpsn<prevpsn):
            #         writer.close()
            #         return guess
            optimizer.step()
            x.clamp(min=0.0)
        return x.detach()

    def normalise(self, x):
        return (x - self.model_parameters.xmin) / (
            self.model_parameters.xmax - self.model_parameters.xmin
        )

    def unnormalise(self, x):
        return (
            x * (self.model_parameters.xmax - self.model_parameters.xmin)
            + self.model_parameters.xmin
        )

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.channels = 16
        param.kernel_size = 5
        param.stride = 1
        param.relu_type = "LeakyReLU"
        param.layers = 5
        param.early_stopping = False
        param.no_steps = 150
        param.step_size = 1e-6
        param.momentum = 0.5
        param.beta_rate = 0.95
        param.xmin = 0.0
        param.xmax = 1.0
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        print("None")
