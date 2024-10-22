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

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    dir="/store/DAMTP/na673/",
    project="ACR_HJ_train",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-4,
        # "architecture": "CNN",
        "dataset": "ICCR",
        "epochs": 25,
    },
)


# Just a temporary SSIM that takes torch tensors (will be added to LION at some point)
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


class smooth(nn.Module):
    def __init__(self, n_chan=1, full_chan=16):
        super(smooth, self).__init__()
        self.act = nn.SiLU
        ker_siz = 7
        self.convnet = nn.Sequential(
            nn.Conv2d(
                n_chan, full_chan, kernel_size=(ker_siz, ker_siz), padding=ker_siz // 2
            ),
            # nn.InstanceNorm2d(16),
            self.act(),
            nn.Conv2d(
                full_chan,
                full_chan * 2,
                kernel_size=(ker_siz, ker_siz),
                padding=ker_siz // 2,
            ),
            # nn.InstanceNorm2d(32),
            # nn.MaxPool2d(5),
            self.act(),
            nn.Conv2d(
                full_chan * 2,
                full_chan * 2,
                kernel_size=(ker_siz, ker_siz),
                padding=ker_siz // 2,
            ),
            # nn.InstanceNorm2d(32),
            self.act(),
            nn.Conv2d(
                full_chan * 2,
                full_chan * 4,
                kernel_size=(ker_siz, ker_siz),
                padding=ker_siz // 2,
            ),
            # nn.InstanceNorm2d(64),
            self.act(),
            nn.Conv2d(
                full_chan * 4,
                full_chan * 4,
                kernel_size=(ker_siz, ker_siz),
                padding=ker_siz // 2,
            ),
            # nn.InstanceNorm2d(64),
            # nn.MaxPool2d(5),
            self.act(),
            # nn.Conv2d(64, 128, kernel_size=(ker_siz, ker_siz),padding=ker_siz//2),
            # self.act()
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(128*(config.size//16)**2, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 1)
        # )

    def init_weights(self, m):
        pass

    def wei_dec(self):
        rate = 10  # 500#10
        for i in range(self.n_kernels):
            self.convnet[i].weight.data = (1 - 2 * rate * self.args.lr) * self.convnet[
                i
            ].weight.data

    def forward(self, image):
        output = self.convnet(image)
        # output = output.view(image.size(0), -1)
        # output = self.fc(output)
        return output


# ACNCR network


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

        self.time_dense = nn.Linear(1, channels)

        if relu_type == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(
                "Only Leaky ReLU supported (needs to be a convex and monotonically nondecreasin fun)"
            )

    def forward(self, z, t, x0):

        t_emb = self.time_dense(t.view(-1, 1))
        # print('hello')
        # print(t.shape)
        # print(z.shape)
        # print(x0.shape)
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
            block.blue2.weight.data = min_val + (max_val - min_val) * torch.rand_like(
                block.blue2.weight.data
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

    def forward(self, x, t):
        # x = fdk(self.op, x)
        t = t.reshape(-1, 1, 1, 1)
        x = self.normalise(x)
        z = self.first_layer(x)
        z = self.first_activation(z)
        for i in range(self.model_parameters.layers):
            layer = primal_module = getattr(self, f"ICNN_layer_{i}")
            z = layer(z, t, x)

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

    def freeze_weights(self):
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            block.blue.requires_grad = False
            block.orange.requires_grad = False
            block.orange_quadratic.requires_grad = False

    def unfreeze_weights(self):
        for i in range(self.model_parameters.layers):
            block = getattr(self, f"ICNN_layer_{i}")
            block.blue.requires_grad = True
            block.orange.requires_grad = True
            block.orange_quadratic.requires_grad = True

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


#%% FBP


def fdk_from_geo(sino: torch.Tensor, geo: Geometry):
    B, _, _, _ = sino.shape
    op = make_operator(geo)
    return fdk(sino, op, *geo.image_size[1:])


def fdk(sino: torch.Tensor, op: ts.Operator.Operator) -> torch.Tensor:
    B, _, _, _ = sino.shape
    recon = None
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino[i])
        sub_recon = torch.clip(sub_recon, min=0)
        if recon is None:
            recon = sino.new_zeros(B, 1, *sub_recon.shape[1:])
        recon[i] = sub_recon
    assert recon is not None
    return recon


#%% Loss function


def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(
        obj, x, dummy, create_graph=True, retain_graph=True
    )[0]
    return derivative


class WGAN_HJ_loss(nn.Module):
    def __init__(self, mu=10.0, mu_1=1e-2):
        self.mu = mu
        self.mu_1 = mu_1
        super().__init__()

    def forward(self, model, data_noisy, data_real, t):
        real_samples = data_real
        fake_samples = data_noisy
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
            real_samples
        )

        interpolates_x = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        # print(interpolates_x.shape)
        # t = torch.from_numpy(np.random.uniform(0.0, T))
        # t = torch.distributions.Uniform(0, T).sample().requires_grad_(True)

        # t = torch.Tensor(np.random.random((real_samples.size(0), 1, 512, 512))).type_as(
        # real_samples
        # )

        # t = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
        #     real_samples
        # )

        # t = t.expand(2,1,512,512)
        # print(t.shape)
        # net_interpolates = model(interpolates_x, t)

        # print(net_interpolates.shape)

        # fake = (
        #     torch.Tensor(real_samples.shape[0], 1)
        #     .fill_(1.0)
        #     .type_as(real_samples)
        #     .requires_grad_(False)
        # )

        # x = torch.cat((t, interpolates_x), dim=1)

        # x = [t, interpolates_x]
        t = t.requires_grad_(True)

        fct = model(interpolates_x, t)

        u_x = fwd_gradients(fct, interpolates_x)
        u_x = u_x.reshape(u_x.shape[0], -1)

        u_t = fwd_gradients(fct, t)
        # print(u_tx.shape)
        # u_t = u_tx[:, -1:]
        # u_x = u_tx[:, 0:-1]

        # gradients_x = torch.autograd.grad(
        #     outputs=net_interpolates,
        #     inputs=interpolates_x,
        #     grad_outputs=fake,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # print(gradients_x.shape)

        # gradients_t = torch.autograd.grad(
        #     outputs=net_interpolates,
        #     inputs=t,
        #     grad_outputs=fake,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # print(gradients_t.shape)

        # gradients_x = gradients_x.view(gradients_x.size(0), -1)
        # gradients_t = gradients_t.view(gradients_t.size(0), -1)

        wgan_loss = (
            model(real_samples, t).mean()
            - model(fake_samples, t).mean()
            + self.mu * (((u_x.norm(2, dim=1) - 1)) ** 2).mean()
        )
        pinn_loss = ((u_t + 1 / 2 * u_x.norm(2, dim=1) ** 2) ** 2).mean()
        print("WGAN-loss:", wgan_loss)
        print("PINN-loss:", pinn_loss)
        return pinn_loss, wgan_loss, self.mu_1 * pinn_loss + 0 * wgan_loss


#%% FBP


def fdk_from_geo(sino: torch.Tensor, geo: Geometry):
    B, _, _, _ = sino.shape
    op = make_operator(geo)
    return fdk(sino, op, *geo.image_size[1:])


def fdk(sino: torch.Tensor, op: ts.Operator.Operator) -> torch.Tensor:
    B, _, _, _ = sino.shape
    recon = None
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(op, sino[i])
        sub_recon = torch.clip(sub_recon, min=0)
        if recon is None:
            recon = sino.new_zeros(B, 1, *sub_recon.shape[1:])
        recon[i] = sub_recon
    assert recon is not None
    return recon


# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("ACR_HJ_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ACR_HJ_check_*.pt")
validation_fname = savefolder.joinpath("ACR_HJ_min_val.pt")

#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
# experiment = ct_experiments.clinicalCTRecon()
experiment = ct_experiments.ExtremeLowDoseCTRecon()

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)


default_parameters = ACR_HJ.default_parameters()
model = ACR_HJ(experiment.geo, default_parameters).to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = WGAN_HJ_loss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 25
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "loss_function"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0


# %% Check if there is a checkpoint saved, and if so, start from there.

# If there is a file with the final results, don't run again
if model.final_file_exists(savefolder.joinpath(final_result_fname)):
    print("final model exists! You already reached final iter")
    exit()

# model, optimiser, start_epoch, total_loss, _ = ACR_HJ.load_checkpoint_if_exists(
#     checkpoint_fname, model, optimiser, total_loss
# )
# print(f"Starting iteration at epoch {start_epoch}")

op = make_operator(experiment.geo)

#%% train
loss_valid = []
loss_train = []

for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

    for sinogram, target_reconstruction in tqdm(lidc_dataloader):

        image = fdk(sinogram, op)
        # print(image.shape)
        optimiser.zero_grad()
        # reconstruction = model(image) <- different type of denoising here!

        reconstruction = image
        # print(reconstruction.shape)

        t = torch.Tensor(np.random.random((target_reconstruction.size(0), 1))).type_as(
            target_reconstruction
        )

        loss_pinn, loss_ar, loss = loss_fcn(
            model, reconstruction, target_reconstruction, t
        )
        # print(loss.shape)
        wandb.log(
            {
                "PINN-loss": loss_pinn,
                "ACR-loss": loss_ar,
            }
        )

        loss_ar.backward()

        # loss = fwd_gradients(loss, reconstruction)

        optimiser.step()

        model.freeze_weights()

        loss_pinn, loss_ar, loss = loss_fcn(
            model, reconstruction, target_reconstruction, t
        )

        loss_pinn.backward()

        optimiser.step()

        model.unfreeze_weights()

        train_loss += loss.item()

        # scheduler.step()
    total_loss[epoch] = train_loss
    loss_train.append(train_loss / len(lidc_dataset))
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):

        image = fdk(sinogram, op)
        t = torch.Tensor(np.random.random((target_reconstruction.size(0), 1))).type_as(
            target_reconstruction
        )
        reconstruction = image
        loss_pinn, loss_ar, loss = loss_fcn(
            model, reconstruction, target_reconstruction, t
        )
        valid_loss += loss.item()
        wandb.log(
            {
                "PINN-loss": loss_pinn,
                "ACR-loss": loss_ar,
            }
        )
    loss_valid.append(valid_loss / len(lidc_dataset_val))
    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataset)} \t\t Validation Loss: {valid_loss / len(lidc_dataset_val)}"
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving State Dict
        model.save(
            validation_fname,
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
            dataset=experiment.param,
        )

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
            dataset=experiment.param,
        )

x = list(range(start_epoch, train_param.epochs + start_epoch))

plt.plot(x, loss_train, label="train")
plt.plot(x, loss_valid, label="valid")
plt.legend()
plt.savefig("learning_curve_acncr.pdf")

model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)

# in order to not stop the code when the laptop is closed:
# nohup python ACR_HJ_train.py &
# tail -f nohup.out
