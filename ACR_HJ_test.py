import torch
import torch.nn as nn
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

# import wandb
from LION.utils.math import power_method
import pathlib
import LION.CTtools.ct_geometry as ct
import LION.experiments.ct_experiments as ct_experiments

from typing import Tuple
import tomosipo as ts
from LION.CTtools.ct_geometry import Geometry
from ts_algorithms import fdk as ts_fdk

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage import data
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import radon

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import tomosipo as ts
from tomosipo.torch_support import to_autograd

# forward operator is radon transform if the pseudo inverse is fbp
# according to the algorithm 2 taken from the paper "Adversarial Regularizer in Inverses Problems" the learned regularization functional
# will be applied with gradient descent to denoise a picture

# the test will be done for only one image

#%% Adversarial Regularizer

# from AR_train import AR
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


#%% Test Adversarial Regularizer

# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("ACR_HJ_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ACR_HJ_check_*.pt")
validation_fname = savefolder.joinpath("ACR_HJ_min_val.pt")


#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon()
# experiment = ct_experiments.clinicalCTRecon()
experiment = ct_experiments.ExtremeLowDoseCTRecon()
# experiment = ct_experiments.SparseAngleCTRecon()
# experiment = ct_experiments.SparseAngleExtremeLowDoseCTRecon()

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname = savefolder.joinpath("ACR_HJ_final_iter.pt")

# % Set device:
# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model, _, _ = ACR_HJ.load(final_result_fname)
model.to(device)

# sample a random batch (size 1, so really just one image, truth pair)
y, gt = next(iter(test_dataloader))

op = make_operator(experiment.geo)
torch_op = to_autograd(op, num_extra_dims=1)

# Use gradient descent to denoise the image with the learned regularization functional
# operator is radon transform

data = fdk(y, op)


class function(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, x, t):
        z = model(x, t)
        return z


train_param = LIONParameter()
train_param.optimiser = "adam"

train_param.epochs = 100
train_param.learning_rate = (
    1e-6  # always changeable to see if the code trains: e.g., 1e-4
)
train_param.betas = (0.9, 0.99)
train_param.beta_rate = 0.95
train_param.loss = "MSELoss"
data_ar = torch.tensor(data, requires_grad=True)

optimiser = torch.optim.Adam(
    [data_ar], lr=train_param.learning_rate, betas=train_param.betas
)

# w = op(data[0])
# w = torch.unsqueeze(w, 0)
# var_energy = 0.5*((w-y)**2).sum() + 20.000 * model(data)

# print(var_energy(data-data.grad*2, y))
from tqdm import tqdm


def gradient_descent(lmb, n_its, op, sino, x, t, funct, model, lr, betas):
    # x = torch.tensor(x, requires_grad = True)
    def var_energy(x, sino):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        w = op(x[0])
        w = torch.unsqueeze(w, 0)
        return 0.5 * ((w - sino) ** 2).sum() + lmb * funct(model, x, t).sum()

    psnr_all = []

    for k in tqdm(range(n_its)):
        x = torch.tensor(x, requires_grad=True)
        sino = torch.tensor(sino, requires_grad=True)
        # optimiser.zero_grad()
        energy = var_energy(x, sino)
        energy.backward()
        while (
            var_energy(x - x.grad * lr, sino)
            > energy - 0.5 * lr * (x.grad.norm(dim=(2, 3)) ** 2).mean()
        ):
            lr = betas * lr
            # print(lr)
        x = x - lr * x.grad

        x_eval = x
        x_eval = np.squeeze(x_eval)
        x_eval = x_eval.detach().cpu().numpy()

        gt_eval = gt
        gt_eval = np.squeeze(gt_eval)
        gt_eval = gt_eval.detach().cpu().numpy()
        psnr = peak_signal_noise_ratio(
            gt_eval, x_eval, data_range=gt_eval.max() - gt_eval.min()
        )
        psnr_all.append(psnr)
        # optimiser.step()
    return x, psnr_all


def proximal_gradient_descent(lmb, n_its, op, sino, x, t, funct, model, lr, betas):
    # x = torch.tensor(x, requires_grad = True)
    device = torch.cuda.current_device()

    def var_energy(x, sino):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        w = op(x)
        # w = torch.unsqueeze(w, 0)
        return 0.5 * ((w - sino) ** 2).sum()

    psnr_all = []

    for k in tqdm(range(n_its)):
        x = torch.tensor(x, requires_grad=True)
        sino = torch.tensor(sino, requires_grad=True)
        # optimiser.zero_grad()
        energy = var_energy(x, sino)
        energy.backward()
        # print(x.grad)
        while (
            var_energy(x - x.grad * lr, sino)
            > energy - 0.5 * lr * (x.grad.norm(dim=(2, 3)) ** 2).mean()
        ):
            lr = betas * lr
            # print(lr)

        x = x - lr * x.grad

        x = x.detach()

        x.requires_grad = True

        t = torch.tensor([lr]).to(device)

        y = lmb * funct(model, x, t)  # replace t with lr
        y.backward()

        x = x - lr * x.grad
        print(x)

        x_eval = x
        x_eval = np.squeeze(x_eval)
        x_eval = x_eval.detach().cpu().numpy()

        gt_eval = gt
        gt_eval = np.squeeze(gt_eval)
        gt_eval = gt_eval.detach().cpu().numpy()
        psnr = peak_signal_noise_ratio(
            gt_eval, x_eval, data_range=gt_eval.max() - gt_eval.min()
        )
        psnr_all.append(psnr)
        # optimiser.step()
    return x, psnr_all


# print(z.backward(retain_graph=True))
# op() 3 dimensions
# x 4 dimensions 1 1 512 512
# x[0] 3 dimensions 1 512 512

grad_function = function()


def estimate_lambda(dataset):
    lamb = 1.0
    if dataset is None:
        lamb = 1.0
    else:
        residual = 0.0
        for index, (data, target) in enumerate(dataset):
            w = op.T(op(target[0]) - data[0])
            w = torch.unsqueeze(w, 0)
            residual += torch.norm(w, dim=(2, 3)).mean()
            # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
        lamb = residual.mean() / len(dataset)
    print("Estimated lambda: " + str(lamb))
    return lamb


lmb = estimate_lambda(test_dataloader)
print(lmb)

t = torch.Tensor(np.random.random((gt.size(0), 1))).type_as(gt)


x, psnr_in_total = proximal_gradient_descent(
    lmb, 100, torch_op, y, data, t * 1e-10, grad_function, model, 1e-4, 0.95
)
print(psnr_in_total)

# x, psnr_in_total = gradient_descent(
#     lmb, 1000, op, y, data, t, grad_function, model, 1e-4, 0.95
# )
# print(psnr_in_total)

# x = data_ar
print(x)

# torch.optim.SGD gradient descent
# epsilon = learning rate
# use var_energy for .backward
# x has to be a parameter so that the gradient knows with respect to which variable to take the derivative
# important optimizer.step() <- at the end of the code!!!
# gradient descent default
# get Zak's code working first, then work on my code
# issue: op is not the proper operator because y should have a different shape!

#%% Evaluation metrics

x_eval = x
x_eval = np.squeeze(x_eval)
x_eval = x_eval.detach().cpu().numpy()

gt_eval = gt
gt_eval = np.squeeze(gt_eval)
gt_eval = gt_eval.detach().cpu().numpy()
# print(gt_eval.dtype)

data_eval = data.detach().cpu().numpy()
data_eval = np.squeeze(data_eval)
# data_eval = data_eval.reshape(-1)
# print(data_eval.dtype)

x_one = x
gt_one = gt
data_one = data

# SSIM
ssim_gt_di = ssim(gt_eval, x_eval, data_range=gt_eval.max() - gt_eval.min())
print("SSIM - ground truth - denoised image:", ssim_gt_di)
ssim_gt_ni = ssim(gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min())
print("SSIM - ground truth - noisy image:", ssim_gt_ni)
# PSNR
psnr_gt_di = peak_signal_noise_ratio(
    gt_eval, x_eval, data_range=gt_eval.max() - gt_eval.min()
)
print("PSNR:", psnr_gt_di)
psnr_gt_ni = peak_signal_noise_ratio(
    gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min()
)
print("PSNR:", psnr_gt_ni)

# PRSN/SSIM for the whole test data set
# batch_size = 4
# lidc_dataloader = DataLoader(test_data, batch_size, shuffle=True)
ssim_gt_di_all = []
ssim_gt_ni_all = []
psnr_gt_di_all = []
psnr_gt_ni_all = []

for data, gt in iter(test_dataloader):
    image = fdk(data, op)
    reconstruction, psnr_in_total = proximal_gradient_descent(
        lmb, 100, torch_op, data, image, t * 1e-10, grad_function, model, 1e-4, 0.95
    )
    gt_eval = gt
    gt_eval = np.squeeze(gt_eval)
    gt_eval = gt_eval.detach().cpu().numpy()
    reconstruction_eval = reconstruction
    reconstruction_eval = np.squeeze(reconstruction_eval)
    reconstruction_eval = reconstruction_eval.detach().cpu().numpy()
    data_eval = image.detach().cpu().numpy()
    data_eval = np.squeeze(data_eval)

    ssim_gt_di = ssim(
        gt_eval, reconstruction_eval, data_range=gt_eval.max() - gt_eval.min()
    )
    # print('SSIM - ground truth - denoised image:', ssim_gt_di)
    ssim_gt_ni = ssim(gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min())
    # print('SSIM - ground truth - noisy image:', ssim_gt_ni)
    psnr_gt_di = peak_signal_noise_ratio(
        gt_eval, reconstruction_eval, data_range=gt_eval.max() - gt_eval.min()
    )
    psnr_gt_ni = peak_signal_noise_ratio(
        gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min()
    )
    # print('PSNR:', psnr)
    ssim_gt_ni_all.append(ssim_gt_ni)
    ssim_gt_di_all.append(ssim_gt_di)
    psnr_gt_di_all.append(psnr_gt_di)
    psnr_gt_ni_all.append(psnr_gt_ni)

ssim_di_all = np.mean(ssim_gt_di_all)
ssim_ni_all = np.mean(ssim_gt_ni_all)
psnr_di_all = np.mean(psnr_gt_di_all)
psnr_ni_all = np.mean(psnr_gt_ni_all)


print("SSIM - ground truth - denoised image mean", np.mean(ssim_gt_di_all))
print("SSIM - ground truth - noisy image mean", np.mean(ssim_gt_ni_all))
print("SSIM - ground truth - denoised image standard deviation", np.std(ssim_gt_di_all))
print("SSIM - ground truth - noisy image standard deviation", np.std(ssim_gt_ni_all))

print("PSNR - ground truth - denoised image mean", np.mean(psnr_gt_di_all))
print("PSNR - ground truth - noisy image mean", np.mean(psnr_gt_ni_all))
print("PSNR - ground truth - denoised image standard deviation", np.std(psnr_gt_di_all))
print("PSNR - ground truth - noisy image standard deviation", np.std(psnr_gt_ni_all))


# put stuff back on the cpu, otherwise matplotlib throws an error
x = x_one.detach().cpu().numpy()
gt = gt_one.detach().cpu().numpy()
data = data_one.detach().cpu().numpy()


# add the ACR to compare the values of the HJ approach and the ACR approach

plt.figure()
plt.subplot(131)
plt.imshow(x[0].T)
# plt.colorbar()
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("ACR-HJ denoised image")
# plt.text(0, 650, f"SSIM:{ssim_gt_di:.2f} \nPSNR:{psnr_gt_di:.2f} \nSSIM mean:{ssim_di_all:.2f} \nPSNR mean:{psnr_di_all:.2f}")
plt.text(
    0,
    750,
    f"SSIM:{ssim_gt_di:.2f} \nPSNR:{psnr_gt_di:.2f} \nSSIM-mean: {np.mean(ssim_gt_di_all):.2f} \nPSNR-mean: {np.mean(psnr_gt_di_all):.2f}",
)
plt.subplot(132)
plt.imshow(gt[0].T)
# plt.colorbar()
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("Ground truth")
plt.subplot(133)
plt.imshow(data[0].T)
# plt.colorbar()
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("Noisy image")
# plt.text(0, 650, f"SSIM:{ssim_gt_ni:.2f} \nPSNR:{psnr_gt_ni:.2f} \nSSIM mean:{ssim_ni_all:.2f} \nPSNR mean:{psnr_ni_all:.2f}")
plt.text(
    0,
    750,
    f"SSIM:{ssim_gt_ni:.2f} \nPSNR:{psnr_gt_ni:.2f} \nSSIM-mean: {np.mean(ssim_gt_ni_all):.2f} \nPSNR-mean: {np.mean(psnr_gt_ni_all):.2f}",
)
plt.savefig("acr_hj_img_mean_values_prox_grad_ex_low_dose.pdf", dpi=300)


# plt.figure()
# plt.subplot(121)
# plt.imshow(x[0].T)
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(data[0].T)
# plt.colorbar()
# plt.savefig("AR_unet_new.png")


# # A is operator op and it is the radon transform
# grad_function = function()
# data = torch.tensor(data, requires_grad = True)
# model = AR(geometry_parameters=experiment.geo).to(device)

# data = torch.nn.Parameter(data)
# data_ar = torch.tensor(data, requires_grad = True)
# z = grad_function(model, data)

# train_param = LIONParameter()
# train_param.optimiser = "adam"

# train_param.epochs = 100
# train_param.learning_rate = 1e-4 # always changeable to see if the code trains: e.g., 1e-4
# train_param.betas = (0.9, 0.99)
# train_param.loss = "MSELoss"
# data_ar = torch.tensor(data, requires_grad = True)

# # optimiser = torch.optim.Adam(
# #    [data_ar], lr=train_param.learning_rate, betas=train_param.betas
# # )

# optimiser = torch.optim.SGD([data_ar], lr=train_param.learning_rate, momentum=0.5)
# # z = grad_function(model, data_ar)
# # z.backward()
# # data_ar = data_ar.grad * 0.001 * (0.001)
# # print(data_ar)

# #data_ar = torch.tensor(data, requires_grad = True)

# # z = grad_function(model, data_ar)
# # z.backward()
# # data_ar = data_ar.grad * 0.001 * (0.001)
# # print(data_ar)
# print('hi')
# print(data_ar)

# for k in range(200):
#     data_ar = torch.tensor(data_ar, requires_grad = True)
#     optimiser.zero_grad()
#     z = grad_function(model, data_ar)
#     z.backward()
#     # print(data_ar.grad)
#     m = data_ar.grad * 0.5 * (0.000001)

#     w = op.T(op(data_ar[0])-y[0])
#     w = torch.unsqueeze(w, 0)
#     #print(w)
#     w = w * 0.000001

#     data_ar = data_ar - w - m
#     optimiser.step() # optimiser is the issue
