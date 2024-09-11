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

import torch
from torch.utils.data import DataLoader
import pathlib
from LION.models.post_processing.FBPConvNet import FBPConvNet
import matplotlib.pyplot as plt
import LION.experiments.ct_experiments as ct_experiments

from torch.nn.functional import relu
import torch.nn.functional as F
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from typing import Tuple
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk

from torch.autograd import Variable


#%% U-Net


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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

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
        self,
        model_parameters: LIONParameter = None,
        geometry_parameters: ct.Geometry = None,
    ):
        super().__init__(model_parameters, geometry_parameters)
        # print(self.model_parameters)
        self.start = DoubleConv(1, 64)

        self.down_convolution_1 = DownSample(64, 128)
        self.down_convolution_2 = DownSample(128, 256)
        self.down_convolution_3 = DownSample(256, 512)
        self.down_convolution_4 = DownSample(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        # self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.finalconv = nn.Conv2d(64, 1, 1, padding="same")

    def forward(self, x):
        x = self.start(x)
        l1 = self.down_convolution_1(x)
        l2 = self.down_convolution_2(l1)
        l3 = self.down_convolution_3(l2)
        l4 = self.down_convolution_4(l3)
        # l5 = self.down_convolution_5(l4)
        # print("l1")
        # print(l1.shape)
        # print(l2.shape)
        # print(l3.shape)
        # print(l4.shape)
        # print(l3.shape)
        u1 = self.up_convolution_1(l4, l3)
        # print(u1.shape)

        u2 = self.up_convolution_2(u1, l2)
        # print(u2.shape)

        u3 = self.up_convolution_3(u2, l1)
        # print(u3.shape)

        u4 = self.up_convolution_4(u3, x)
        # print(x.shape)

        out = self.finalconv(x)
        return out

    @staticmethod
    def default_parameters():
        UNet_params = LIONParameter()
        UNet_params.in_channels = 1
        return UNet_params


#%% This example shows how to train UNet for full angle, noisy measurements.

# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("UNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("UNet_check_*.pt")
validation_fname = savefolder.joinpath("UNet_min_val.pt")
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
# experiment = ct_experiments.clinicalCTRecon()
experiment = ct_experiments.ExtremeLowDoseCTRecon()
# experiment = ct_experiments.SparseAngleCTRecon()

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

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


#%% AR


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


class network(nn.Module):
    def __init__(self, n_chan=1):
        super(network, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 16, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2),
            self.leaky_relu,
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2, stride=2),
            self.leaky_relu,
        )
        # size = 1024
        size = 512
        self.fc = nn.Sequential(
            nn.Linear(128 * (size // 2**4) ** 2, 256),
            self.leaky_relu,
            nn.Linear(256, 1)
            # nn.Linear(4, 1)
        )

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output


class AR(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):

        super().__init__(model_parameters, geometry_parameters)
        self._make_operator()

        self.network = network()
        # First Conv
        self.estimate_lambda()
        self.step_amounts = torch.tensor([150.0])
        self.op_norm = power_method(self.op)
        self.model_parameters.step_size = 0.2 / (self.op_norm) ** 2

    def forward(self, x):
        # x = fdk(self.op, x)
        x = self.normalise(x)
        # print(self.pool(z).mean(),self.L2(z).mean())
        # print(self.network(x).reshape(-1,1).shape)
        return self.network(x).reshape(
            -1, 1
        )  # + self.L2(z) <-- here is an issue because the reshape takes some dimensions which leads to an error
        # return x

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
                # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
            self.lamb = residual.mean() / len(dataset)
        print("Estimated lambda: " + str(self.lamb))

    # def output(self, x):
    # return self.AT(x)

    def var_energy(self, x, y):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        return 0.5 * ((self.A(x) - y) ** 2).sum() + self.lamb * self.forward(x).sum()

    ### What is the difference between .sum() and .mean()??? idfk but PSNR is lower when I do .sum

    def output(self, y, truth=None):
        # wandb.log({'Eearly_stopping_steps': self.step_amounts.mean().item(), 'Eearly_stopping_steps_std': self.step_amounts.std().item()})
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
            [x], lr=self.model_parameters.step_size, momentum=0.5
        )  # self.model_parameters.momentum)
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
                # print('decay')
            for g in optimizer.param_groups:
                g["lr"] = lr
            # x.grad+=data_misfit_grad
            if truth is not None:
                loss = WGAN_gradient_penalty_loss()(
                    x.detach(), truth.detach().to(device)
                )
                psnr_val = my_psnr(truth.detach().to(device), x.detach()).mean()
                ssim_val = my_ssim(truth.detach().to(device), x.detach()).mean()
                # wandb.log({'MSE Loss': loss.item(),'SSIM':ssim_val,'PSNR':psnr_val})
                # wandb.log({'MSE Loss'+str(self.model_parameters.step_size): loss.item(),'SSIM'+str(self.model_parameters.step_size):ssim_val,'PSNR'+str(self.model_parameters.step_size):psnr_val})
                print(
                    f"{j}: SSIM: {my_ssim(truth.to(device).detach(),x.detach())}, PSNR: {my_psnr(truth.to(device).detach(),x.detach())}, Energy: {energy.detach().item()}"
                )

                #     if(self.args.outp):
                #         print(j)
                prevpsn = curpsn
                curpsn = psnr_val
                # if(curpsn<prevpsn):
                #     self.step_amounts = torch.cat((self.step_amounts,torch.tensor([j*1.0])))
                #     return x.detach()
            elif j > self.step_amounts.mean().item():
                # print('only for testing')
                x.clamp(min=0.0)
                return x.detach()
            elif lr * self.op_norm**2 < 1e-3:
                x.clamp(min=0.0)
                return x.detach()
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
        if cite_format == "MLA":
            print("Mukherjee, Subhadip, et al.")
            print('"Data-Driven Convex Regularizers for Inverse Problems."')
            print(
                "ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024"
            )
            print("arXiv:2008.02839 (2020).")
        elif cite_format == "bib":
            string = """
            @inproceedings{mukherjee2024data,
            title={Data-Driven Convex Regularizers for Inverse Problems},
            author={Mukherjee, S and Dittmer, S and Shumaylov, Z and Lunz, S and {\"O}ktem, O and Sch{\"o}nlieb, C-B},
            booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
            pages={13386--13390},
            year={2024},
            organization={IEEE}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )


# Learning a regularization functional, i.e. train the AR:

#%% Loss function


class WGAN_gradient_penalty_loss(nn.Module):
    def __init__(self, mu=10.0 * 1e-2):
        self.mu = mu
        super().__init__()

    def forward(self, model, data_marginal_noisy, data_marginal_real):
        """Calculates the gradient penalty loss for WGAN GP"""
        real_samples = data_marginal_real
        print(real_samples.shape)
        fake_samples = data_marginal_noisy
        print(fake_samples.shape)
        # fake_samples=fake_samples[:,:,None, None]
        # print(fake_samples.shape)
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
            real_samples
        )
        print(alpha.shape)
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        net_interpolates = model(interpolates)
        fake = (
            torch.Tensor(real_samples.shape[0], 1)
            .fill_(1.0)
            .type_as(real_samples)
            .requires_grad_(False)
        )
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        # print(model(real_samples).mean()-model(fake_samples).mean(),self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean())
        loss = (
            model(real_samples).mean()
            - model(fake_samples).mean()
            + self.mu * (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        )
        return loss


#%% ACNCR


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


class ACNCR(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):
        super().__init__(model_parameters, geometry_parameters)

        full_chan = 16  # sparse
        self.convnet = convexnet(n_channels=16, n_chan=full_chan * 4)
        self.convnet_data = convexnet(n_channels=16, n_chan=1, n_layers=10)
        self.op = make_operator(experiment.geo)
        self.nw = power_method(self.op)
        self.smooth = smooth()

    def init_weights(self, m):
        pass

    def clamp_weights(self):
        self.convnet.clamp_weights()
        self.convnet_data.clamp_weights()

    def wei_dec(self):
        self.convnet.wei_dec()
        self.convnet_data.wei_dec()
        self.smooth.wei_dec()

    def forward(self, image):
        # output = self.convnet(self.smooth(image)) + self.convnet_data(data_img)
        w = self.op(image[0])
        # w = torch.unsqueeze(w, 0)
        sinogram = w / self.nw
        # print('hi')
        # print(sinogram.shape)
        # print(sinogram.max(),sinogram.min())
        # output = self.convnet(self.smooth(sinogram/(config.fwd_op_norm)))# + self.convnet_data(image)
        output = self.convnet(self.smooth(sinogram)) + self.convnet_data(image)
        return output

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


# Learning a regularization functional, i.e. train the AR:

#%% Loss function


class loss_function(nn.Module):
    def __init__(self, mu=10.0 * 1e-2):
        self.mu = mu
        super().__init__()

    def forward(self, model, scans, truth):
        """Calculates the gradient penalty loss for WGAN GP"""
        op = make_operator(experiment.geo)
        fake_samples = scans
        real_samples = truth

        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
            truth
        )
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        net_interpolates = model(interpolates)
        fake = Variable(
            torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(truth),
            requires_grad=False,
        )
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        # print((gradients.norm(2, dim=1)))

        decay_loss = 0
        loss = (
            model(real_samples).mean()
            - model(fake_samples).mean()
            + self.mu * (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        )
        # loss = self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean()#+self.mu*(((gradients_2.norm(2, dim=1) - 1)) ** 2).mean()
        return loss


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


#%% First run FBPConvNet_train.py to train and save model, then run this.
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname_fbp = savefolder.joinpath("FBPConvNet_final_iter.pt")
final_result_fname_unet = savefolder.joinpath("UNet_final_iter.pt")
final_result_fname_ar = savefolder.joinpath("AR_final_iter.pt")
final_result_fname_acncr = savefolder.joinpath("ACNCR_final_iter.pt")

# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
experiment = ct_experiments.clinicalCTRecon()
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model_fbp, _, _ = FBPConvNet.load(final_result_fname_fbp)
model_fbp.to(device)

model_unet, _, _ = UNet.load(final_result_fname_unet)
model_unet.to(device)

model_ar, _, _ = AR.load(final_result_fname_ar)
model_ar.to(device)

model_acncr, _, _ = ACNCR.load(final_result_fname_acncr)
model_acncr.to(device)
# sample a random batch (size 1, so really just one image, truth pair)
data, gt = next(iter(test_dataloader))
op = make_operator(experiment.geo)
x_fdk = fdk(data, op)

x_fbp = fdk(data, op)

x_unet = model_unet(x_fdk)


class function(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, x):
        z = model(x)
        return z


data_ar = torch.tensor(data, requires_grad=True)


def gradient_descent(lmb, n_its, op, sino, x, funct, model, lr, betas):
    # x = torch.tensor(x, requires_grad = True)
    def var_energy(x, sino):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        w = op(x[0])
        w = torch.unsqueeze(w, 0)
        return 0.5 * ((w - sino) ** 2).sum() + lmb * funct(model, x).sum()

    psnr_all = []

    for k in range(n_its):
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
        # print('outside')
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

x_ar, psnr_in_total = gradient_descent(
    lmb, 500, op, data, x_fbp, grad_function, model_ar, 1e-6, 0.95
)

x_acncr, psnr_in_total = gradient_descent(
    lmb, 500, op, data, x_fbp, grad_function, model_acncr, 1e-8, 0.95
)


x_eval_fbp = x_fbp
x_eval_fbp = np.squeeze(x_eval_fbp)
x_eval_fbp = x_eval_fbp.detach().cpu().numpy()

x_eval_ar = x_ar
x_eval_ar = np.squeeze(x_eval_ar)
x_eval_ar = x_eval_ar.detach().cpu().numpy()

x_eval_unet = x_unet
x_eval_unet = np.squeeze(x_eval_unet)
x_eval_unet = x_eval_unet.detach().cpu().numpy()

x_eval_acncr = x_acncr
x_eval_acncr = np.squeeze(x_eval_acncr)
x_eval_acncr = x_eval_acncr.detach().cpu().numpy()

gt_eval = gt
gt_eval = np.squeeze(gt_eval)
gt_eval = gt_eval.detach().cpu().numpy()

psnr_gt_di_fbp = peak_signal_noise_ratio(
    gt_eval, x_eval_fbp, data_range=gt_eval.max() - gt_eval.min()
)
ssim_gt_di_fbp = ssim(gt_eval, x_eval_fbp, data_range=gt_eval.max() - gt_eval.min())

psnr_gt_di_ar = peak_signal_noise_ratio(
    gt_eval, x_eval_ar, data_range=gt_eval.max() - gt_eval.min()
)
ssim_gt_di_ar = ssim(gt_eval, x_eval_ar, data_range=gt_eval.max() - gt_eval.min())

psnr_gt_di_unet = peak_signal_noise_ratio(
    gt_eval, x_eval_unet, data_range=gt_eval.max() - gt_eval.min()
)
ssim_gt_di_unet = ssim(gt_eval, x_eval_unet, data_range=gt_eval.max() - gt_eval.min())

psnr_gt_di_acncr = peak_signal_noise_ratio(
    gt_eval, x_eval_acncr, data_range=gt_eval.max() - gt_eval.min()
)
ssim_gt_di_acncr = ssim(gt_eval, x_eval_acncr, data_range=gt_eval.max() - gt_eval.min())

# put stuff back on the cpu, otherwise matplotlib throws an error
gt = gt.detach().cpu().numpy()
x_fbp = x_fbp.detach().cpu().numpy()
x_ar = x_ar.detach().cpu().numpy()
x_unet = x_unet.detach().cpu().numpy()
x_acncr = x_acncr.detach().cpu().numpy()

plt.figure()
plt.subplot(151)
plt.imshow(gt[0].T)
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("Ground truth", fontsize=6)
plt.subplot(152)
plt.imshow(x_fbp[0].T)
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("FBP denoised image", fontsize=6)
plt.text(0, 700, f"SSIM:{ssim_gt_di_fbp:.2f} \nPSNR:{psnr_gt_di_fbp:.2f}")
plt.subplot(153)
plt.imshow(x_unet[0].T)
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("UNet denoised image", fontsize=6)
plt.text(0, 700, f"SSIM:{ssim_gt_di_unet:.2f} \nPSNR:{psnr_gt_di_unet:.2f}")
plt.subplot(154)
plt.imshow(x_ar[0].T)
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("AR denoised image", fontsize=6)
plt.text(0, 700, f"SSIM:{ssim_gt_di_ar:.2f} \nPSNR:{psnr_gt_di_ar:.2f}")
plt.subplot(155)
plt.imshow(x_acncr[0].T)
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("ACNCR denoised image", fontsize=6)
plt.text(0, 700, f"SSIM:{ssim_gt_di_acncr:.2f} \nPSNR:{psnr_gt_di_acncr:.2f}")
plt.savefig("comparison.pdf", dpi=300)
plt.savefig("comparison.png", dpi=300)
