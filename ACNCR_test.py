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
from torch.autograd import Variable

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


#%% Test Adversarial Regularizer

# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("ACNCR_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ACNCR_check_*.pt")
validation_fname = savefolder.joinpath("ACNCR_min_val.pt")


#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon()
experiment = ct_experiments.clinicalCTRecon()
# experiment = ct_experiments.ExtremeLowDoseCTRecon()
# experiment = ct_experiments.SparseAngleCTRecon()
# experiment = ct_experiments.SparseAngleExtremeLowDoseCTRecon()

#%% Dataset
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname = savefolder.joinpath("ACNCR_final_iter.pt")

# % Set device:
# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model, _, _ = ACNCR.load(final_result_fname)
model.to(device)

# sample a random batch (size 1, so really just one image, truth pair)
y, gt = next(iter(test_dataloader))

op = make_operator(experiment.geo)

# Use gradient descent to denoise the image with the learned regularization functional
# operator is radon transform

data = fdk(y, op)


class function(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, x):
        z = model(x)
        return z


# w = op(data[0])
# w = torch.unsqueeze(w, 0)
# var_energy = 0.5*((w-y)**2).sum() + 20.000 * model(data)

# print(var_energy(data-data.grad*2, y))
from tqdm import tqdm


def gradient_descent(lmb, n_its, op, sino, x, funct, model, lr, betas):
    # x = torch.tensor(x, requires_grad = True)
    def var_energy(x, sino):
        # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
        w = op(x[0])
        w = torch.unsqueeze(w, 0)
        return 0.5 * ((w - sino) ** 2).sum() + lmb * funct(model, x).sum()

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

x, psnr_in_total = gradient_descent(
    lmb, 500, op, y, data, grad_function, model, 1e-8, 0.95
)
print(psnr_in_total)

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

# x = x.detach().cpu().numpy()
# gt = gt.detach().cpu().numpy()
# data = data.detach().cpu().numpy()

# plt.figure()
# plt.subplot(131)
# plt.imshow(x[0].T)
# # plt.colorbar()
# plt.clim(vmin=0.0, vmax=2.5)
# plt.axis("off")
# plt.title("ACNCR denoised image")
# plt.text(0, 650, f"SSIM:{ssim_gt_di:.2f} \nPSNR:{psnr_gt_di:.2f}")
# plt.subplot(132)
# plt.imshow(gt[0].T)
# # plt.colorbar()
# plt.clim(vmin=0.0, vmax=2.5)
# plt.axis("off")
# plt.title("Ground truth")
# plt.subplot(133)
# plt.imshow(data[0].T)
# # plt.colorbar()
# plt.clim(vmin=0.0, vmax=2.5)
# plt.axis("off")
# plt.title("Noisy image")
# plt.text(0, 650, f"SSIM:{ssim_gt_ni:.2f} \nPSNR:{psnr_gt_ni:.2f}")
# plt.savefig("acncr_img.png", dpi=300)


# # PRSN/SSIM for the whole test data set
# batch_size = 4
# lidc_dataloader = DataLoader(test_data, batch_size, shuffle=True)
# ssim_gt_di_all = []
# ssim_gt_ni_all = []
# psnr_gt_di_all = []
# psnr_gt_ni_all = []

# for y, gt in iter(test_dataloader):
#     image = fdk(y, op)
#     reconstruction, psnr_in_total = gradient_descent(
#         lmb, 100, op, y, image, grad_function, model, 1e-6, 0.95
#     )
#     gt_eval = gt
#     gt_eval = np.squeeze(gt_eval)
#     gt_eval = gt_eval.detach().cpu().numpy()
#     reconstruction_eval = reconstruction
#     reconstruction_eval = np.squeeze(reconstruction_eval)
#     reconstruction_eval = reconstruction_eval.detach().cpu().numpy()
#     data_eval = image.detach().cpu().numpy()
#     data_eval = np.squeeze(data_eval)

#     ssim_gt_di = ssim(
#         gt_eval, reconstruction_eval, data_range=gt_eval.max() - gt_eval.min()
#     )
#     # print('SSIM - ground truth - denoised image:', ssim_gt_di)
#     ssim_gt_ni = ssim(gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min())
#     # print('SSIM - ground truth - noisy image:', ssim_gt_ni)
#     psnr_gt_di = peak_signal_noise_ratio(
#         gt_eval, reconstruction_eval, data_range=gt_eval.max() - gt_eval.min()
#     )
#     psnr_gt_ni = peak_signal_noise_ratio(
#         gt_eval, data_eval, data_range=gt_eval.max() - gt_eval.min()
#     )
#     # print('PSNR:', psnr)
#     ssim_gt_ni_all.append(ssim_gt_ni)
#     ssim_gt_di_all.append(ssim_gt_di)
#     psnr_gt_di_all.append(psnr_gt_di)
#     psnr_gt_ni_all.append(psnr_gt_ni)

# ssim_di_all = np.mean(ssim_gt_di_all)
# ssim_ni_all = np.mean(ssim_gt_ni_all)
# psnr_di_all = np.mean(psnr_gt_di_all)
# psnr_ni_all = np.mean(psnr_gt_ni_all)


# print("SSIM - ground truth - denoised image mean", np.mean(ssim_gt_di_all))
# print("SSIM - ground truth - noisy image mean", np.mean(ssim_gt_ni_all))
# print("SSIM - ground truth - denoised image standard deviation", np.std(ssim_gt_di_all))
# print("SSIM - ground truth - noisy image standard deviation", np.std(ssim_gt_ni_all))

# print("PSNR - ground truth - denoised image mean", np.mean(psnr_gt_di_all))
# print("PSNR - ground truth - noisy image mean", np.mean(psnr_gt_ni_all))
# print("PSNR - ground truth - denoised image standard deviation", np.std(psnr_gt_di_all))
# print("PSNR - ground truth - noisy image standard deviation", np.std(psnr_gt_ni_all))


# put stuff back on the cpu, otherwise matplotlib throws an error
x = x.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()
data = data.detach().cpu().numpy()

plt.figure()
plt.subplot(131)
plt.imshow(x[0].T)
# plt.colorbar()
plt.clim(vmin=0.0, vmax=2.5)
plt.axis("off")
plt.title("ACNCR denoised image")
# plt.text(0, 650, f"SSIM:{ssim_gt_di:.2f} \nPSNR:{psnr_gt_di:.2f} \nSSIM mean:{ssim_di_all:.2f} \nPSNR mean:{psnr_di_all:.2f}")
plt.text(0, 650, f"SSIM:{ssim_gt_di:.2f} \nPSNR:{psnr_gt_di:.2f}")
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
plt.text(0, 650, f"SSIM:{ssim_gt_ni:.2f} \nPSNR:{psnr_gt_ni:.2f}")
# plt.text(0, 650, f"SSIM:{ssim_gt_ni:.2f} \nPSNR:{psnr_gt_ni:.2f} \nSSIM mean:{ssim_ni_all:.2f} \nPSNR mean:{psnr_ni_all:.2f}")
plt.savefig("acncr_img.pdf", dpi=300)
plt.savefig("acncr_img.png", dpi=300)
