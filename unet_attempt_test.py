import torch
from torch.utils.data import DataLoader
import pathlib
from LION.models.post_processing.FBPConvNet import FBPConvNet
import matplotlib.pyplot as plt
import LION.experiments.ct_experiments as ct_experiments


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from LION.models import LIONmodel
import pathlib

# from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.experiments.ct_experiments as ct_experiments
import tomosipo as ts
from torch.nn.functional import relu
import torch.nn.functional as F
from typing import Tuple
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk

from skimage.metrics import structural_similarity as ssim
from skimage import data
from skimage.metrics import peak_signal_noise_ratio

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


# from unet_attempt_2 import UNet

#%% First run FBPConvNet_train.py to train and save model, then run this.
# % Set device:
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname = savefolder.joinpath("UNet_final_iter.pt")

# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model, _, _ = UNet.load(final_result_fname)
model.to(device)

# sample a random batch (size 1, so really just one image, truth pair)
data, gt = next(iter(test_dataloader))

op = make_operator(experiment.geo)
data = fdk(data, op)

x = model(data)


#%% Evaluation metrics

x_eval = x
x_eval = x_eval.reshape(-1)
x_eval = x_eval.detach().cpu().numpy()

gt_eval = gt
gt_eval = gt_eval.reshape(-1)
gt_eval = gt_eval.detach().cpu().numpy()
# print(gt_eval.dtype)

data_eval = data.detach().cpu().numpy()
data_eval = np.squeeze(data_eval)
data_eval = data_eval.reshape(-1)
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
plt.title("Denoised image")
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
plt.savefig("img_unet.pdf", dpi=300)

# PRSN/SSIM for the whole test data set
batch_size = 4
lidc_dataloader = DataLoader(test_data, batch_size, shuffle=True)
ssim_gt_di_all = []
ssim_gt_ni_all = []
psnr_gt_di_all = []
psnr_gt_ni_all = []

for data, gt in iter(test_dataloader):
    image = fdk(data, op)
    reconstruction = model(image)
    gt_eval = gt
    gt_eval = gt_eval.reshape(-1)
    gt_eval = gt_eval.detach().cpu().numpy()
    reconstruction_eval = reconstruction
    reconstruction_eval = reconstruction_eval.reshape(-1)
    reconstruction_eval = reconstruction_eval.detach().cpu().numpy()
    data_eval = image.detach().cpu().numpy()
    data_eval = np.squeeze(data_eval)
    data_eval = data_eval.reshape(-1)
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

print("SSIM - ground truth - denoised image mean", np.mean(ssim_gt_di_all))
print("SSIM - ground truth - noisy image mean", np.mean(ssim_gt_ni_all))
print("SSIM - ground truth - denoised image standard deviation", np.std(ssim_gt_di_all))
print("SSIM - ground truth - noisy image standard deviation", np.std(ssim_gt_ni_all))

print("PSNR - ground truth - denoised image mean", np.mean(psnr_gt_di_all))
print("PSNR - ground truth - noisy image mean", np.mean(psnr_gt_ni_all))
print("PSNR - ground truth - denoised image standard deviation", np.std(psnr_gt_di_all))
print("PSNR - ground truth - noisy image standard deviation", np.std(psnr_gt_ni_all))


# plt.figure()
# plt.subplot(121)
# plt.imshow(x[0].T)
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(data[0].T)
# plt.colorbar()
# plt.savefig("img_unet_new.png")
