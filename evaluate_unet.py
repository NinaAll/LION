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
        geometry_parameters: ct.Geometry = None,
        model_parameters: LIONParameter = None,
    ):  # always the input you need when you do a LION model
        super().__init__(model_parameters, geometry_parameters)
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


# Change default parameters:


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
experiment = ct_experiments.clinicalCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

# remove files = rm /store/DAMTP/na673/UNet*

#%% FBP

from typing import Tuple
import torch
import tomosipo as ts
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk
import tomosipo as ts


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


#%% Define DataLoader
# Use the same amount of training
batch_size = 4
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

# Change default parameters:
# parameters = UNet.default_parameters()
# parameters.in_channels = 2
# model = UNet(parameters).to(device)

model = UNet().to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 3
train_param.learning_rate = 1e-3
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
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

model, optimiser, start_epoch, total_loss, _ = UNet.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

op = make_operator(experiment.geo)

#%% train
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

loss_valid = []
loss_train = []

for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

    for sinogram, target_reconstruction in tqdm(lidc_dataloader):

        image = fdk(sinogram, op)
        # print(image.shape)
        optimiser.zero_grad()
        reconstruction = model(image)

        loss = loss_fcn(reconstruction, target_reconstruction)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
        scheduler.step()
    total_loss[epoch] = train_loss
    loss_train.append(train_loss)
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):

        image = fdk(sinogram, op)

        reconstruction = model(image)
        loss = loss_fcn(target_reconstruction, reconstruction)
        valid_loss += loss.item()
    loss_valid.append(valid_loss)
    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
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


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
#%% First run FBPConvNet_train.py to train and save model, then run this.
# % Set device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname = savefolder.joinpath("UNet_final_iter.pt")

# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
experiment = ct_experiments.clinicalCTRecon()
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


# put stuff back on the cpu, otherwise matplotlib throws an error
x = x.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()
data = data.detach().cpu().numpy()

plt.figure()
plt.subplot(121)
plt.imshow(x[0].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(gt[0].T)
plt.colorbar()
plt.savefig("img_unet.png")


plt.figure()
plt.subplot(121)
plt.imshow(x[0].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(data[0].T)
plt.colorbar()
plt.savefig("img_unet_new.png")
