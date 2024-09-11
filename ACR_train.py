# This file is part of LION library
# License : BSD-3
#
# Author  : Zakhar Shumaylov, Subhadip Mukherjee
# Modifications: Ander Biguri, Zakhar Shumaylov
# =============================================================================


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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

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


# Learning a regularization functional, i.e. train the AR:

#%% Loss function


class WGAN_gradient_penalty_loss(nn.Module):
    def __init__(self, mu=10.0 * 1e-2):
        self.mu = mu
        super().__init__()

    def forward(self, model, data_marginal_noisy, data_marginal_real):
        """Calculates the gradient penalty loss for WGAN GP"""
        real_samples = data_marginal_real
        # print(real_samples.shape)
        fake_samples = data_marginal_noisy
        # print(fake_samples.shape)
        # fake_samples=fake_samples[:,:,None, None]
        # print(fake_samples.shape)
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
            real_samples
        )
        # print(alpha.shape)
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

final_result_fname = savefolder.joinpath("ACR_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ACR_check_*.pt")
validation_fname = savefolder.joinpath("ACR_min_val.pt")

#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
# experiment = ct_experiments.clinicalCTRecon()
experiment = ct_experiments.ExtremeLowDoseCTRecon()

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

batch_size = 5
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

from LION.models.learned_regularizer.ACR import ACR

default_parameters = ACR.default_parameters()
model = ACR(experiment.geo, default_parameters).to(device)

# model = ACR(geometry_parameters=experiment.geo).to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = WGAN_gradient_penalty_loss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 25
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "WGAN_gradient_penalty_loss"
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

model, optimiser, start_epoch, total_loss, _ = ACR.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

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

        loss = loss_fcn(model, reconstruction, target_reconstruction)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
        # scheduler.step()
    total_loss[epoch] = train_loss
    loss_train.append(train_loss / len(lidc_dataset))
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):

        image = fdk(sinogram, op)

        reconstruction = image
        loss = loss_fcn(model, reconstruction, target_reconstruction)
        valid_loss += loss.item()
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
plt.savefig("learning_curve.png")

model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
