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

from LION.models.learned_regularizer.ACR_HJ import ACR_HJ

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


class WGAN_HJ_loss(nn.Module):
    def __init__(self, mu=10.0 * 1e-2):
        self.mu = mu
        super().__init__()

    def forward(self, model, data_noisy, data_real, T):
        real_samples = data_real
        fake_samples = data_noisy
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(
            real_samples
        )
        interpolates_x = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        # t = torch.from_numpy(np.random.uniform(0.0, T))
        t = torch.distributions.Uniform(0, T).sample().requires_grad_(True)
        net_interpolates = model(interpolates_x, t)

        print(net_interpolates.shape)

        fake = (
            torch.Tensor(real_samples.shape[0], 1)
            .fill_(1.0)
            .type_as(real_samples)
            .requires_grad_(False)
        )

        gradients_x = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates_x,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        print(gradients_x.shape)

        gradients_t = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=t,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        print(gradients_t.shape)

        gradients_x = gradients_x.view(gradients_x.size(0), -1)
        gradients_t = gradients_t.view(gradients_t.size(0), -1)

        wgan_loss = (
            model(real_samples).mean()
            - model(fake_samples).mean()
            + self.mu * (((gradients_x.norm(2, dim=1) - 1)) ** 2).mean()
        )
        pinn_loss = gradients_t + 1 / 2 * gradients_x.norm(2, dim=1)
        return self.mu * pinn_loss + wgan_loss


#%% PINN


def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph=True)[0]
    return derivative


def hamilton_jacobi_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    nu = 0  # 10000
    e = u_t + 1 / 2 * (u_x) ** 2 - nu * u_xx
    return e


def resplot(x, t, t_data, x_data, Exact, u_pred):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, Exact[:, 0], "-")
    plt.plot(x, u_pred[:, 0], "--")
    plt.legend(["Reference", "Prediction"])
    plt.title("Initial condition ($t=0$)")

    plt.subplot(2, 2, 2)
    t_step = int(0.25 * len(t))
    plt.plot(x, Exact[:, t_step], "-")
    plt.plot(x, u_pred[:, t_step], "--")
    plt.legend(["Reference", "Prediction"])
    plt.title("$t=0.25$")

    plt.subplot(2, 2, 3)
    t_step = int(0.5 * len(t))
    plt.plot(x, Exact[:, t_step], "-")
    plt.plot(x, u_pred[:, t_step], "--")
    plt.legend(["Reference", "Prediction"])
    plt.title("$t=0.5$")

    plt.subplot(2, 2, 4)
    t_step = int(0.99 * len(t))
    plt.plot(x, Exact[:, t_step], "-")
    plt.plot(x, u_pred[:, t_step], "--")
    plt.legend(["Reference", "Prediction"])
    plt.title("$t=0.99$")
    plt.show()
    plt.close()


num_t = 256
num_x = 256
# num_t = 128
# num_x = 128
num_epochs = 50000
num_hidden = 4
num_nodes = 128
lr = 1e-2

print("Operation mode: ", device)

eq = "HJPDE"

#% Training both PINN and ACR at the same time:

# % Chose device:
# device = torch.device("cuda:1")
# torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("PINNS_and_ACR_final_iter.pt")
checkpoint_fname = savefolder.joinpath("PINNS_and_ACR_check_*.pt")
validation_fname = savefolder.joinpath("PINNS_and_ACR_min_val.pt")

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

# layer_list = [2] + num_hidden * [num_nodes] + [1]

default_parameters = ACR_HJ.default_parameters()
model = ACR_HJ(experiment.geo, default_parameters).to(device)

# model = ACR(geometry_parameters=experiment.geo).to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
# loss_fcn = WGAN_HJ_loss()
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


loss_fcn = WGAN_gradient_penalty_loss()
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
    print(savefolder.joinpath(final_result_fname))
    exit()

model, optimiser, start_epoch, total_loss, _ = ACR_HJ.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

op = make_operator(experiment.geo)


#%% train
loss_valid = []
loss_train_pinn = []
loss_train_acr = []

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

        loss = loss_fcn(model, reconstruction, target_reconstruction)

        loss.backward()

        l = loss.item()

        train_loss += loss.item()

        optimiser.step()
        # scheduler.step()
    total_loss[epoch] = train_loss
    loss_train_pinn.append(train_loss / len(lidc_dataset))
    loss_train_acr.append(train_loss / len(lidc_dataset))
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

plt.plot(x, loss_train_pinn, label="train_pinn")
plt.plot(x, loss_train_acr, label="train_acr")
plt.plot(x, loss_valid, label="valid")
plt.legend()
plt.savefig("learning_curve_pinns_and_acr.pdf")

model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
