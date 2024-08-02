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
        print(self.network(x).reshape(-1, 1).shape)
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

final_result_fname = savefolder.joinpath("AR_final_iter.pt")
checkpoint_fname = savefolder.joinpath("AR_check_*.pt")
validation_fname = savefolder.joinpath("AR_min_val.pt")


#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon()
# experiment = ct_experiments.clinicalCTRecon()
experiment = ct_experiments.ExtremeLowDoseCTRecon()

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
# Give paths to trained models
savefolder = pathlib.Path("/store/DAMTP/na673/")
final_result_fname = savefolder.joinpath("AR_final_iter.pt")

# % Set device:
# set up experiment model was trained on
# the same experiment should be used, results cannot be guaranteed otherwise
test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

# load trained model
model, _, _ = AR.load(final_result_fname)
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
        print("outside")
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
    lmb, 500, op, y, data, grad_function, model, 1e-6, 0.95
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
plt.title("AR denoised image")
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
plt.savefig("ar_img.pdf", dpi=300)


# PRSN/SSIM for the whole test data set
batch_size = 4
lidc_dataloader = DataLoader(test_data, batch_size, shuffle=True)
ssim_gt_di_all = []
ssim_gt_ni_all = []
psnr_gt_di_all = []
psnr_gt_ni_all = []

for data, gt in iter(test_dataloader):
    image = fdk(data, op)
    reconstruction, psnr_in_total = gradient_descent(
        lmb, 100, op, data, image, grad_function, model, 1e-6, 0.95
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
