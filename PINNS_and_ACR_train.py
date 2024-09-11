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


class LinearBlock(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim=0)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        return x


class convexnet(nn.Module):
    def __init__(
        self,
        layer_list,
        n_channels=16,
        kernel_size=5,
        n_layers=5,
        convex=True,
        n_chan=1,
    ):
        super().__init__()
        self.device = torch.device("cuda:1")
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
        self.input_layer = nn.utils.weight_norm(
            nn.Linear(layer_list[0], layer_list[1]), dim=0
        )
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.output_layer = nn.Linear(layer_list[-2], layer_list[-1])
        self.initialize_weights()

    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) - 1):
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            layers.append(block)
        return nn.Sequential(*layers)

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

    def forward(self, t, x):
        self.clamp_weights()
        # x = x.view(2,1,512*512)
        x = x.view(-1)
        t = torch.tensor([t], device=self.device)
        z = torch.cat((t, x), dim=0)  # Prepend the scalar to the tensor
        t_0, x_0 = z[0], z[1:]

        z = self.input_layer(z)
        z = torch.tanh(z)
        z = self.hidden_layers(z)
        z = self.output_layer(z)
        # print(x.shape)
        # x = torch.absolute(x_0)+torch.tanh(t_0)*x
        z = (1 - torch.tanh(t_0)) * torch.absolute(x_0) + torch.tanh(t_0) * z
        # x = (1-torch.tanh(t_0)) * (x_0)**2 + torch.tanh(t_0)*x
        # x = torch.tanh(t_0)*x
        # return -x+torch.absolute(x_0)
        return z


class ACR(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):
        super().__init__(model_parameters, geometry_parameters)

        full_chan = 16  # sparse
        num_t = 256
        num_x = 256
        # num_t = 128
        # num_x = 128
        num_epochs = 50000
        num_hidden = 4
        num_nodes = 1
        layer_list = [524289] + num_hidden * [num_nodes] + [1]
        # self.convnet=convexnet(layer_list, n_channels=16,n_chan=full_chan*4)
        self.convnet = convexnet(layer_list, n_channels=16, n_chan=1, n_layers=10)
        self.op = make_operator(experiment.geo)
        self.nw = power_method(self.op)

    # def init_weights(self,m):
    #     pass

    def clamp_weights(self):
        self.convnet.clamp_weights()
        # self.convnet_data.clamp_weights()

    def wei_dec(self):
        self.convnet.wei_dec()
        # self.convnet_data.wei_dec()
        self.pinn.wei_dec()

    def forward(self, image, t):
        # output = self.convnet(self.smooth(image)) + self.convnet_data(data_img)
        # w = self.op(image[0])
        # w = torch.unsqueeze(w, 0)
        # sinogram = w/self.nw
        # print('hi')
        # print(sinogram.shape)
        # print(sinogram.max(),sinogram.min())
        # output = self.convnet(self.smooth(sinogram/(config.fwd_op_norm)))# + self.convnet_data(image)
        output = self.convnet(t, image)
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

model = ACR(geometry_parameters=experiment.geo).to(device)


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

model, optimiser, start_epoch, total_loss, _ = ACR.load_checkpoint_if_exists(
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

        loss = loss_fcn(model, reconstruction, target_reconstruction, 1)

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

# def forward(self, x, grady=False):
#     # for layer in range(self.n_layers):
#     #     print((self.wzs[layer].weight.data<0).sum())
#     # if self.convex:
#     self.clamp_weights() # makes sure that it is convex

#     z = self.leaky_relu(self.wxs[0](x))
#     for layer_idx in range(self.n_layers):
#         z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx+1](x))
#     z = self.final_conv2d(z)
#     net_output = z.view(z.shape[0], -1).mean(dim=1,keepdim=True)
#     # assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape}"
#     # print(net_output.shape)
#     # print(net_output.mean().item(),foe_out.mean().item(),l2_out.mean().item())
#     return net_output


# class ACR(LIONmodel.LIONmodel):
#     def __init__(
#         self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
#     ):

#         super().__init__(model_parameters, geometry_parameters)
#         self._make_operator()

#         self.network = convexnet()

#         # First Conv
#         self.estimate_lambda()
#         self.step_amounts = torch.tensor([150.0])
#         self.op_norm = power_method(self.op)
#         self.model_parameters.step_size = 0.2 / (self.op_norm) ** 2

#     def forward(self, x, t):
#         # x = fdk(self.op, x)
#         x = self.normalise(x)
#         # print(self.pool(z).mean(),self.L2(z).mean())
#         print(self.network(x, t).reshape(-1, 1).shape)
#         return self.network(x, t).reshape(
#             -1, 1
#         )

#     def estimate_lambda(self, dataset=None):
#         self.lamb = 1.0
#         if dataset is None:
#             self.lamb = 1.0
#         else:
#             residual = 0.0
#             for index, (data, target) in enumerate(dataset):
#                 residual += torch.norm(
#                     self.AT(self.A(target) - data), dim=(2, 3)
#                 ).mean()
#                 # residual += torch.sqrt(((self.AT(self.A(target) - data))**2).sum())
#             self.lamb = residual.mean() / len(dataset)
#         print("Estimated lambda: " + str(self.lamb))

#     # def output(self, x):
#     # return self.AT(x)

#     def var_energy(self, x, y):
#         # return torch.norm(x) + 0.5*(torch.norm(self.A(x)-y,dim=(2,3))**2).sum()#self.lamb * self.forward(x).sum()
#         return 0.5 * ((self.A(x) - y) ** 2).sum() + self.lamb * self.forward(x).sum()

#     def identity(self, t):
#         return t

#     ### What is the difference between .sum() and .mean()??? idfk but PSNR is lower when I do .sum

#     def output(self, y, t, truth=None):
#         # wandb.log({'Eearly_stopping_steps': self.step_amounts.mean().item(), 'Eearly_stopping_steps_std': self.step_amounts.std().item()})
#         x0 = []
#         device = torch.cuda.current_device()
#         for i in range(y.shape[0]):
#             x0.append(fdk(self.op, y[i]))
#         x = torch.stack(x0)
#         # print(x.shape)
#         # print(x.min(),x.max())
#         # print(my_psnr(truth.detach().to(device),x.detach()).mean(),my_ssim(truth.detach().to(device),x.detach()).mean())
#         x = torch.nn.Parameter(x)  # .requires_grad_(True)

#         t = self.identity(t)

#         optimizer = torch.optim.SGD(
#             [x, t], lr=self.model_parameters.step_size, momentum=0.5
#         )  # self.model_parameters.momentum)
#         lr = self.model_parameters.step_size
#         prevpsn = 0
#         curpsn = 0
#         for j in range(self.model_parameters.no_steps):
#             # print(x.min(),x.max())
#             # data_misfit=self.A(x)-y
#             # data_misfit_grad = self.AT(data_misfit)

#             optimizer.zero_grad()
#             # reg_func=self.lamb * self.forward(x).mean()
#             # reg_func.backward()
#             # print(x.requires_grad, reg_func.requires_grad)
#             energy = self.var_energy(x, y)
#             energy.backward()
#             while (
#                 self.var_energy(x - x.grad * lr, y)
#                 > energy - 0.5 * lr * (x.grad.norm(dim=(2, 3)) ** 2).mean()
#             ):
#                 lr = self.model_parameters.beta_rate * lr
#                 # print('decay')
#             for g in optimizer.param_groups:
#                 g["lr"] = lr
#             # x.grad+=data_misfit_grad
#             if truth is not None:
#                 loss = WGAN_HJ_loss()(
#                     x.detach(), truth.detach().to(device)
#                 )
#                 psnr_val = my_psnr(truth.detach().to(device), x.detach()).mean()
#                 ssim_val = my_ssim(truth.detach().to(device), x.detach()).mean()
#                 # wandb.log({'MSE Loss': loss.item(),'SSIM':ssim_val,'PSNR':psnr_val})
#                 # wandb.log({'MSE Loss'+str(self.model_parameters.step_size): loss.item(),'SSIM'+str(self.model_parameters.step_size):ssim_val,'PSNR'+str(self.model_parameters.step_size):psnr_val})
#                 print(
#                     f"{j}: SSIM: {my_ssim(truth.to(device).detach(),x.detach())}, PSNR: {my_psnr(truth.to(device).detach(),x.detach())}, Energy: {energy.detach().item()}"
#                 )

#                 #     if(self.args.outp):
#                 #         print(j)
#                 prevpsn = curpsn
#                 curpsn = psnr_val
#                 # if(curpsn<prevpsn):
#                 #     self.step_amounts = torch.cat((self.step_amounts,torch.tensor([j*1.0])))
#                 #     return x.detach()
#             elif j > self.step_amounts.mean().item():
#                 # print('only for testing')
#                 x.clamp(min=0.0)
#                 return x.detach()
#             elif lr * self.op_norm**2 < 1e-3:
#                 x.clamp(min=0.0)
#                 return x.detach()
#             optimizer.step()
#             x.clamp(min=0.0)
#         return x.detach(), t

#     def normalise(self, x):
#         return (x - self.model_parameters.xmin) / (
#             self.model_parameters.xmax - self.model_parameters.xmin
#         )

#     def unnormalise(self, x):
#         return (
#             x * (self.model_parameters.xmax - self.model_parameters.xmin)
#             + self.model_parameters.xmin
#         )

#     @staticmethod
#     def default_parameters():
#         param = LIONParameter()
#         param.channels = 16
#         param.kernel_size = 5
#         param.stride = 1
#         param.relu_type = "LeakyReLU"
#         param.layers = 5
#         param.early_stopping = False
#         param.no_steps = 150
#         param.step_size = 1e-6
#         param.momentum = 0.5
#         param.beta_rate = 0.95
#         param.xmin = 0.0
#         param.xmax = 1.0
#         return param

#     @staticmethod
#     def cite(cite_format="MLA"):
#         if cite_format == "MLA":
#             print("Mukherjee, Subhadip, et al.")
#             print('"Data-Driven Convex Regularizers for Inverse Problems."')
#             print(
#                 "ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024"
#             )
#             print("arXiv:2008.02839 (2020).")
#         elif cite_format == "bib":
#             string = """
#             @inproceedings{mukherjee2024data,
#             title={Data-Driven Convex Regularizers for Inverse Problems},
#             author={Mukherjee, S and Dittmer, S and Shumaylov, Z and Lunz, S and {\"O}ktem, O and Sch{\"o}nlieb, C-B},
#             booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
#             pages={13386--13390},
#             year={2024},
#             organization={IEEE}
#             }
#             """
#             print(string)
#         else:
#             raise AttributeError(
#                 'cite_format not understood, only "MLA" and "bib" supported'
#             )
