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


# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/na673/")

final_result_fname = savefolder.joinpath("ACNCR_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ACNCR_check_*.pt")
validation_fname = savefolder.joinpath("ACNCR_min_val.pt")

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

model = ACNCR(geometry_parameters=experiment.geo).to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = loss_function()
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

model, optimiser, start_epoch, total_loss, _ = ACNCR.load_checkpoint_if_exists(
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
        # print(reconstruction.shape)

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
plt.savefig("learning_curve_acncr.pdf")

model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
