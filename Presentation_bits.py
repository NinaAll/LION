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


device = torch.device("cuda:1")
torch.cuda.set_device(device)


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


experiment = ct_experiments.ExtremeLowDoseCTRecon()

test_data = experiment.get_testing_dataset()
test_dataloader = DataLoader(test_data, 1, shuffle=True)

data, gt = next(iter(test_dataloader))

op = make_operator(experiment.geo)

fbp_rec = fdk(data, op)

data = data.detach().cpu().numpy()
gt = gt.detach().cpu().numpy()
fbp_rec = fbp_rec.detach().cpu().numpy()

plt.figure()
plt.imshow(data[0].T)
plt.axis("off")
plt.savefig("sinogram.pdf", dpi=300)

plt.figure()
plt.imshow(gt[0].T)
plt.axis("off")
plt.savefig("appurtenantground_truth.pdf", dpi=300)

plt.figure()
plt.imshow(fbp_rec[0].T)
plt.axis("off")
plt.savefig("fbp_rec.pdf", dpi=300)
