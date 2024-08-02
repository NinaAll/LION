import numpy as np
from numpy import linalg as LA
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import LION.CTtools.ct_utils as ct

print(torch.version.cuda)

phantom = np.ones((512, 512)) * -1000  # lets assume its 512^2
phantom[200:250, 200:250] = 300
phantom = np.expand_dims(phantom, 0)
phantom_mu = ct.from_HU_to_mu(phantom)
phantom = phantom_mu
plt.figure()
plt.imshow(phantom[0].T)
plt.colorbar()
plt.savefig("phantom.png")

vg = ts.volume(shape=(1, *phantom.shape[1:]), size=(5, 300, 300))
pg = ts.cone(
    angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
)
A = ts.operator(vg, pg)

dev = torch.device("cuda")
phantom = torch.from_numpy(phantom).to(dev)
sino = A(phantom)

sino_noisy = ct.sinogram_add_noise(
    sino, I0=10000, sigma=5, cross_talk=0.05, flat_field=None, dark_field=None
)
sino = sino.detach().cpu().numpy()
sino_noisy = sino_noisy.detach().cpu().numpy()

plt.figure()
plt.subplot(121)
plt.imshow(sino[0].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(sino_noisy[0].T)
plt.colorbar()
plt.savefig("Sino.png")

lmb = -0.001
x = np.zeros((1, 512, 512))
x = x + lmb * A.T(A(x) - sino)
plt.figure()
plt.imshow(x[0].T)
plt.savefig("x.png")


def gradient_descent(lmb, n_its, A=A, sino=sino, x=np.zeros((1, 512, 512))):
    for k in range(n_its):
        x = x + lmb * A.T(A(x) - sino)
    return x


xs = gradient_descent(lmb=-0.00000001, n_its=10000)
plt.figure()
plt.imshow(xs[0].T)
plt.colorbar()
plt.savefig("xs.png")
