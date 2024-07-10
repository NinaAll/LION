import LION
import LION.utils.math
import LION.utils.parameter
import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import LION.CTtools.ct_utils as ct
import pathlib

sino = ct.forward_projection_fan(phantom, geo)
sino = A(phantom)


def gradient_descent(lmb, A, b, x0, n_its):
    x = x0
    for k in range(n_its):
        x = x + lmb * A.T(A(x) - b)
    return x
