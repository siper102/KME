from framework.kernels import kernel_vec
import numpy as np
import time
from numpy.linalg import inv
from framework2.kernel import kernel_matrix
from framework.kernels import Kernel_Matrix

def f_a(a, z):
    return np.exp(-0.5 * np.power(z, 2)) * np.sin(a * z)

n = 500
sigma_Z_mcmd = 0.1
lamb = 0.01

samples = 500
Z_1 = np.random.normal(size = samples)
Nx_1 = 0.3 * np.random.normal(size = samples)

X = np.exp(-0.5 * Z_1**2) * np.sin(2 * Z_1) + Nx_1

start = time.time()
kk = Kernel_Matrix(X, X,  "gaussian", dict(sigma = 0.1))
print(kk.shape)



