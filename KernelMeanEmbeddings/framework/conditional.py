import numpy as np
from .kernels import kernel_vec, Kernel_Matrix
from numpy.linalg import inv


def MCMD(x, xp, y, yp, lamb = 0.01, lambp = 0.01):

    n = len(x)
    m = len(xp)

    k_y = kernel_vec(y)
    K_y = Kernel_Matrix(y, y)
    W_y = inv(K_y + n * lamb * np.eye(n))
    K_x = Kernel_Matrix(x, x)

    k_yp = kernel_vec(yp)
    K_yp = Kernel_Matrix(yp, yp)
    W_yp = inv(K_yp + n * lambp * np.eye(m))
    K_xxp = Kernel_Matrix(x, xp)

    K_xp = Kernel_Matrix(xp, xp)

    return lambda y: k_y(y).T@W_y@K_x@W_y.T@k_y(y) \
                     - 2 * k_y(y).T@W_y @ K_xxp @ W_yp.T @k_yp(y) \
                     + k_yp(y).T@W_yp @K_xp @W_yp.T @k_yp(y)

