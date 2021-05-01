import numpy as np
from .kernels import kernel_vec, Kernel_Matrix
from numpy.linalg import inv


def F_head(x, y, lamb = 0.01):
    n = len(x)
    k_y = kernel_vec(y)
    k_x = kernel_vec(x)
    K_y = Kernel_Matrix(y.reshape(-1, 1), y.reshape(-1, 1))
    W = inv(K_y + n * lamb * np.eye(n))
    return lambda x, y: k_y(y).T @W @k_x(x)


def MCMD(x, xp, y, yp, lamb = 0.01, lambp = 0.01):

    n = len(x)
    m = len(xp)

    k_y = kernel_vec(y)
    K_y = Kernel_Matrix(y, y)
    W_y = inv(K_y + lamb * np.eye(n))
    K_x = Kernel_Matrix(x, x)

    k_yp = kernel_vec(yp)
    K_yp = Kernel_Matrix(yp, yp)
    W_yp = inv(K_yp + lambp * np.eye(m))
    K_xxp = Kernel_Matrix(x, xp)

    K_xp = Kernel_Matrix(xp, xp)
    print(K_x.shape)
    return lambda y: k_y(y).T@W_y@K_x@W_y.T@k_y(y) \
                     - 2 * k_y(y).T@W_y @ K_xxp @ W_yp.T @k_yp(y) \
                     + k_yp(y).T@W_yp @K_xp @W_yp.T @k_yp(y)







def HSCIC(x, y, z, lamb = 0.3, kernel_x = "gaussian", kernel_y = "gaussian", kernel_z = "gaussian",
          kernel_x_args = dict(), kernel_y_args = dict(), kernel_z_args = dict()):

    if isinstance(kernel_x, str):
        kernel_x = get_kernel(kernel_x, kernel_x_args)

    if isinstance(kernel_y, str):
        kernel_y = get_kernel(kernel_y, kernel_y_args)

    if isinstance(kernel_z, str):
        kernel_z = get_kernel(kernel_z, kernel_z_args)

    n = len(z)
    k_z = kernel_vec(z, kernel_z)
    K_z = Kernel_Matrix(z, z, kernel_z)
    K_x = Kernel_Matrix(x, x, kernel_x)
    K_y = Kernel_Matrix(y, y, kernel_y)

    W = inv(K_z + lamb*n*np.eye(n))

    return lambda z: k_z(z).T @ W @ (K_x * K_y) @ W.T @ k_z(z) \
                     + (k_z(z).T @ W @ K_x @ W.T @ k_z(z)) * (k_z(z).T @ W @ K_y @ W.T @ k_z(z)) \
                     - 2*k_z(z).T @ W @ ((K_x@W.T@k_z(z)) * (K_y@W.T@k_z(z)))
