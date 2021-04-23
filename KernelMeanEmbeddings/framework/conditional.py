import numpy as np
from .kernels import get_kernel, kernel_vec, Kernel_Matrix
from numpy.linalg import inv

def F_head(x, y, lamb = 0.3, kernel_x = "gaussian", kernel_y = "gaussian",
           kernel_x_args = dict(), kernel_y_args = dict()):
    if isinstance(kernel_x, str):
        kernel_x = get_kernel(kernel_x, kernel_x_args)
    if isinstance(kernel_y, str):
        kernel_y = get_kernel(kernel_y, kernel_y_args)
    n = len(x)
    k_y = kernel_vec(y, kernel_y)
    k_x = kernel_vec(x, kernel_x)
    K_y = Kernel_Matrix(y.reshape(-1, 1), y.reshape(-1, 1), kernel_y)
    W = inv(K_y + n * lamb * np.eye(n))
    return lambda x, y: k_y(y).T @W @k_x(x)


def MCMD(x, xp, y, yp, kernel_x = "gaussian", kernel_y = "gaussian",
         kernel_x_args = dict(), kernel_y_args = dict(), lamb = 0.2, lambp = 0.2):


    if isinstance(kernel_x, str):
        kernel_x = get_kernel(kernel_x, kernel_x_args)

    if isinstance(kernel_y, str):
        kernel_y = get_kernel(kernel_y, kernel_y_args)
    n = len(x)
    m = len(xp)

    k_y = kernel_vec(y, kernel_y)
    W_y = inv(Kernel_Matrix(y, y, kernel_y) + n * lamb * np.eye(n))
    K_x = Kernel_Matrix(x, x, kernel_x)

    k_yp = kernel_vec(yp, kernel_y)
    W_yp = inv(Kernel_Matrix(yp, yp, kernel_y) + m * lambp * np.eye(m))
    K_xxp = Kernel_Matrix(x, xp, kernel_x)

    K_xp = Kernel_Matrix(xp, xp, kernel_x)

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



