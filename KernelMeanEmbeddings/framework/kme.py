import numpy as np
from .kernels import get_kernel, kernel_vec, Kernel_Matrix

def kernel_schetzer(x, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)
    k_x = kernel_vec(x, kernel)
    return lambda y: np.mean(k_x(y))



def MMD(x, y, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)
    n = len(x)
    m = len(y)

    K_x = Kernel_Matrix(x, x, kernel=kernel)
    K_y = Kernel_Matrix(y, y, kernel=kernel)
    K_xy = Kernel_Matrix(x, y, kernel=kernel)

    return 1/n**2 * np.sum(K_x) + 1/m**2 * np.sum(K_y) - 2/(m*n) * np.sum(K_xy)


def witness(x, y, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)

    return lambda t: (kernel_schetzer(x, kernel = kernel)(t) -  kernel_schetzer(y, kernel = kernel)(t))