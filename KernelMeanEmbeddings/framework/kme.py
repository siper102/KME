import numpy as np
from .kernels import get_kernel, kernel_vec

def kernel_schetzer(x, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)
    k_x = kernel_vec(x, kernel)
    return lambda y: np.mean(k_x(y))

