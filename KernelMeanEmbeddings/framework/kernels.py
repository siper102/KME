import numpy as np


def gauss_kernel(sigma = 0.5):
    return lambda x, y: np.exp(1 / (2*sigma**2) * - np.linalg.norm(x - y)**2)

def laplace_kernel(sigma = 0.5):
    return lambda x, y: np.exp(-1/sigma * np.linalg.norm(x - y))

def polynomial_kernel(p = 2):
    return lambda x, y: (x.T@y + 1)**p

def get_kernel(string, kernel_args):
    if string == "gaussian":
        return gauss_kernel(**kernel_args)
    elif string == "laplace":
        return laplace_kernel(**kernel_args)
    elif string == "poly":
        return polynomial_kernel(**kernel_args)
    else:
        return "Error"


def Kernel_Matrix(x, y, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)
    K = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            K[i, j] = kernel(x[i], y[j])
    return K

def k_x(x, y, kernel):
    ret = np.zeros_like(x)
    if isinstance(y, (float, int)):
        y = np.ones_like(x) * y
    for i in range(len(x)):
        ret[i] = kernel(x[i], y[i])
    return ret

def kernel_vec(x, kernel = "gaussian", kernel_args = dict()):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel, kernel_args)
    return lambda y: k_x(x, y, kernel)