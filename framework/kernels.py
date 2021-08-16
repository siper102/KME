import numpy as np


def gauss_kernel(sigma = 0.1):
    return lambda x, y: np.exp(-1 * 1 / 2 * sigma * np.linalg.norm(x - y)**2)


def Kernel_Matrix(X, Y, sigma = 0.1):
    X_new = np.reshape(X, [-1, 1])
    Y_new = np.reshape(Y, [-1, 1])
    x_squared = np.sum(np.power(X_new, 2), axis=-1, keepdims=True)
    y_squared = np.sum(np.power(Y_new, 2), axis=-1, keepdims=True).T
    xy_inner = np.matmul(X_new, Y_new.T)
    kernel_input = x_squared + y_squared - 2 * xy_inner
    return np.exp(-0.5 * sigma * kernel_input)

def kernel_vec(X, sigma = 0.1):
    return lambda y: Kernel_Matrix(X, np.reshape(y, [1, -1]), sigma)
