from framework.kernels import kernel_vec
import numpy as np

def kernel_matrix(x, y=None, sigma=None):
    if y is None:
        y = x
    if len(x.shape) == 1:
        x = np.reshape(x, [-1, 1])
    if len(y.shape) == 1:
        y = np.reshape(y, [-1, 1])
    x_squared = np.sum(np.power(x, 2), axis=-1, keepdims=True)
    y_squared = np.sum(np.power(y, 2), axis=-1, keepdims=True).T
    xy_inner = np.matmul(x, y.T)
    kernel_input = x_squared + y_squared - 2 * xy_inner
    return np.exp(-0.5 * sigma * kernel_input)

def k_vec(x, sigma = 0.01):
    return lambda y: kernel_matrix(x, np.reshape(y, [1, -1]), sigma)

x = np.array([1,2,3])
k_vec_1 = k_vec(x)
k_vec_2 =kernel_vec(x)

print(k_vec_1(-3) - k_vec_2(-3))