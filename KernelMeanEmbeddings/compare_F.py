from framework.Embedding.F_estimator import F_estimator
from framework.kernels import Kernel_Matrix
from F_jun import witness
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


n = 500
Z = np.random.normal(size=n)
Z_prime = np.random.normal(size=n)

X = np.exp(-1/2*Z**2) * np.sin(2*Z) + 0.3 * np.random.normal(size=n)
X_prime_same = np.exp(-1/2*Z_prime**2) * np.sin(2*Z_prime) + 0.3 * np.random.normal(size=n)

K_Z_mcmd = kernel_matrix(Z, sigma=0.1)
K_Z_prime_mcmd = kernel_matrix(Z_prime, sigma=0.1)
W_mcmd = np.linalg.inv(K_Z_mcmd + 0.01 * np.identity(n))
W_prime_mcmd = np.linalg.inv(K_Z_prime_mcmd + 0.01 * np.identity(n))

f_1 = F_estimator()
f_2 = F_estimator()
f_1.fit(X, Z)
f_2.fit(X_prime_same, Z_prime)

x_arguments = np.arange(-3, 3, 0.1)
z_arguments_mcmd = np.arange(-3, 3, 0.1)

wit_same = np.asarray([[witness(p, q, X, X_prime_same, Z, Z_prime, W_mcmd, W_prime_mcmd, 0.1, 0.1)
                        for p in z_arguments_mcmd] for q in x_arguments])

wit_same_2 = f_1(x_arguments, z_arguments_mcmd) - f_2(x_arguments, z_arguments_mcmd)

print(wit_same - wit_same_2)
