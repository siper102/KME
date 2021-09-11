import numpy as np
import matplotlib.pyplot as plt

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


def hscic(arg, k_x, k_y, z, w, sigma_z):
    kk_z = kernel_matrix(z, np.reshape(arg, [1, -1]), sigma_z)
    first = kk_z.T @ w @ np.multiply(k_x, k_y) @ w @ kk_z
    second = kk_z.T @ w @ np.multiply(k_x @ w @ kk_z, k_y @ w @ kk_z)
    third = (kk_z.T @ w @ k_x @ w @ kk_z) * (kk_z.T @ w @ k_y @ w @ kk_z)
    return (first - 2 * second + third)[0, 0]

def f_a(a, z):
    return np.exp(-0.5 * np.power(z, 2)) * np.sin(a * z)

n = 500
sigma_X = 0.1
sigma_Y = 0.1
sigma_Z_mcmd = 0.1
sigma_Z_hscic = 0.1
lamb = 0.01

Z = np.random.normal(size=n)

K_Z_hscic = kernel_matrix(Z, sigma=sigma_Z_hscic)
W_hscic = np.linalg.inv(K_Z_hscic + lamb * np.identity(n))

X_hscic = f_a(2, Z) * np.random.normal(size=n)
K_X_hscic = kernel_matrix(X_hscic, sigma=sigma_X)
Y_ind = f_a(2, Z) * np.random.normal(size=n)
K_Y_ind = kernel_matrix(Y_ind, sigma=sigma_Y)
Y_dep = f_a(2, Z) * np.random.normal(size=n) + 0.2 * X_hscic
K_Y_dep = kernel_matrix(Y_dep, sigma=sigma_Y)
Y_dep_prime = f_a(2, Z) * np.random.normal(size=n) + 0.4 * X_hscic
K_Y_dep_prime = kernel_matrix(Y_dep_prime, sigma=sigma_Y)


K_Z_hscic = kernel_matrix(Z, sigma=sigma_Z_hscic)
W_hscic = np.linalg.inv(K_Z_hscic + lamb * np.identity(n))
X_hscic = f_a(2, Z) * np.random.normal(size=n)
K_X_hscic = kernel_matrix(X_hscic, sigma=sigma_X)
Y_ind = f_a(2, Z) * np.random.normal(size=n)
K_Y_ind = kernel_matrix(Y_ind, sigma=sigma_Y)
Y_dep = f_a(2, Z) * np.random.normal(size=n) + 0.2 * X_hscic
K_Y_dep = kernel_matrix(Y_dep, sigma=sigma_Y)
Y_dep_prime = f_a(2, Z) * np.random.normal(size=n) + 0.4 * X_hscic
K_Y_dep_prime = kernel_matrix(Y_dep_prime, sigma=sigma_Y)

z_arguments_hscic = np.arange(-2, 2, 0.1)
hscic_ind = np.asarray([hscic(p, K_X_hscic, K_Y_ind, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep = np.asarray([hscic(p, K_X_hscic, K_Y_dep, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])
hscic_dep_prime = np.asarray([hscic(p, K_X_hscic, K_Y_dep_prime, Z, W_hscic, sigma_Z_hscic) for p in z_arguments_hscic])

plt.plot(z_arguments_hscic, hscic_ind, label="HSCIC(X,Y_ind|Z)", linewidth=5, color="orange")
plt.plot(z_arguments_hscic, hscic_dep, label="HSCIC(X,Y_dep|Z)", linestyle="dashed", linewidth=5, color="green")
plt.plot(z_arguments_hscic, hscic_dep_prime, label="HSCIC(X,Y'_dep|Z)", linestyle="dotted", linewidth = 5,     color="red")
plt.legend(fontsize=13)
plt.title('HSCIC values', fontsize=20)
plt.xlabel("z", fontsize=20)
plt.ylabel("HSCIC", fontsize=20)
plt.show()