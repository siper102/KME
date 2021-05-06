import numpy as np
import matplotlib.pyplot as plt
# sigma. If y=None, the Gram matrix is computed based on x.
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


def witness(arg_z, arg_x, x, x_prime, z, z_prime, w, w_prime, sigma_x, sigma_z):
    kk_x = kernel_matrix(x, np.reshape(arg_x, [1, -1]), sigma_x)
    kk_x_prime = kernel_matrix(x_prime, np.reshape(arg_x, [1, -1]), sigma_x)
    kk_z = kernel_matrix(z, np.reshape(arg_z, [1, -1]), sigma_z)
    kk_z_prime = kernel_matrix(z_prime, np.reshape(arg_z, [1, -1]), sigma_z)
    return (kk_z.T @ w @ kk_x - kk_z_prime.T @ w_prime @ kk_x_prime)[0, 0]

def f_a(a, z):
    return np.exp(-0.5 * np.power(z, 2)) * np.sin(a * z)


# Set the hyperparameters.
np.random.seed(22)
n = 500
sigma_X = 0.1
sigma_Y = 0.1
sigma_Z_mcmd = 0.1
sigma_Z_hscic = 0.1
lamb = 0.01

Z = np.random.normal(size=n)
Z_prime = np.random.normal(size=n)
K_Z_mcmd = kernel_matrix(Z, sigma=sigma_Z_mcmd)
K_Z_prime_mcmd = kernel_matrix(Z_prime, sigma=sigma_Z_mcmd)
W_mcmd = np.linalg.inv(K_Z_mcmd + lamb * np.identity(n))
W_prime_mcmd = np.linalg.inv(K_Z_prime_mcmd + lamb * np.identity(n))
X = f_a(2, Z) + 0.3 * np.random.normal(size=n)
K_X = kernel_matrix(X, sigma=sigma_X)
X_prime_same = f_a(2, Z_prime) + 0.3 * np.random.normal(size=n)
K_X_prime_same = kernel_matrix(X_prime_same, sigma=sigma_X)
K_X_X_prime_same = kernel_matrix(X, X_prime_same, sigma_X)
X_prime_diff = Z_prime + 0.3 * np.random.normal(size=n)
K_X_prime_diff = kernel_matrix(X_prime_diff, sigma=sigma_X)
K_X_X_prime_diff = kernel_matrix(X, X_prime_diff, sigma_X)


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

z_arguments_mcmd = np.arange(-3, 3, 0.1)
x_arguments = np.arange(-3, 3, 0.1)
zz = np.linspace(-3, 3, 60)
xx = np.linspace(-3, 3, 60)
ZZ, XX = np.meshgrid(zz, xx)
wit_same = np.asarray([[witness(p, q, X, X_prime_same, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_X, sigma_Z_mcmd)
                        for p in z_arguments_mcmd] for q in x_arguments])

wit_diff = np.asarray([[witness(p, q, X, X_prime_diff, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_X, sigma_Z_mcmd)
                        for p in z_arguments_mcmd] for q in x_arguments])
#maxi = max(np.max(wit_same), np.max(wit_diff))
#mini = min(np.min(wit_same), np.min(wit_diff))

im_same = plt.imshow(wit_same, cmap='viridis', interpolation='nearest', #vmin=mini, vmax=maxi,
                     extent=[z_arguments_mcmd[0], z_arguments_mcmd[-1], x_arguments[0], x_arguments[-1]])

print(wit_same.shape)
plt.title("Witness between X and X'_same", fontsize=20)
plt.show()