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


# The following function evaluates the function M^2 conditioned on arg (see Section 5.1)
def mcmd(arg, k_x, k_x_prime, k_x_x_prime, z, z_prime, w, w_prime, sigma_z):
    kk_z = kernel_matrix(z, np.reshape(arg, [1, -1]), sigma_z)
    kk_z_prime = kernel_matrix(z_prime, np.reshape(arg, [1, -1]), sigma_z)
    first = kk_z.T @ w @ k_x @ w @ kk_z
    second = kk_z.T @ w @ k_x_x_prime @ w_prime @ kk_z_prime
    third = kk_z_prime.T @ w_prime @ k_x_prime @ w_prime @ kk_z_prime
    return (first - 2 * second + third)[0, 0]

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

# In the following, we simulate our data for MCMD experiments, as well as their respective kernel matrices.
Z = np.random.normal(size=n)
Z_prime = np.random.normal(size=n)
X = f_a(2, Z) + 0.3 * np.random.normal(size=n)
X_prime_same = f_a(2, Z_prime) + 0.3 * np.random.normal(size=n)

K_Z_mcmd = kernel_matrix(Z, sigma=sigma_Z_mcmd)
K_Z_prime_mcmd = kernel_matrix(Z_prime, sigma=sigma_Z_mcmd)

W_mcmd = np.linalg.inv(K_Z_mcmd + lamb * np.identity(n))
W_prime_mcmd = np.linalg.inv(K_Z_prime_mcmd + lamb * np.identity(n))
K_X = kernel_matrix(X, sigma=sigma_X)
K_X_prime_same = kernel_matrix(X_prime_same, sigma=sigma_X)
K_X_X_prime_same = kernel_matrix(X, X_prime_same, sigma_X)
X_prime_diff = Z_prime + 0.3 * np.random.normal(size=n)
K_X_prime_diff = kernel_matrix(X_prime_diff, sigma=sigma_X)
K_X_X_prime_diff = kernel_matrix(X, X_prime_diff, sigma_X)

z_arguments_mcmd = np.arange(-3, 3, 0.1)
mcmd_same = np.asarray([mcmd(p, K_X, K_X_prime_same, K_X_X_prime_same, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_Z_mcmd)
                        for p in z_arguments_mcmd])
mcmd_diff = np.asarray([mcmd(p, K_X, K_X_prime_diff, K_X_X_prime_diff, Z, Z_prime, W_mcmd, W_prime_mcmd, sigma_Z_mcmd)
                        for p in z_arguments_mcmd])


plot_indices = np.sort(np.random.choice(500, 200))
plt.scatter(Z[plot_indices], X[plot_indices], label="X")
plt.scatter(Z_prime[plot_indices], X_prime_same[plot_indices], label="X'_same", marker="^")
plt.scatter(Z_prime[plot_indices], X_prime_diff[plot_indices], label="X'_diff", marker="x")
plt.legend(fontsize=13)
plt.title('Simulated Data', fontsize=20)
plt.xlabel("z", fontsize=20)
plt.ylabel("x", fontsize=20)
plt.show()
plt.plot(z_arguments_mcmd, mcmd_same, label="MCMD(X,X'_same|Z)", linewidth=5, color="orange")
plt.plot(z_arguments_mcmd, mcmd_diff, label="MCMD(X,X'_diff|Z)", linestyle="dashed", linewidth=5, color="green")
plt.legend(fontsize=13)
plt.title('MCMD values', fontsize=20)
plt.xlabel("z", fontsize=20)
plt.ylabel("MCMD", fontsize=20)
plt.show()