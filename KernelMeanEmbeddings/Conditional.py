import numpy as np
import scipy.stats
import scipy.integrate as int
import matplotlib.pyplot as plt

sig = 0.3




def gauss_kernel(x, y):
    return np.exp(1 / (2*sig**2) * - np.linalg.norm(x - y)**2)

def integrate(y):
    gauss_fun = lambda x, y: scipy.stats.laplace.pdf(x) * gauss_kernel(x, y)
    ret = np.zeros_like(y)
    for i in range(len(y)):
        func = lambda x: gauss_fun(x, y[i])
        ret[i] = int.quad(func, -np.inf, np.inf)[0]
    return ret

def cont():
    return lambda y: integrate(y)

def k(x, y):
    summe = 0
    for xi in x:
        summe += gauss_kernel(xi, y)
    return 1 / len(x) * summe

def kme_fun(x, y):
    ret = np.zeros_like(y)
    for i in range(len(y)):
        ret[i] = k(x, y[i])
    return ret

def kme(x):
    return lambda y: kme_fun(x, y)



mu_x = 1
mu_y = 3
sigma_x = 2
sigma_y = 3
p = 0.5
mu = np.array([mu_x, mu_y])
cov = np.array([[sigma_x**2, p * sigma_x * sigma_y],
                [p * sigma_x * sigma_y, sigma_y**2]])

y = 2

mu_x_b = mu_x + p * (sigma_x / sigma_y) * (y - mu_y)
sigma_x_b = sigma_x**2 * (1 - p**2)

d = np.random.multivariate_normal(mu, cov, 20000)
plt.scatter(d[:,0], d[:,1])

d2 = np.random.normal(mu_x_b, sigma_x_b, 200)
plt.scatter(d2, np.ones_like(d2) * 2)
plt.show()
