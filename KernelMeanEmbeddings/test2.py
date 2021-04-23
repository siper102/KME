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


gauss_rand = np.random.normal(size = 20000)
#lapl_rand = np.random.laplace(size = 20000)

kme_gauss = kme(gauss_rand)
#kme_lapk = kme(lapl_rand)


x = np.linspace(-4, 4, 500)
plt.plot(x, cont()(x))
plt.show()