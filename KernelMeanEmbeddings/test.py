import numpy as np
from framework.conditional import MCMD
import matplotlib.pyplot as plt

Z_1 = np.random.normal(size = 500)
Nx_1 = 0.3 * np.random.normal(size = 500)

X = np.exp(-0.5 * Z_1**2) * np.sin(2 * Z_1) + Nx_1

Zp_2 =  np.random.normal(size = 500)
Nx_2 = 0.3 * np.random.normal(size = 500)

Xp_same = np.exp(-0.5 * Zp_2**2) * np.sin(2 * Zp_2) + Nx_2

Zp_3 = np.random.normal(size = 500)
Nx_3 = 0.3 * np.random.normal(size = 500)

Xp_diff = Zp_3 + Nx_3

plt.scatter(Z_1, X)
plt.scatter(Zp_2, Xp_same)
plt.scatter(Zp_3, Xp_diff)
plt.show()

mcmd_1 = MCMD(X, Xp_same, Z_1, Zp_2, kernel_x_args= dict(sigma = 0.1), kernel_y_args=dict(sigma = 0.1), lamb=0.01, lambp=0.01)
mcmd_2 = MCMD(X, Xp_diff, Z_1, Zp_3, kernel_x_args= dict(sigma = 0.1), kernel_y_args=dict(sigma = 0.1), lamb=0.01, lambp=0.01)


xp = np.linspace(-3, 3, 10000)
ysame = np.zeros_like(xp)
ydiff = np.zeros_like(xp)
for i in range(len(xp)):
    ysame[i] = mcmd_1(xp[i])
    ydiff[i] = mcmd_2(xp[i])

plt.plot(xp, ysame)
plt.plot(xp, ydiff)
plt.ylim((0, 1))
plt.show()
