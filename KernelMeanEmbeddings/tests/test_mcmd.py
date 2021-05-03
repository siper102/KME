import numpy as np
from framework.conditional import MCMD
import matplotlib.pyplot as plt

Z_1 = np.random.normal(size = 500)
X = np.exp(-0.5 * Z_1**2) * np.sin(2 * Z_1) + 0.3 * np.random.normal(size = 500)

Zp_2 =  np.random.normal(size = 500)
Xp_same = np.exp(-0.5 * Zp_2**2) * np.sin(2 * Zp_2) + 0.3 * np.random.normal(size = 500)

Zp_3 = np.random.normal(size = 500)
Nx_3 = 0.3 * np.random.normal(size = 500)

Xp_diff = Zp_3 + Nx_3

plt.scatter(Z_1, X)
plt.scatter(Zp_2, Xp_same, c = "r", marker="+", )
plt.scatter(Zp_3, Xp_diff, c = "k", marker = "*")
plt.legend(["X", "$X_{same}$", "$X_{diff}$"])
plt.xlabel("Z")
plt.xlabel("X")
#plt.savefig("scatter.jpg")
plt.show()

y_plot = np.arange(-3, 3, 0.01)
mcmd_1 = MCMD(X, Xp_same, Z_1, Zp_2, lamb=0.01, lambp=0.01)
mcmd_2 = MCMD(X, Xp_diff, Z_1, Zp_3, lamb=0.01, lambp=0.01)

F_1 = np.zeros_like(y_plot)
F_2= np.zeros_like(y_plot)
for i in range(len(y_plot)):
    F_1[i] = mcmd_1(y_plot[i])
    F_2[i] = mcmd_2(y_plot[i])


plt.plot(y_plot, F_1)
plt.plot(y_plot, F_2)
plt.legend(["$\widehat{MCMD}^{2}(P^{X|Z},P^{X_{same}|Z}, H_{\mathcal{X}})$",
            "$\widehat{MCMD}^{2}(P^{X|Z}, P^{X_{diff}|Z}, H_{\mathcal{X}})$"], loc = "upper center")
plt.show()