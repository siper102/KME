import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.MCMD import MCMD

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
plt.show()

mcmd_1 = MCMD()
mcmd_2 = MCMD()

mcmd_1.fit(X, Xp_same, Z_1, Zp_2)
mcmd_2.fit(X, Xp_diff, Z_1, Zp_3)

xp = np.linspace(-2, 2, 200)
plt.plot(xp, mcmd_1(xp))
plt.plot(xp, mcmd_2(xp))
plt.legend(["$\widehat{MCMD}^{2}(P^{X|Z},P^{X_{same}|Z}, H_{\mathcal{X}})$",
            "$\widehat{MCMD}^{2}(P^{X|Z}, P^{X_{diff}|Z}, H_{\mathcal{X}})$"], loc = "upper center")
plt.show()