import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.MCMD import MCMD

np.random.seed(22)
Y_1 = np.random.normal(size = 500)
eps_1 =  0.3 * np.random.normal(size = 500)
X = np.exp(-0.5 * Y_1**2) * np.sin(2 * Y_1) + eps_1

Yp_2 =  np.random.normal(size = 500)
eps_2 = + 0.3 * np.random.normal(size = 500)
Xp_same = np.exp(-0.5 * Yp_2 ** 2) * np.sin(2 * Yp_2) * eps_2

Yp_3 = np.random.normal(size = 500)
eps_3 = + 0.3 * np.random.normal(size = 500)
Xp_diff = Yp_3 + eps_3

plt.scatter(Y_1, X)
plt.scatter(Yp_2, Xp_same, c ="r", marker="+", )
plt.scatter(Yp_3, Xp_diff, c ="k", marker ="*")
plt.legend(["X", "$X_{same}$", "$X_{diff}$"])
plt.xlabel("Y")
plt.xlabel("X")
plt.savefig("Scatter_MCMD.pdf")
plt.show()

mcmd_1 = MCMD()
mcmd_2 = MCMD()

mcmd_1.fit(X, Xp_same, Y_1, Yp_2)
mcmd_2.fit(X, Xp_diff, Y_1, Yp_3)

xp = np.linspace(-3, 3, 200)
plt.plot(xp, mcmd_1(xp))
plt.plot(xp, mcmd_2(xp))

plt.legend(["$\widehat{MCMD}^{2}(P^{X|Y},P^{X_{same}|Y}, H_{\mathcal{X}})$",
            "$\widehat{MCMD}^{2}(P^{X|Y}, P^{X_{diff}|Y}, H_{\mathcal{X}})$"], loc = "upper center")
plt.savefig("Experiment_MCMD.pdf")
plt.show()