import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.F_estimator import F_estimator
from framework.Embedding.KME import KME

mu = np.array([2, 1])
Sigma = np.array([[4, 3], [3, 5]])
y = 3

mu_bed = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (y - mu[1])
Sig_bed = Sigma[0, 0] - Sigma[0, 1] / Sigma[1, 1] * Sigma[1, 0]

X_1 = np.random.multivariate_normal(mean = mu, cov=Sigma, size = 500)
X_2 = np.random.normal(loc = mu_bed, scale=Sig_bed, size=500)

plt.scatter(X_1[:,0], X_1[:,1])
plt.scatter(X_2, np.ones_like(X_2) * y)
plt.show()


F = F_estimator()
kme = KME()

F.fit(X_1[:,0], X_1[:,1])
kme.fit(X_2)

xp = np.arange(-4, 10, 0.01)
plt.plot(xp, F(xp, 3))
plt.plot(xp, kme(xp))
plt.legend(["F", "KME"])

plt.show()