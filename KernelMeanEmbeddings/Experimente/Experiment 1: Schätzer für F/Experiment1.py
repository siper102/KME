import numpy as np
from numpy.random import multivariate_normal, normal
import matplotlib.pyplot as plt
from framework.Embedding.F_estimator import F_estimator
from framework.Embedding.KME import KME
from scipy.integrate import quad

mu = np.array([2, 1])
Sigma = np.array([[4, 3], [3, 5]])
y = 3


mu_bed = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (y - mu[1])
Sig_bed = Sigma[0, 0] - Sigma[0, 1]**2 / Sigma[1, 1]

X = multivariate_normal(mean = mu, cov=Sigma, size = 500)
Z = normal(loc = mu_bed, scale=Sig_bed, size=500)

plt.scatter(X[:,0], X[:,1])
plt.scatter(Z, np.ones_like(Z) * y)
plt.legend(["(X, Y)", "Z = (X|Y=4)"])
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Experiment_F_norm_scatter.pdf")
plt.show()


F = F_estimator()
kme = KME()

F.fit(X[:,0], X[:,1])
kme.fit(Z)
dist = np.sqrt(quad(lambda x: (F(x, 4) - kme(x))**2, -4, 10)[0])


xp = np.arange(-4, 10, 0.01)
plt.plot(xp, F(xp, 3))
plt.plot(xp, kme(xp))
plt.legend(["$\hat{F}_{P^{X|Y},n, \lambda}$", "$(\mu_{P^{Z}})_{n}$"])
plt.title("$\||\hat{F}_{P^{X|Y},n, \lambda} - (\mu_{P^{Z}})_{n}\||$ =" + f"{round(dist, 3)}")
plt.xlabel("Z")
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Experiment_F_2.pdf")
plt.show()