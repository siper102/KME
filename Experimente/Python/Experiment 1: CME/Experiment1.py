import numpy as np
from numpy.random import multivariate_normal, normal
import matplotlib.pyplot as plt
from framework.Embedding.CME import CME
from framework.Embedding.KME import KME
from scipy.integrate import quad

# Die wichtigsten Werte für dieses Experiment
mu = np.array([2, 1])
Sigma = np.array([[4, 3], [3, 5]])
y = 3

# Mittelwert und Varianz der bedingten Zufalsvariable
mu_bed = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (y - mu[1])
Sig_bed = Sigma[0, 0] - Sigma[0, 1]**2 / Sigma[1, 1]
# Erzeugen der Daten
X = multivariate_normal(mean = mu, cov=Sigma, size = 500)
Z = normal(loc = mu_bed, scale=Sig_bed, size=500)
# Visualisieren der Daten
plt.scatter(X[:,0], X[:,1])
plt.scatter(Z, np.ones_like(Z) * y)
plt.legend(["(X, Y)", "Z = (X|Y=3)"])
plt.show()

# Initialisierung der Schätzer mit den default-Werten
cme = CME()
kme = KME()
# Anpassen der Schätzer
cme.fit(X[:, 0], X[:, 1])
kme.fit(Z)
# L^{2} Abstand der Schätzer
dist = np.sqrt(quad(lambda x: (cme(x, 3) - kme(x)) ** 2, -4, 10)[0])

# Visualisieren der resultierenden Funktionen
xp = np.arange(-4, 10, 0.01)
plt.plot(xp, cme(xp, 3))
plt.plot(xp, kme(xp))
plt.legend(["$\hat{F}_{P^{X|Y},n, \lambda}$", "$\hat{\mu}_{P^{Z}}$"], fontsize = 15)
plt.title("$\||\hat{F}_{X|Y,500, 0.01}(3) - \hat{\mu}_{P^{Z}}\||$ =" + f"{round(dist, 3)}",fontsize = 20)
plt.xlabel("x")
plt.show()