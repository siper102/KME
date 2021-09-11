from scipy.stats.distributions import norm, laplace
import numpy as np
from framework.Embedding.KME import KME
import matplotlib.pyplot as plt

x = np.random.normal(size = 100)
y = np.random.laplace(size = 100)
sig = 1/(0.7)**2

kme_g = KME(sig)
kme_l = KME(sig)
kme_m = KME(sig)

kme_g.fit(x)
kme_l.fit(y)
kme_m.fit(abs(x-y))

xp = np.linspace(-4, 4, 1000)

plt.plot(xp, kme_g(xp) - kme_l(xp))
plt.plot(xp, norm.pdf(xp))
plt.plot(xp, laplace.pdf(xp))
plt.show()
