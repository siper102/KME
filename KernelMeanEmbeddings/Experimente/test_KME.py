from framework.Embedding.KME import KME
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

size = 2*10**4
x = np.random.normal(size = size)
y = np.random.laplace(size = size)

sigma = 0.5**-2
kme = KME(sigma = sigma)
kme.fit(x)

kme_2 = KME(sigma = sigma)
kme_2.fit(y)

xp = np.linspace(-6, 6, 100)
plt.plot(xp, (kme(xp) - kme_2(xp)) * 5)
plt.plot(xp, st.norm.pdf(xp))
plt.plot(xp, st.laplace.pdf(xp))
plt.ylim((-0.2, 0.8))
plt.show()