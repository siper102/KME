import numpy as np
import scipy.stats
import scipy.integrate as int
import matplotlib.pyplot as plt
from framework.kme import kernel_schetzer

x = np.random.exponential(size = 100)

kme = kernel_schetzer(x=x)

y = np.linspace(0, 3, 100)
pl = np.zeros_like(y)
for i in range(len(y)):
    pl[i] = kme(y[i])

plt.plot(y, pl)
plt.plot(y, scipy.stats.expon.pdf(y))
plt.show()