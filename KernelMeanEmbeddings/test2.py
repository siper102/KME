import numpy as np
import scipy.stats
import scipy.integrate as int
import matplotlib.pyplot as plt
from framework.kme import kernel_schetzer, witness

x = np.random.normal(size=500)
y = np.random.laplace(size = 500)

wit = witness(x, y)

yp = np.linspace(-3, 3, 100)
pl = np.zeros_like(yp)
for i in range(len(yp)):
    pl[i] = wit(yp[i])

plt.plot(yp, pl)
plt.show()