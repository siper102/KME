import numpy as np
from framework.conditional import HSCIC
import matplotlib.pyplot as plt


Z  = np.random.normal(size=500)
X = np.exp(-0.5*Z**2) * np.sin(2*Z) + np.random.normal(size=500) * 0.3
Y = np.random.normal(size = 500) * 0.3

Z_2  = np.random.normal(size=500)
X_2 = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3
Y_dep = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3 + 0.2*X

Z_3  = np.random.normal(size=500)
X_3 = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3
Yp_dep = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3 + 0.4*X_3

hsic = HSCIC(X, Y, Z)
hsic_2 = HSCIC(X_2, Y_dep, Z_2)
hsic_3 = HSCIC(X_3, Yp_dep, Z_3)

xp = np.linspace(-2, 2, 1000)
y = np.zeros_like(xp)
y_2 = np.zeros_like(xp)
y_3 = np.zeros_like(xp)
for i in range(len(xp)):
    y[i] = hsic(xp[i])
    y_2[i] = hsic_2(xp[i])
    y_3[i] = hsic_3(xp[i])

plt.plot(xp, y)
plt.plot(xp, y_2)
plt.plot(xp, y_3)
plt.figsave("HSCIC.pdf")
plt.show()

plt.savefig("HSCIC.pdf")