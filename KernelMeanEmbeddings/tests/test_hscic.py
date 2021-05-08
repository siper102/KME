import numpy as np
from framework.Embedding.HSCIC import HSCIC
import matplotlib.pyplot as plt

Z  = np.random.normal(size=500)
X = np.exp(-0.5*Z**2) * np.sin(2*Z) + np.random.normal(size=500) * 0.3
Y = np.random.normal(size = 500) * 0.3

Z_2  = np.random.normal(size=500)
X_2 = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3
Y_dep = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3 + 0.2*X_2

Z_3  = np.random.normal(size=500)
X_3 = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3
Yp_dep = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3 + 0.4*X_3

hscic = HSCIC()
hscic_2 = HSCIC()
hscic_3 = HSCIC()

hscic.fit(X, Y, Z)
hscic_2.fit(X_2, Y_dep, Z_2)
hscic_3.fit(X_3, Yp_dep, Z_3)

xp = np.linspace(-2, 2, 200)
plt.plot(xp, hscic(xp))
plt.plot(xp, hscic_2(xp))
plt.plot(xp, hscic_3(xp))
plt.legend(["HSCIC(X, Y, Z, )", "HSCIC(X, Y_dep, Z)", "HSCIC(X, Yp_dep, Z)"])
plt.show()
