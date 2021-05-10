import numpy as np
from framework.Embedding.HSCIC import HSCIC
import matplotlib.pyplot as plt

Z  = np.random.normal(size=500)
X = np.exp(-0.5*Z**2) * np.sin(2*Z) + np.random.normal(size=500) * 0.3
Y_noise = np.random.normal(size = 500) * 0.3

Z_2  = np.random.normal(size=500)
X_2 = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3
Y_dep = np.exp(-0.5*Z_2**2) * np.sin(2*Z_2) + np.random.normal(size=500) * 0.3 + 0.2*X_2

Z_3  = np.random.normal(size=500)
X_3 = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3
Yp_dep = np.exp(-0.5*Z_3**2) * np.sin(2*Z_3) + np.random.normal(size=500) * 0.3 + 0.4*X_3

plt.scatter(Z, X)
plt.scatter(Z, Y_noise)
plt.scatter(Z_2, Y_dep)
plt.scatter(Z_3, Yp_dep)
plt.legend(["X", "$Y_{noise}$", "$Y^{\prime}_{dep}$", "$Y_{dep}$"])
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Scatter_HSCIC_addi.pdf")
plt.show()


hscic = HSCIC()
hscic_2 = HSCIC()
hscic_3 = HSCIC()

hscic.fit(X, Y_noise, Z)
hscic_2.fit(X_2, Y_dep, Z_2)
hscic_3.fit(X_3, Yp_dep, Z_3)

xp = np.linspace(-2, 2, 200)
plt.plot(xp, hscic(xp))
plt.plot(xp, hscic_2(xp))
plt.plot(xp, hscic_3(xp))
plt.legend(["HSCIC(X, $Y_{noise}$, Z)", "HSCIC(X, $Y_{dep}$, Z)", "$HSCIC(X, Y^{\prime}_{dep}, Z)$"])
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Experiment_HSCIC_addi.pdf")
plt.show()
