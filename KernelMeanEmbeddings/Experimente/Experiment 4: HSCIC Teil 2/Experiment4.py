import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.HSCIC import HSCIC



Z = np.random.normal(size=500)
X = np.exp(-1/2 * Z**2) * np.sin(2*Z)  * np.random.normal(size=500)
Y_ind = np.exp(-1/2 * Z**2) * np.sin(2*Z) * np.random.normal(size=500)
Y_dep = np.exp(-1/2 * Z**2) * np.sin(2*Z)  * np.random.normal(size=500) + 0.2 * X
Yp_dep = np.exp(-1/2 * Z**2) * np.sin(2*Z)  * np.random.normal(size=500) + 0.4 * X

plt.scatter(Z, X)
plt.scatter(Z, Y_ind)
plt.scatter(Z, Y_dep)
plt.scatter(Z, Yp_dep)
plt.legend(["X", "$Y_{ind}$", "$Y_{dep}$", "$Y^{\prime}_{dep}$"])
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Scatter_HSCIC_mult.pdf")
plt.show()

hscic_1 = HSCIC()
hscic_2 = HSCIC()
hscic_3 = HSCIC()

hscic_1.fit(X, Y_ind, Z)
hscic_2.fit(X, Y_dep, Z)
hscic_3.fit(X, Yp_dep, Z)

xp = np.linspace(-2, 2, 200)
plt.plot(xp, hscic_1(xp))
plt.plot(xp, hscic_2(xp))
plt.plot(xp, hscic_3(xp))
plt.legend(["HSCIC(X, $Y_{ind}$, Z)", "$HSCIC(X, Y_{dep}, Z)$", "$HSCIC(X, Y^{\prime}_{dep}, Z)$"])

plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Experiment_HSCIC_mult.pdf")
plt.show()

