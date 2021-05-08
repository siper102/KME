import numpy as np
import matplotlib.pyplot as plt
from framework.Embedding.HSCIC import HSCIC

Z_1 = np.random.normal(size = 500)
X_1 = np.exp(-1/2 * Z_1 ** 2) * np.sin(2 * Z_1) * (0.3 * np.random.normal(size = 500))
Y_ind = X_1

Z_2 = np.random.normal(size = 500)
X_2 = np.exp(-1/2 * Z_2 ** 2) * np.sin(2 * Z_2) * (0.3 * np.random.normal(size = 500))
Y_dep = np.exp(-1/2 * Z_2 ** 2) * np.sin(2 * Z_2) * (0.3 * np.random.normal(size = 500)) + 0.2*X_2

Z_3 = np.random.normal(size = 500)
X_3 = np.exp(-1/2 * Z_3 ** 2) * np.sin(2 * Z_3) * (0.3 * np.random.normal(size = 500))
Yp_dep = np.exp(-1/2 * Z_3 ** 2) * np.sin(2 * Z_3) * (0.3 * np.random.normal(size = 500)) + 0.2*X_3

plt.scatter(Z_1, X_1)
plt.scatter(Z_1, Y_ind)
plt.scatter(Z_2, Y_dep)
plt.scatter(Z_3, Yp_dep)
plt.show()

hscic_1 = HSCIC()
hscic_2 = HSCIC()
hscic_3 = HSCIC()

hscic_1.fit(X_1, Y_ind, Z_1)
hscic_2.fit(X_2, Y_dep, Z_2)
hscic_3.fit(X_3, Yp_dep, Z_3)

xp = np.linspace(-2, 2, 200)
plt.plot(xp, hscic_1(xp))
plt.plot(xp, hscic_2(xp))
plt.plot(xp, hscic_3(xp))
plt.show()

