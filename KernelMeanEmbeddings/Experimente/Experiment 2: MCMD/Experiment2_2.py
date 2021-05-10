from framework.Embedding.F_estimator import F_estimator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

n = 500
np.random.seed(22)

Y_1 = np.random.normal(size=n)
Yp_2 = np.random.normal(size=n)

X = np.exp(-1 / 2 * Y_1 ** 2) * np.sin(2 * Y_1) + 0.3 * np.random.normal(size=n)
Xp_same = np.exp(-1 / 2 * Yp_2 ** 2) * np.sin(2 * Yp_2) + 0.3 * np.random.normal(size=n)

f_1 = F_estimator()
f_2 = F_estimator()
f_1.fit(X, Y_1)
f_2.fit(Xp_same, Yp_2)

xp = np.arange(-3, 3, 0.1)
yp = np.arange(-3, 3, 0.1)

wit_same = f_1(xp, yp) - f_2(xp, yp)

im_same = plt.imshow(wit_same, cmap='viridis', interpolation='nearest',
                     extent=[yp[0], yp[-1], xp[0], xp[-1]])

plt.colorbar()
plt.savefig("/Users/simonperschel/Dropbox/Bachelorarbeit/Arbeit/6.Experimente/Images/Experiment_F_same.pdf")
plt.show()