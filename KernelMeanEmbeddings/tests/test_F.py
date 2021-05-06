from framework.Embedding.F_estimator import F_estimator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

n = 500
np.random.seed(22)

Z = np.random.normal(size=n)
Z_prime = np.random.normal(size=n)

X = np.exp(-1/2*Z**2) * np.sin(2*Z) + 0.3 * np.random.normal(size=n)
X_prime_same = np.exp(-1/2*Z_prime**2) * np.sin(2*Z_prime) + 0.3 * np.random.normal(size=n)

f_1 = F_estimator()
f_2 = F_estimator()
f_1.fit(X, Z)
f_2.fit(X_prime_same, Z_prime)

x_arguments = np.arange(-3, 3, 0.1)
z_arguments_mcmd = np.arange(-3, 3, 0.1)


wit_same = f_1(x_arguments, z_arguments_mcmd) - f_2(x_arguments, z_arguments_mcmd)


im_same = plt.imshow(wit_same, cmap='jet', interpolation='nearest',
                    extent=[z_arguments_mcmd[0], z_arguments_mcmd[-1], x_arguments[0], x_arguments[-1]],)

plt.colorbar()

plt.show()