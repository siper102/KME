from ..kernels import kernel_vec, Kernel_Matrix
import numpy as np
from numpy.linalg import inv

class MCMD:

    def __init__(self, lamb = 0.01, lambp = 0.01, sigma = 0.1):
        self.sigma = sigma
        self.lamb = lamb
        self.lambp = lambp
        self.k_y = None
        self.K_y = None
        self.W_y = None
        self.K_x = None
        self.k_yp = None
        self.K_yp = None
        self.W_yp = None
        self.K_xxp = None
        self.K_xp = None


    def fit(self, x, xp, y, yp):
        n = len(x)
        m = len(xp)

        self.k_y = kernel_vec(y)
        self.K_y = Kernel_Matrix(y, y)
        self.W_y = inv(self.K_y + self.lamb * np.eye(n))
        self.K_x = Kernel_Matrix(x, x)
        self.k_yp = kernel_vec(yp)
        self.K_yp = Kernel_Matrix(yp, yp)
        self.W_yp = inv(self.K_yp + self.lambp * np.eye(m))
        self.K_xxp = Kernel_Matrix(x, xp)
        self.K_xp = Kernel_Matrix(xp, xp)

    def evaluate(self, y):
        return self.k_y(y).T@self.W_y@self.K_x@self.W_y.T@self.k_y(y) \
               - 2 * self.k_y(y).T@self.W_y @ self.K_xxp @ self.W_yp.T @self.k_yp(y) \
               + self.k_yp(y).T@self.W_yp @self.K_xp @self.W_yp.T @self.k_yp(y)

    def __call__(self, y):
        y = np.reshape(y, [-1, 1])
        return np.asarray([self.evaluate(yi) for yi in y])

