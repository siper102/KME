from ..kernels import kernel_vec, Kernel_Matrix
from numpy.linalg import inv
import numpy as np

class F_estimator:

    def __init__(self, sigma = 0.1, lamb = 0.01):
        self.sigma = sigma
        self.lamb = lamb
        self.k_x = None
        self.k_y = None
        self.K_y = None
        self.W = None

    def fit(self, x, y):
        n = len(x)
        self.k_x = kernel_vec(x, self.sigma)
        self.l_y = kernel_vec(y, self.sigma)
        self.K_y = Kernel_Matrix(y, y)
        self.W = inv(self.K_y + self.lamb * np.eye(n))


    def evaluate(self, y, x):
        return (self.l_y(y).T @ self.W @ self.k_x(x))[0, 0]

    def __call__(self, x, y):
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])
        return np.asarray([[self.evaluate(q, p) for q in y] for p in x])