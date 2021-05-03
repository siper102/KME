from ..kernels import kernel_vec, Kernel_Matrix
import numpy as np
from numpy.linalg import inv

class HSCIC:

    def __init__(self, lamb = 0.01, sigma = 0.1):
        self.lamb = lamb
        self.sigma = sigma
        self.k_z = None
        self.K_z = None
        self.K_x = None
        self.K_y = None
        self.W = None


    def fit(self, X, Y, Z):
        n = len(X)

        self.k_z = kernel_vec(Z)
        self.K_z = Kernel_Matrix(Z, Z)
        self.K_x = Kernel_Matrix(X, X)
        self.K_y = Kernel_Matrix(Y, Y)
        self.W = inv(self.K_z + self.lamb * np.eye(n))

    def evaluate(self, z):
        return self.k_z(z).T@self.W@ (self.K_x * self.K_y) @ self.W.T@ self.k_z(z) \
               - 2*(self.k_z(z).T@self.W@((self.K_x@self.W.T@self.k_z(z)) * (self.K_y@self.W.T@self.k_z(z)))) \
               + (self.k_z(z).T@self.W@self.K_x@self.W.T@self.k_z(z)) * (self.k_z(z).T@self.W@self.K_y@self.W.T@self.k_z(z))

    def __call__(self, y):
        y = np.reshape(y, [-1, 1])
        return np.asarray([self.evaluate(yi) for yi in y])