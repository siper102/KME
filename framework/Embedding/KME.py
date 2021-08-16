from .Embedd import Embedd
from ..kernels import kernel_vec
import numpy as np

class KME(Embedd):

    def __init__(self, sigma = 0.1):
        self.sigma = sigma
        self.k_x = None
        self.n = None

    def fit(self, x):
        self.n = len(x)
        self.k_x = kernel_vec(x, self.sigma)

    def evaluate(self, y):
        return np.array([[np.mean(self.k_x(y))]])
