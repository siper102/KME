import numpy as np

class Embedd:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, element):
        pass

    def __call__(self, elements):
        y = np.reshape(y, [-1, 1])
        return np.asarray([self.evaluate(yi) for yi in y])