import numpy as np

class Embedd:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, element):
        pass

    def __call__(self, elements):
        y = np.reshape(elements, [-1, 1])
        return np.asarray([self.evaluate(yi)[0, 0] for yi in y])