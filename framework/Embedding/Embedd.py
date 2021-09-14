import numpy as np

# So sollen die implementierungen der Schätzer aufgebaut sein
class Embedd:
    def __init__(self):     # Hyperparameter werden an das Objekt gegeben
        pass

    def fit(self, *args, **kwargs):     # Die Parameter sollen mit den Daten definiert werden.
        pass

    def evaluate(self, element):         # Auswertung an einem bestimmten Punkt
        pass

    def __call__(self, elements):        # Auswertung für jeden Punkt in einer Liste
        y = np.reshape(elements, [-1, 1])
        return np.asarray([self.evaluate(yi)[0, 0] for yi in y])