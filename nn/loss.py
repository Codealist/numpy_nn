import numpy as np


class MSE:
    @staticmethod
    def compute(a, y):
        y = y.reshape(1, -1)
        assert(a.shape == y.shape)
        m = a.shape[1]
        return np.sum((a - y) ** 2) / m

    @staticmethod
    def differentiate(a, y):
        return 2 * (y - a)
