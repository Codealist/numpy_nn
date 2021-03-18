import numpy as np


class Activation:
    """
    Activation function
    Meant to be calculated as follows:
    f(W * X.T + b)
    Where:
    - f - non-linear activation function
    - X is m (batch,examples) by n (prev units)
    - W is m (units) by n (prev units)
    - b is intercept term (bias) - vector of n units
    """
    @staticmethod
    def activate(z):
        raise NotImplemented()

    @staticmethod
    def differentiate(dA, a):
        raise NotImplemented()


class Sigmoid(Activation):

    @staticmethod
    def activate(z):
        """
        Parameters
        ----------
        z - linear product (W @ A + b)

        Returns
        -------
        A - [batch * L-units] Activation output from the current layer
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def differentiate(dA, a):
        return dA * a * (1 - a)
