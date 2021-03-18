import numpy as np
from typing import Type
from nn.activations import Activation, Sigmoid


class Layer:
    m, n = None, None


class Input(Layer):
    def __init__(self, n_features, batch_size):
        self.m = n_features
        self.n = batch_size
        self.cache = dict()

    def forward_step(self, x):
        self.cache["A"] = x


class HiddenLayer(Layer):
    """
    Fully connected layer
    """
    activation: Type[Activation] = None
    weights = None
    bias = None
    gradients = None

    def __init__(self, prev_layer: Layer, units: int, activation: Type[Activation], seed=1):
        self.m = units
        self.n = prev_layer.m
        self.activation = activation
        self.prev_layer = prev_layer
        self.weights = np.random.rand(self.m, self.n)
        self.bias = np.random.rand(self.m, 1)
        self.gradients = dict()
        self.cache = dict()

    def forward_step(self):
        def linear_product(a_prev, W, b):
            """
            Parameters
            ----------
            A -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)

            Returns
            -------
            Z -- linear product
            """
            return np.dot(W, a_prev) + b
        a_prev = self.prev_layer.cache["A"]
        z = linear_product(a_prev, self.weights, self.bias)
        a = self.activation.activate(z)
        self.cache["A"] = a

    def backward_step(self):
        dA = self.gradients["dA"]
        a_prev = self.prev_layer.cache["A"]
        dZ = self.activation.differentiate(dA, self.cache["A"])
        # m = batch size
        m = a_prev.shape[1]
        # don't need to store the next layer dA anymore, overwrite
        dW = 1 / m * np.dot(dZ, a_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.weights.T, dZ)
        # saving results
        try:
            self.prev_layer.gradients["dA"] = dA
        except AttributeError:
            # print("reached input layer, backpropagation finished")
            pass
        self.gradients = {"dW": dW, "db": db}


# class Output(Layer):
#     def __init__(self, prev_layer: Layer, out_units: int, loss_function):
#         self.m = prev_layer.n
#         self.n = out_units
#         self.loss_function = loss_function
#         self.prev_layer = prev_layer
#
#     def forward_step(self):
#
#
#     def backward_step(self):
#         pass
