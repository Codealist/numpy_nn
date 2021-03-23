import numpy as np
from typing import Type
from nn.activations import Sigmoid, Activation as ActFunction
from scipy.sparse import csr_matrix


class Layer:
    m, n = None, None


class Input(Layer):
    def __init__(self, n_features, batch_size):
        self.m = n_features
        self.n = batch_size
        self.cache = dict()

    def forward_step(self, x):
        self.cache["A"] = x


class FullyConnected(Layer):
    """
    Fully connected layer
    """
    activation: Type[ActFunction] = None
    weights = None
    bias = None
    gradients = None

    def __init__(self, prev_layer: Layer, units: int, activation: Type[ActFunction], seed=1):
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


class Activation(Layer):
    pass


class Convolution(Layer):

    def __init__(self, kernels, padding, strides):
        """
        Initializes the layer with convolution
        Output is a 3D tensor of shape:
        (W−K+2P)/S+1 by (W−K+2P)/S+1 by number of kernels

        Parameters
        ----------
        kernels:
            Collection of kernels to train. [(2,2), (3,3), etc]
            Note that it has to be aligned with padding and strides,
            so that resulting feature map is consistent in size
            For example, given input rows=cols=32, two kernels [(2,2), (4,4)]
            there should be paddings [1, 0] and same strides, so the output
            shape is 16x16x2
        padding - padding numbers (no "valid" and "same" for now)
        strides - steps over the matrix (only square for now)
        """
        self.kernels = kernels
        self.padding = padding
        self.strides = strides
        self.cache = dict()
        self.gradients = dict()

    @staticmethod
    def feature_map_size(w, k, p, s):
        return 1 + (w - k + 2 * p) / s

    def sparse_kernel(self, kernel, w, p, s):
        k = kernel.shape[0]
        fm_size = self.feature_map_size(w, k, p, s)
        out_size = fm_size ** 2
        vsteps = hsteps = fm_size

        def right_zero_pad(matrix, size):
            return np.concatenate((kernel, np.zeros((matrix.shape[0], size))), axis=1)
        kernel = right_zero_pad(kernel, w - k).flatten()
        kernel = right_zero_pad(kernel, w**2 - kernel.shape[0]).reshape(1, -1)
        for v in range(vsteps):
            # go to new line
            _s = s + w
            for h in range(hsteps):
                kernel = np.stack((kernel, np.roll(kernel[-1], _s)), axis=0)
                # back to default stride
                _s = s

        # row = np.array(range(out_size)).repeat(k)
        # gap = w - k
        # for i in range(k):
        #
        #
        # col = list(range(k))
        #
        # col += list(range(col[-1] + gap, col[-1] + gap + k))
        # kernel = kernel.flatten()
        # data = np.tile(kernel, out_size)
        # kernel = csr_matrix((data, (row, col)), shape=(out_size, 3)).toarray()
        return kernel

    def convolve(self):
        pass

    def forward_step(self):
        pass

    def backward_step(self):
        pass


class Pooling(Layer):

    def __init__(self, window, stride):
        pass
