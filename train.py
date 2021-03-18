import numpy as np
from nn.layers import Input, HiddenLayer
from nn.activations import Sigmoid
from nn.models import MLP
from nn.loss import MSE

features = 7
batch_size = 512

layers_ = list()
layers_.append(inp := Input(features, batch_size))
layers_.append(hl1 := HiddenLayer(inp, 10, Sigmoid))
layers_.append(hl2 := HiddenLayer(hl1, 16, Sigmoid))
layers_.append(hl3 := HiddenLayer(hl2, 24, Sigmoid))
layers_.append(hl4 := HiddenLayer(hl3, 16, Sigmoid))
layers_.append(hl5 := HiddenLayer(hl4, 8, Sigmoid))
layers_.append(out := HiddenLayer(hl5, 1, Sigmoid))


x = np.random.rand(batch_size, features)
y = np.random.rand(batch_size)

mlp = MLP(layers_, MSE, 0.08)
mlp.train(x.T, y, epochs=100)
