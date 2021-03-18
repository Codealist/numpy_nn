class MLP:
    loss_function = None
    layers = None

    def __init__(self, layers, loss_function, learning_rate):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def train(self, x, y, epochs=10):
        for e in range(1, epochs+1):
            self.feed_forward(x)
            self.backprop(y)
            self._update_parameters()
            print(f"Train score after epoch {e} is: {self.score(self.predict(x), y)}")

    def feed_forward(self, x):
        input_ = self.layers[0]
        input_.forward_step(x)
        predictions = None
        for l in self.layers[1:]:
            l.forward_step()
            predictions = l.cache["A"]
        return predictions

    def backprop(self, y):
        output_ = self.layers[-1]
        dAL = self.loss_function.differentiate(output_.cache["A"], y)
        output_.gradients["dA"] = dAL
        for l in reversed(self.layers[1:]):
            l.backward_step()

    def predict(self, x):
        return self.feed_forward(x)

    def _update_parameters(self):
        for l in self.layers[1:]:
            l.weights = l.weights - self.learning_rate * l.gradients["dW"]
            l.bias = l.bias - self.learning_rate * l.gradients["db"]

    def score(self, predictions, y):
        return self.loss_function.compute(predictions, y)

