"""NumPy-only MLP with manual backpropagation on a three-class spiral."""

import argparse

import numpy as np


def spiral_data(samples_per_class=100, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((samples_per_class * classes, 2))
    y = np.zeros(samples_per_class * classes, dtype=np.int64)
    for cls in range(classes):
        rows = slice(cls * samples_per_class, (cls + 1) * samples_per_class)
        radius = np.linspace(0, 1, samples_per_class)
        theta = np.linspace(cls * 4, (cls + 1) * 4, samples_per_class)
        theta += rng.normal(0, 0.2, samples_per_class)
        x[rows] = np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))
        y[rows] = cls
    return x, y


class Dense:
    def __init__(self, inputs, outputs, rng):
        self.weights = rng.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
        self.bias = np.zeros(outputs)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, gradient):
        self.dweights = self.x.T @ gradient
        self.dbias = gradient.sum(axis=0)
        return gradient @ self.weights.T

    def step(self, rate):
        self.weights -= rate * self.dweights
        self.bias -= rate * self.dbias


class ReLU:
    def forward(self, x):
        self.active = x > 0
        return np.maximum(x, 0)

    def backward(self, gradient):
        return gradient * self.active


def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
    gradient = probabilities.copy()
    gradient[np.arange(len(labels)), labels] -= 1
    gradient /= len(labels)
    return loss, gradient, probabilities


def train(epochs=1000, seed=0):
    x, y = spiral_data(seed=seed)
    rng = np.random.default_rng(seed)
    layers = [Dense(2, 64, rng), ReLU(), Dense(64, 64, rng), ReLU(), Dense(64, 3, rng)]
    for epoch in range(epochs):
        output = x
        for layer in layers:
            output = layer.forward(output)
        loss, gradient, probabilities = softmax_cross_entropy(output, y)
        for layer in reversed(layers):
            gradient = layer.backward(gradient)
        rate = 1 / (1 + 0.001 * epoch)
        for layer in layers:
            if isinstance(layer, Dense):
                layer.step(rate)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            accuracy = np.mean(probabilities.argmax(axis=1) == y)
            print(f"epoch {epoch:4d} | loss {loss:.4f} | accuracy {accuracy:.3f}")
    return layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.epochs, args.seed)
