# Neural network: backpropagation by hand

[`numpy_spiral.py`](numpy_spiral.py) trains a three-class spiral classifier using only NumPy. It implements dense layers, ReLU, stable softmax cross-entropy, and the backward pass directly. No autograd framework computes the gradients.

For softmax cross-entropy, the logit gradient for a batch of $N$ examples is $(p-y)/N$. A dense layer $z=xW+b$ then computes $\partial L/\partial W=x^\top(\partial L/\partial z)$, $\partial L/\partial b=\sum_i\partial L/\partial z_i$, and passes $(\partial L/\partial z)W^\top$ to the previous layer.

```bash
uv run python 01-foundations/01-neural-network/numpy_spiral.py --epochs 300
```

Adapted from my [neural-network-scratch](https://github.com/Ishan8840/neural-network-scratch) repo. The PyTorch Iris example and a separate FashionMNIST MLP were omitted because they repeated the same introductory network idea.

Related note: [Loss Functions](Loss%20Functions.md) covers MSE, softmax cross-entropy, and KL divergence.
