# Convolutional neural network

[`cnn.py`](cnn.py) is a compact **LeNet-style** MNIST classifier: two convolution-and-pooling stages followed by fully connected layers. It uses ReLU and average pooling, so it is an educational variant rather than an exact LeNet-5 reproduction. PyTorch provides the convolution operator; the model architecture and training loop are written here.

| File |
| --- |
| [cnn.py](cnn.py) |

Run from the repository root:

```bash
uv run python 01-foundations/02-cnn/cnn.py --smoke-test --epochs 1
uv run python 01-foundations/02-cnn/cnn.py --download --epochs 3
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/01-foundations/03-cnn).
