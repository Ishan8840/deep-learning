# Convolutional neural network

A CNN learns spatial features with three ideas: local receptive fields, shared filters, and pooling. A filter applies the same weights at many image locations, producing a feature map. Stacking filters lets later layers combine simple patterns into more useful representations.

[`cnn.py`](cnn.py) is a compact **LeNet-style** MNIST classifier: two convolution-and-pooling stages followed by fully connected layers. It uses ReLU and average pooling, so it is an educational variant rather than an exact LeNet-5 reproduction. PyTorch provides the convolution operator; the model architecture and training loop are written here.

From the repository root:

```bash
uv run python 01-foundations/02-cnn/cnn.py --smoke-test --epochs 1
uv run python 01-foundations/02-cnn/cnn.py --download --epochs 3
```

The first command uses random images and labels to verify the pipeline. The second trains on MNIST and evaluates on its test split.

For historical context, the path from [LeNet-5](https://hal.science/hal-03926082/document) to [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) shows how convolution, larger data and compute, and residual connections made deeper image models practical.

My fuller [CNN study notes](CNN.md) include the LeNet layer breakdown and the later architecture milestones.
