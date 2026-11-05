# Convolutional neural network

A CNN learns spatial features with three ideas: local receptive fields, shared filters, and pooling. A filter applies the same weights at many image locations, producing a feature map. Stacking filters lets later layers combine simple patterns into more useful representations.

[`cnn.py`](cnn.py) is a compact **LeNet-style** MNIST classifier: two convolution-and-pooling stages followed by fully connected layers. It uses ReLU and average pooling, so it is an educational variant rather than an exact LeNet-5 reproduction. PyTorch provides the convolution operator; the model architecture and training loop are written here.

From the repository root:

```bash
uv run python 01-foundations/03-cnn/cnn.py --smoke-test --epochs 1
uv run python 01-foundations/03-cnn/cnn.py --download --epochs 3
```

The first command uses random images and labels to verify the pipeline. The second trains on MNIST and evaluates on its test split.

For historical context, the path from [LeNet-5](https://hal.science/hal-03926082/document) to [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) shows how convolution, larger data and compute, and residual connections made deeper image models practical.

The study notes below include the LeNet layer breakdown and later architecture milestones.

## Study notes

Before CNNs, image recognition systems were often split into two parts: a hand-designed feature extractor and a classifier. The feature extractor converted the raw image into features that the classifier could use. These systems often required the input to be carefully normalized, centered, or otherwise preprocessed to handle translations and distortions.

### CNNs

There were three main additions that CNNs contributed; local receptive fields, shared weights, and spatial or temporal subsampling.

**local receptive fields**
- extract elementary visual features such as edges, endpoints, corners, etc...
- combine these extracted elementary features with subsequent layers to extract higher order features

**shared weights**
- force a set of units, whose receptive fields are located in different areas to share the same weights
- elementary feature detectors that are useful in one part of an image are often useful in other areas too
- these are called filters/kernels and perform the same operation on on different parts of the image which produce feature maps
- a complete convolutional layer is composed of several differnt feature maps to extract various features

**spatial or temporal subsampling**
- position of features is not really important and can in fact be harmful
- 2x2 area that computes the average of the 4 inputs, multiplies it by a trainable coefficient, and adds a bias, passes the result to a sigmoid function
- makes network less dependent on position variance

### LeNet-5

The architecture consists of 7 layers. The input is a 32x32 image, values of input pixels are normalized so that the white corresponds to -0.1 and black corresponds to 1.175. LeNet showed that the same network could learn the feature extractor and classifier from raw pixels. Convolution allowed the network to learn features and exploit the spatial structure of images, shared weights, reduced the number of paramters, and subsampling made the representation less sensitive to small changes in feature positions.

**layer C1**
- conv layer with 6 feaure maps
- each feature map is connected to a 5x5 neighbohood in the input
- $(25 + 1) \times 6 = 156$ learnable params

**layer S2**
- subsampling layer with six feature maps of 14x14
- each unit in the feature map is connected to a 2x2 neighborhood in the corresponding feature map in C1
- the 2x2 receptive fields are nonoverlapping
- $6 \times 2 = 12$ learnable params

**layer C3**
- conv layer with 16 feature maps
- each one is connected to several 5x5 neighborhoods at identical locations of a subset of S2's feature maps
- this forces a break of symmetry in the network
- 6 C3 maps connect to 3 S2 maps
- 9 C3 maps connect to 4 S2 maps
- 1 C3 map connects to 6 S2 maps
- $(5 \times 5 \times 3 + 1) \times 6 = 456$
-  $(5 \times 5 \times 4 + 1) \times 9 = 909$
-  $(5 \times 5 \times 6 + 1) = 151$
- $456 + 909 + 151 = 1516$ learnable params

**layer S4**
- subsampling layer with 16 feature maps of size 5x5
- each feature map is connected to a 2x2 neighborhood in the corresponding feature map in C3
- $16 \times 2 = 32$ learnable params

**layer C5**
- conv layer with 120 feature maps
- each unit is connected to a 5x5 neighborhood on all 16 of S4's feature maps
- $(5 \times 5 \times 16 + 1) \times 120 = 48120$ learnable params

**layer F6**
- 84 units and is fully connected to C5
- $121 \times 84 = 10164$ learnable params

**ouput layer**
- the ouput layer is Euclidiean RBF units, one for each class with 84 inputs each
- the ouputs of each unit is computed as follows
$$
y_h = \sum_j (x_j - w_{ij})^2
$$
- bassically measuring the distance between the 84 input vector and the known vector for the class

**loss**
- use a more discriminative loss that makes the correct class's distance small while making the incorrect classes' distances large
- use backprop to train all the CNN weights, since conv layers share weights, their gradients must be summed across all the connections that use the shared weights

### AlexNet

Eight learned layers; 5 convolutional, 3 fully connected. It showed that large/deep CNNs coudl acheive major gains on ImageNet. Demonstrated that CNNs could scale to millions of high-resolution images. GPU training made much larger CNNs practical

**ReLU**
- LeNet-5 used sigmoid or tanh activation functions. When the input is very positive, sigmoid gets close to 1. When the input is very negative, sigmoid gets close to 0. In both regions, the curve becomes almost flat. this is a problem because the this means the gradient becomes tiny. so if each layer has a small gradient and you are mulitplying repeatedly by small numbers, you have a vanishing gradient problem
- relu is $max(0, x)$, so for positive values $f(x) = x$, therefore $f'(x) = 1$
- so the gradient can pass through without being shrunk by the activation function
- AlexNet was a huge CNN and around 60 million parameters.
- one problem with ReLU is that for negative inputs the gradient is 0. so a neuron that stays negative can stop recieving gradietn through ReLU. this is related to the dying ReLU problem
- currently: smoother ReLU alternatives (GELU/SiLU)

**local response normalization**
- makes nearby CNN filters compete with each other
- LRN looks at nearby activations and divides each activation by a value based on how big the neighboring activations are
- this encourages network to a distinctive feature responses instead of letting many filter become active at the same time
$$
b_i = \frac{a_i}{\left(k + \alpha \sum_j a_j^2\right)^\beta}
$$
**overlapping pooling**
- Overlapping pooling means the pooling windows share some of the input. AlexNet found this generalized slightly better than non-overlapping pooling.

### ResNet
- training framework to ease the training of deeper networks
- deeper networks can actually perform worse because it is harder to optimize, this is called the degredation problem
- isntead of letting layers fit a desired mapping, they let them fit a residual
- $h(x) = f(x) + x$

### Sources
[LeNet-5](https://hal.science/hal-03926082/document)
[AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
[ResNet](https://arxiv.org/pdf/1512.03385)
