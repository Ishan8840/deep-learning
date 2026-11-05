# Loss functions

These are the notes behind the small [MSE](mse.py), [cross-entropy](ce.py), and [KL divergence](kl.py) implementations in this folder.

### Mean Squared Error
- calculates the loss by taking the average of the squared residuals, where the residual is the difference between the actual data and the predicted data
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- n is the number of observations in the dataset
- $y_i$ is the actual value of the observation
- $\hat{y}_i$ is the predicted value of the observation

**Gradient**
$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$
**Example**
actual values: [10, 20, 30, 40 , 50]
predicted values: [12, 18, 32, 38, 48]

squared differences: [4, 4, 4, 4, 4]
average of these: 4 = mse

gradient = [4, -4, 4, -4, -4] / 5

**Why**
- it penalizes large errors more heavily than small ones
- mathimatically easy

### Softmax
- logits are the raw unnormalized scores returned by the last layer of the neural network
- softmax takes those logits and transforms it into a probability vector, where each element is between (0, 1) and all the elements sum to 1
$$\text{softmax} = \frac{e^{a_i}}{\sum_{k=1}^N e^{a_k}}$$
### Cross-Entropy
- used for training models in classification
$$\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^C (y_{i,j} \cdot \log(p_{i,j}))
$$
- N is number of samples
- C is number of classes
- $y_{i,j}$ is 1 if class j is correct for sample i, 0 otherwise
- $p_{i,j}$ is model-predicted probability of sample i being in class j
- when using one hot encodings it reduces to this
$$\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i,\mathrm{correct}})$$
- log sum exp trick which prevents overflow which is when a number becomes too large for a computer to store numerically
$$
\log\left(\sum_i e^{x_i}\right)
=
m+\log\left(\sum_i e^{x_i-m}\right),
\qquad
m=\max_i(x_i)
$$
**Gradient**
$$\frac{\partial L}{\partial a_{i,j}} = \frac{p_{i,j} - y_{i,j}}{N}$$
### KL Divergence
- A way to quanitfy the difference between two probalility distributions
- Forward KL is mode covering, cover everything P cares about
- Reverse KL is mode seeking, stay inside good regions of P
$$KL = \sum_i P(i)\log \frac{P(i)}{Q(i)}$$
**Gaussian Distribution**
- A VAE has an encoder that looks at an input $x$ and produces a distribution for the latent variable $z$.
- KL divergence can keep the encoder's latent distribution close to a chosen prior.
