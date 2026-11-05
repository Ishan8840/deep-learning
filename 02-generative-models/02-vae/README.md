# Variational Autoencoders (VAEs)

[`autoencoder.py`](autoencoder.py) trains a VAE on MNIST. The checked-in [latent space](latent_space.png) and [interpolation](latent_interpolation.png) images show example outputs. The note below explains the probabilistic objective behind the code.

See also the shorter [Autoencoder](../01-autoencoder/README.md) note for the deterministic starting point.

> A VAE learns to generate data from latent variables. Its encoder maps an example to a distribution over latent vectors, and its decoder maps a sampled latent vector to a distribution over possible outputs.

## 1. Intuition: why latent variables?

Imagine generating handwritten digits. Sampling each pixel independently would ignore the structure of the image: the top and bottom halves need to form a coherent digit.

Instead, we introduce a **latent vector** $z$: a hidden representation that can capture factors such as digit shape, stroke thickness, and slant. These factors are learned from data; we do not explicitly assign a meaning to each coordinate.

The decoder uses this shared latent vector to generate a coherent image. Even if it models pixels as independent *given* $z$, they can still be dependent overall because they share the same $z$.

### Notation

| Symbol | Meaning |
| --- | --- |
| $x$ | An observed example, such as an image |
| $z$ | A latent vector |
| $p(z)$ | Prior: a distribution over latent vectors before observing $x$ |
| $p_\theta(x\mid z)$ | Decoder: a distribution over outputs given $z$ |
| $p_\theta(z\mid x)$ | True posterior: a distribution over latent vectors given $x$ |
| $q_\phi(z\mid x)$ | Encoder: a learned approximation to the true posterior |
| $\theta,\phi$ | Decoder and encoder parameters, respectively |

## 2. The generative process

To generate a new image:

1. Sample a latent vector from a simple prior, usually $z\sim\mathcal N(0,I)$.
2. Use the decoder to sample an image: $x\sim p_\theta(x\mid z)$.

The neural network learns how to turn simple Gaussian latent vectors into a complex distribution over images.

### What do we want to maximize?

We want the model to assign high likelihood to the training data. The likelihood of one image sums over all possible latent explanations:

$$
p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz
$$

Training aims to maximize the sum of $\log p_\theta(x)$ across the dataset.

**The difficulty:** For a neural decoder, this integral is generally intractable to evaluate exactly. Estimating it by sampling from the prior can also be inefficient: most sampled latent vectors may assign very low likelihood to a particular training image.

## 3. The true posterior: which latent vectors explain an image?

Generation goes from $z$ to $x$. Given an observed image, we also want to reason in the other direction: which latent vectors could have produced it?

Bayes' rule gives the answer:

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)}
$$

This is the **true posterior** for the current generative model.

**The difficulty:** Its denominator is the same intractable likelihood from the previous section, so we cannot usually evaluate the posterior directly.

## 4. The encoder: approximate the true posterior

We introduce an encoder that learns an approximation:

$$
q_\phi(z\mid x)\approx p_\theta(z\mid x)
$$

Given an image of a $7$, the encoder should concentrate probability on latent vectors that help the decoder reconstruct that particular $7$.

The encoder outputs a **distribution**, rather than just a single vector. A common choice is a Gaussian whose mean and variance are predicted by a neural network:

$$
q_\phi(z\mid x)
=\mathcal N\!\left(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right)
$$

During training, we sample from this distribution to obtain latent vectors suited to the input. To generate new images after training, we sample from the prior $p(z)$ and use the decoder.

## 5. KL divergence: measure the approximation gap

**KL divergence** measures a discrepancy between two probability distributions. Here, we use it to compare the encoder with the true posterior:

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
=\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)-\log p_\theta(z\mid x)
\right]
$$

Read this as three steps:

1. Sample $z$ from the encoder distribution.
2. Compare the log densities assigned to $z$ by the encoder and the true posterior.
3. Average the difference over the encoder distribution.

KL divergence is nonnegative and equals zero when the distributions match almost everywhere. It is not symmetric, so the order of the distributions matters.

**We cannot evaluate this KL directly** because it contains the true posterior. Instead, we use Bayes' rule to rearrange it into a tractable training objective.

## 6. The ELBO: an objective we can optimize

Substituting Bayes' rule into the KL expression gives:

$$
\begin{aligned}
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
&=\log p_\theta(x)
-\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]\\
&\quad+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right).
\end{aligned}
$$

Rearranging yields:

$$
\log p_\theta(x)
=\mathcal L_{\mathrm{ELBO}}(x)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
$$

where the **evidence lower bound (ELBO)** is:

$$
\boxed{
\mathcal L_{\mathrm{ELBO}}(x)
=\underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{reconstruction term}}
-\underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)}_{\text{prior regularization}}
}
$$

Because the posterior KL is nonnegative, $\mathcal L_{\mathrm{ELBO}}(x)\leq\log p_\theta(x)$. We train both networks by maximizing this lower bound, usually estimating the reconstruction expectation with samples.

| Term | What optimizing it encourages |
| --- | --- |
| Reconstruction term | Latent vectors that let the decoder assign high likelihood to the input |
| Prior regularization | Encoder distributions that stay close to the prior used for generation |

**Keep the two KL terms distinct:** The KL to the *true posterior* is the gap between the ELBO and the log likelihood. The KL to the *prior* is the regularizer we actually evaluate in the objective.

When implementing training as a minimization problem, use the negative ELBO:

$$
\text{VAE loss}
=-\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)
$$

## Sources

- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)
- [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908)
