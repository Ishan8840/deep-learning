# ACT

## Core Ideas

Small errors in predicted actions can compound into large errors over time.

- **Action chunks:** Predict target joint positions for the next $k$ timesteps, reducing the effective task horizon by a factor of $k$.
- **Temporal ensembling:** Query the policy more frequently and average overlapping action chunks.
- **CVAE training:** Train the policy with a reconstruction loss and a term that regularizes the latent distribution toward a Gaussian prior just like [VAE](../vae/README.md)s.

## CVAE Encoder: Learning the Style Variable

The encoder is a **BERT-like transformer**. For faster training, it uses proprioceptive observations and action sequences, leaving out images.

Its inputs are:

1. A learned `[CLS]` token.
2. Current joint positions $q_t$.
3. The demonstrated action sequence $A$ of length $k$.

The transformer outputs one vector per input token. The `[CLS]` output summarizes the sequence and predicts the **mean and variance of the style variable $z$**.

$z$ can capture different ways to perform the same task from the same observation, helping avoid averaging distinct trajectories.

## CVAE Decoder: Predicting an Action Chunk

The decoder receives the **current observation and $z$** and predicts the next $k$ actions. It combines ResNet image encoders, a transformer encoder, and a transformer decoder.

### 1. Encode the observation

- **Images:** ResNet18 backbones convert each $480\times640\times3$ RGB image into a $15\times20\times512$ feature map.
- **Flatten:** Convert each feature map into a sequence of $300\times512$.
- **Spatial information:** Add 2D sinusoidal positional embeddings.
- **Joint positions and $z$:** Project each to 512 dimensions with linear layers and append them to the image tokens.

The resulting **$1202\times512$** sequence is passed to the transformer encoder.

### 2. Decode the actions

The [transformer](../transformer/README.md) decoder takes $k$ positional query embeddings of dimension 512. Through **cross-attention**, these queries attend to the 1202 encoded observation tokens, which supply the keys and values.

The decoder produces a $k\times512$ sequence. A final MLP maps it to **$k\times14$**: target joint positions for the next $k$ steps.

## Training Loss

- **Reconstruction loss $L_a$:** Measures the error between demonstrated and predicted actions.
- **Regularization loss $L_r$:** Encourages the encoder's latent distribution to match the Gaussian prior.

$$
L_r = D_{\mathrm{KL}}\left(q_\phi(z\mid q_t,A)\,\|\,\mathcal N(0,I)\right)
$$

## Inference

At test time, set $z$ to the mean of the prior: **zero**.

## Model and Compute

| Detail | Value |
| --- | --- |
| Parameters | Approximately 80M |
| Training | From scratch for each task |
| Training time | Around 5 hours on a single 11 GB RTX 2080 Ti |
| Inference time | Around 0.01 seconds on the same GPU |

## Sources

[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705)
[VAE](../vae/README.md)
