# DiT

DiT replaces the U-Net backbone in latent diffusion with a [Transformer](../../01-foundations/04-transformer/README.md), following the design practices of [ViTs](../../01-foundations/05-vision-transformer/README.md). Diffusion operates in a [VAE](../../02-generative-models/01-vae/README.md)'s latent space.

## Latent representation

A pretrained VAE encoder compresses the image into a latent representation:

$$
x\in\mathbb{R}^{256\times256\times3}
\quad\xrightarrow{\text{VAE encoder}}\quad
z\in\mathbb{R}^{32\times32\times4}
$$

## Patch embeddings

With patch size $p=2$, divide the latent's $32\times32$ spatial dimensions into $2\times2$ patches.

| Quantity | Shape or value |
| --- | --- |
| Patches along each spatial dimension | $32/2=16$ |
| Total patches | $16\times16=256$ |
| Shape of each patch | $2\times2\times4$ |
| Flattened patch | $16$ values |
| Projected token sequence | $256\times d$ |

Flatten each patch into 16 values, then apply a linear projection from 16 to the hidden dimension $d$:

$$
[256,2,2,4]\rightarrow[256,16]\rightarrow[256,d]
$$

## Positional embeddings

Add a 2D sine-cosine positional embedding to each patch token:

$$
\text{token}_i = \text{patch embedding}_i + \text{position embedding}_i
$$

## Conditioning

The timestep $t$ and class $c$ can condition the image tokens in four ways:

| Method | How it works |
| --- | --- |
| In-context | Append timestep and class embeddings as extra tokens for normal self-attention. |
| Cross-attention | Keep conditioning embeddings separate; image tokens attend to them through an added cross-attention layer. |
| adaLN | Generate $\gamma$ and $\beta$ to scale and shift normalized image-token features, similar to FiLM. |
| adaLN-Zero | Add zero-initialized residual gates to adaLN so each transformer block starts as an identity and gradually learns its transformations. |

## Output projection

Each token represents a $p\times p\times C$ latent patch. DiT predicts both noise $\epsilon_\theta$ and diagonal covariance, so the final linear layer maps each token to:

$$
d\rightarrow p\times p\times 2C
$$

With $p=2$ and $C=4$, each of the 256 tokens produces $2\times2\times8=32$ values.

## Unpatchify

Arrange the outputs in their original $16\times16$ patch grid, then reassemble the spatial dimensions:

$$
[256,32]\rightarrow[16,16,2,2,8]\rightarrow[32,32,8]
$$

Split the 8 output channels into two predictions:

| Prediction | Shape |
| --- | --- |
| Noise $\epsilon_\theta$ | $32\times32\times4$ |
| Diagonal covariance | $32\times32\times4$ |

#### EMA
keeps a slowly moving average of the model's weights during training, which gives you a smoother and often better model to use at inference.


## Sources

- [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)
- [DDPM](../../02-generative-models/02-ddpm/README.md)
