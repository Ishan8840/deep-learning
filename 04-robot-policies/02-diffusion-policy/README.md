# Diffusion Policy

[`model.py`](model.py) implements a transformer-based action denoiser conditioned on PushT observations. [`train.py`](train.py) trains it and [`eval.py`](eval.py) runs closed-loop rollouts. The [example rollout](success_rollout.mp4) shows one successful episode. See [DiT architecture notes](DiT.md) for the image-generation model that motivated the backbone.

## Core Ideas

Represent a robot's **visuomotor policy as a conditional denoising diffusion process**.

- Model **multimodal action distributions**, addressing the concern about distinct action trajectories discussed in [ACT](../01-action-chunking/README.md).
- Use action sequences in closed-loop control.
- Predict actions conditioned on observations without also inferring future states, reducing computation and improving action accuracy.

## Adapting DDPM to Robot Actions

[DDPM](../../02-generative-models/02-ddpm/README.md)s become diffusion policies through two changes:

1. The output represents **robot actions**.
2. The denoising process is **conditioned on robot observations**.

## CNN Architecture: Temporal U-Net

The example below uses **16 action timesteps**, each with **2 action dimensions**. Feature shapes are written as **timesteps × channels**.

### 1. Project and Condition

- Project each action into a larger feature dimension.
- Apply **1D temporal convolutions** to mix information between nearby action timesteps.
- Use **FiLM** to add conditioning.

Example conditioned features:

$$
h\in\mathbb{R}^{16\times256}
$$

### 2. Downsample

Downsample along the **time dimension**. For example:

$$
[16,256]\rightarrow[8,512]
$$

The next level represents 8 broader temporal regions instead of 16 individual action positions.

### 3. Upsample with Skip Connections

Upsampling restores a separate feature vector for each action position, now informed by the **larger trajectory structure**.

Downsampling can lose fine-grained temporal information. **Skip connections** pass saved features from the downsampling path directly to the corresponding upsampling layers to preserve those details.

The resulting features return to:

$$
h_{\mathrm{final}}\in\mathbb{R}^{16\times256}
$$

### 4. Project to Predicted Noise

The final layer maps each 256-dimensional feature vector back to the original 2 action dimensions:

$$
[16,256]\rightarrow[16,2]
$$

This produces a noise prediction with the same shape as the action sequence.

### Limitation: Rapid Action Changes

Temporal convolutions tend to favor low-frequency signals. This likely explains poorer performance when actions change sharply over time, as with velocity commands.

## [Transformer](../../01-foundations/04-transformer/README.md) Architecture

The transformer decoder receives two input sequences:

| Input | Preparation |
| --- | --- |
| Noisy actions $A_t^k$ | Use actions as tokens and prepend a sinusoidal embedding of diffusion step $k$. |
| Observation $O_t$ | Use a shared MLP to produce an observation embedding sequence that conditions the decoder. |

## Visual Encoder

Use **ResNet-18 without pretraining**, with two modifications:

- **Spatial softmax pooling** replaces global average pooling to preserve spatial information (Mandlekar et al., 2021).
- **GroupNorm** replaces BatchNorm for stable training (Wu & He, 2018).

## Noise Schedule and Sampling

| Setting | Choice |
| --- | --- |
| Noise schedule | Squared cosine |
| Sampling method | DDIM |
| Training diffusion steps | 100 |
| Inference denoising steps | 10 |

## Position vs. Velocity Control

Diffusion Policy performs better with **position control** than velocity control:

- Action multimodality is more pronounced in position control.
- Position control suffers less from compounding errors, making it more suitable for action-sequence prediction.

## Sources

[Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137)
