# ALOHA Unleashed

## Core Architecture

A [Transformer](../../01-foundations/04-transformer/Transformer.md)-based architecture trained with a **diffusion loss**, scaling up an encoder–decoder design similar to [ACT](../01-action-chunking/README.md).

## Observation Encoder

### 1. Encode the Images

- Resize each of the **4 RGB images** to $480\times640\times3$.
- Feed each image into a separate **ResNet50** backbone.
- Take the stage 4 feature map: $15\times20\times512$ per image.
- Flatten the spatial dimensions to obtain **1,200 image tokens**, each with 512 dimensions, across all four images.

### 2. Add the Robot State

Project the robot's **proprioceptive state**—joint positions and gripper values for both arms—into one additional embedding.

This gives **1,201 tokens**, each with 512 dimensions.

### 3. Encode the Tokens

Add positional embeddings and pass the sequence through an **85M-parameter transformer encoder**.

## Diffusion Denoiser

The observation latents condition a **55M-parameter transformer decoder** with bidirectional attention.

### Inputs and Conditioning

- **Noised action chunk:** A $50\times14$ tensor, with learned positional embeddings.
- **Observation conditioning:** Action embeddings cross-attend to the observation encoder's latent embeddings.
- **Diffusion timestep:** Represented as a one-hot vector and also supplied through cross-attention.

### Output Projection

The decoder produces a $50\times512$ sequence. A linear layer projects it back to the action dimensions:

$$
[50,512]\rightarrow[50,14]
$$

## Sources

[Aloha Unleashed](https://aloha-unleashed.github.io/assets/aloha_unleashed.pdf)
[CNN](../../01-foundations/02-cnn/CNN.md)
[VAE](../../02-generative-models/01-vae/README.md)
[DDPM](../../02-generative-models/02-ddpm/README.md)
