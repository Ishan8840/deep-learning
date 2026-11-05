# Mixture of Transformers

**Mixture of Transformers (MoT)** is a multimodal architecture that gives text, image, and speech tokens separate Transformer parameters while keeping **global self-attention** across the full sequence. A token uses the parameters for its own modality, but it can still attend to tokens from every modality. This increases total model capacity without increasing the amount of Transformer computation used by each token at a fixed model width.

## Why separate parameters by modality?

An early-fusion multimodal [Transformer](../06-transformer/README.md) puts text, image, and speech tokens in one sequence and processes them with shared parameters. These modalities have different structures: text is discrete and linguistic, images are spatial, and speech has temporal and acoustic structure.

The paper gives several reasons to consider separate parameters:

- **Training is expensive.** Chameleon used 9.2 trillion multimodal tokens to reach text performance comparable to LLaMA 2 trained on 2 trillion text tokens. These token counts include different data mixes, so they are an illustration rather than a controlled comparison.
- **Training objectives can interfere.** Updates that help one modality may not help another, especially when modalities use different objectives.
- **Representations separate anyway.** In a dense multimodal model, the authors used principal component analysis (PCA) to visualize hidden states and found that text, image, and speech tokens occupied distinct regions of the feature space.

A [Mixture of Experts](../12-mixture-of-experts/README.md) also provides sparse computation, but its learned router can use experts unevenly and make training harder to balance. MoT uses each token's known modality to choose its parameters.

## How MoT works

For each modality, MoT has separate **attention projection matrices**, **feed-forward networks (FFNs)**, and **layer normalization parameters**. If token $x_i$ has modality $m_i$, its query, key, and value are projected with that modality's weights:

$$
Q_i = x_i W_Q^{m_i}, \qquad
K_i = x_i W_K^{m_i}, \qquad
V_i = x_i W_V^{m_i}.
$$

For example, a text token uses $W_Q^{\text{text}}$, while an image token uses $W_Q^{\text{image}}$. The same modality-specific choice applies to the key, value, output projection, FFN, and layer norms.

**Attention remains global:** after the modality-specific projections, queries, keys, and values are restored to their original sequence order. Each token can attend to text, image, and speech tokens in the same attention operation. The attention output then passes through the output projection and FFN for that token's modality.

At a fixed model width, each token runs one set of projections and one FFN, as it would in a dense Transformer. The model stores more parameters because it has a separate set for each modality, but only the relevant set is active for a given token.

## Reported results

In the paper's Chameleon text-and-image setting, a 7B MoT model matched the dense baseline's performance using **55.8% of its training FLOPs**. With speech added as a third modality, it reached comparable speech performance with **37.2% of the dense baseline's FLOPs for speech**. These comparisons refer to the training needed to reach similar performance; a MoT layer at the same width has the same per-token FLOP count as a dense layer.

## Source
[Liang et al., *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*](https://arxiv.org/abs/2411.04996)
