# Transformer

[`transformer.py`](transformer.py) is a one-file, character-level language model adapted from my [nanoGPT](https://github.com/Ishan8840/nanoGPT) repo. It contains the tokenizer, causal multi-head self-attention, Transformer blocks, training loop, checkpoint loading, and text generation in one place.

## The idea

Attention lets each token combine information from other positions. Each head projects the input into queries, keys, and values, then computes

$$\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,$$

where $M$ masks future tokens. The model adds token and learned positional embeddings, stacks pre-normalized attention and feed-forward blocks with residual connections, and predicts the next character. This is a **decoder-only** Transformer. The original [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) model also had an encoder and cross-attention for translation.

## Run it

From the repository root, train briefly on the built-in demo text:

```bash
uv run python 01-foundations/06-transformer/transformer.py --steps 20 --context 32 --layers 2 --new-tokens 40
```

For a real character-level experiment, pass a UTF-8 text file and train longer:

```bash
uv run python 01-foundations/06-transformer/transformer.py --text path/to/input.txt --steps 5000 --prompt "Hello" --new-tokens 200
uv run python 01-foundations/06-transformer/transformer.py --load --prompt "Hello" --new-tokens 200
```

The checkpoint defaults to `outputs/transformer.pt`, which Git ignores. A prompt can use only characters seen during training. The short demo confirms that the pipeline works; it is not a language-quality result.

This version omits nanoGPT's KV cache and attention visualizer so the complete learning path stays in one readable file. Generation recomputes the current context for each new token.

The study notes below cover the original Transformer. Continue with [attention](../05-attention/README.md), [LoRA](../11-lora/README.md), [Mixture of Experts](../12-mixture-of-experts/README.md), and [Mixture of Transformers](../13-mixture-of-transformers/README.md).

## Study notes

The transformer is the first transduction model relying entirely on [Attention](../05-attention/README.md) to compute representations of its input and output without sequence aligned [RNN + LSTM](../04-recurrent-networks/README.md) or [CNN](../03-cnn/README.md).

#### Encoder
- stack of 6 identical layers, each layer has two sub-layers
- multi-head self-attention and a simply position wise fully connected feed-forward neural network
- residual connection around each of the sublayers + layer normalization

#### Decoder
- stack of 6 identical layers
- third sublayer, performs multi-head attention over the output of the encoder stack

####  Position-Wise FFN
- two linear transformations with a ReLU activation between them

#### Positional encoding

One catch: attention is permutation-invariant. Without positional information, the model can't tell the difference between "dog bites man" and "man bites dog". The original paper handles this with sine and cosine functions of different frequencies, added to the input embeddings.

Sinusoidal positional encoding is rarely used now. Modern models lean on **rotary position encoding (RoPE)** or **attention with linear biases (ALiBi)** instead.

The original transformer was also designed for translation, an encoder-decoder setup. Today, the dominant variant is **decoder-only** (GPT-style), which keeps just the masked self-attention side of the original architecture.

#### KV caching

When a decoder-only model generates text, it produces one token at a time. Naively, each new token would recompute attention over the entire sequence from scratch, including all the keys and values for tokens that haven't changed since the last step. That's wasteful.

The trick: for each token we generate, we cache its **K** and **V** vectors. On the next step, we only compute K and V for the new token and append them to the cache. We still compute a fresh **Q** for the new token (since it's the one doing the looking), but the keys and values it attends to are read from the cache.

This turns generation from quadratic to linear per step. The tradeoff is memory, the KV cache grows with sequence length and quickly dominates GPU memory for long contexts.

#### Grouped-query attention (GQA)

Standard multi-head attention gives every head its own Q, K, and V projections. If you have 32 heads, you're storing 32 separate K and V tensors per layer in the KV cache. Painful.

**Multi-query attention (MQA)** went to the other extreme: all heads share a single K and V, with only Q being per-head. Way smaller cache, but quality drops.

**Grouped-query attention** is the compromise that won: heads are split into groups, and each group shares one K and V. So 32 query heads might map to 8 KV heads. You get most of the quality of full multi-head attention with a fraction of the cache size. Llama 2/3, Mistral, and most current open models use GQA.

### Sources
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
