The transformer is the first transduction model relying entirely on Attention to compute representations of its input and output without sequence aligned RNN + LSTM or CNN.

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
