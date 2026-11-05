# Transformer

[`transformer.py`](transformer.py) is a one-file, character-level language model adapted from my [nanoGPT](https://github.com/Ishan8840/nanoGPT) repo. It contains the tokenizer, causal multi-head self-attention, Transformer blocks, training loop, checkpoint loading, and text generation in one place.

## The idea

Attention lets each token combine information from other positions. Each head projects the input into queries, keys, and values, then computes

$$\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,$$

where $M$ masks future tokens. The model adds token and learned positional embeddings, stacks pre-normalized attention and feed-forward blocks with residual connections, and predicts the next character. This is a **decoder-only** Transformer. The original [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) model also had an encoder and cross-attention for translation.

## Run it

From the repository root, train briefly on the built-in demo text:

```bash
uv run python 01-foundations/04-transformer/transformer.py --steps 20 --context 32 --layers 2 --new-tokens 40
```

For a real character-level experiment, pass a UTF-8 text file and train longer:

```bash
uv run python 01-foundations/04-transformer/transformer.py --text path/to/input.txt --steps 5000 --prompt "Hello" --new-tokens 200
uv run python 01-foundations/04-transformer/transformer.py --load --prompt "Hello" --new-tokens 200
```

The checkpoint defaults to `outputs/transformer.pt`, which Git ignores. A prompt can use only characters seen during training. The short demo confirms that the pipeline works; it is not a language-quality result.

This version omits nanoGPT's KV cache and attention visualizer so the complete learning path stays in one readable file. Generation recomputes the current context for each new token.
