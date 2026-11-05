# Transformer

[`transformer.py`](transformer.py) is a one-file, character-level language model adapted from my [nanoGPT](https://github.com/Ishan8840/nanoGPT) repo. It contains the tokenizer, causal multi-head self-attention, Transformer blocks, training loop, checkpoint loading, and text generation in one place.

| File |
| --- |
| [transformer.py](transformer.py) |

Run from the repository root:

```bash
uv run python 01-foundations/04-transformer/transformer.py --steps 20 --context 32 --layers 2 --new-tokens 40
```

[Read the visual notes](https://ishan8840.github.io/deep-learning-notes/#/01-foundations/06-transformer).
