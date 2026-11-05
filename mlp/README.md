# Multilayer perceptron on FashionMNIST

[`pt_nn.py`](pt_nn.py) is a small image-classification training loop. It flattens each 28 × 28 grayscale image into 784 values, then applies two 512-unit linear layers with GELU and dropout before producing ten class logits.

The code uses cross-entropy loss, AdamW, and a cosine learning-rate schedule. It splits the training set into training and validation subsets, evaluates on the separate test set, and plots training and validation loss. This is a baseline for understanding the training loop rather than a recorded benchmark result.

Run it from this folder with the base project environment:

```bash
uv run python pt_nn.py
```

The script downloads FashionMNIST if needed and opens a Matplotlib plot at the end.
