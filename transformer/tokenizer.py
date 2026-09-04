import torch
import torch.nn as nn


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join(self.itos[i] for i in tokens)