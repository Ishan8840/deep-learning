"""One-file character-level GPT: causal attention, training, and generation."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, context)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(),
                                         nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, context=64, width=64, heads=4, layers=4):
        super().__init__()
        if min(vocab_size, context, width, heads, layers) < 1:
            raise ValueError("model dimensions must be positive")
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(TransformerBlock(width, heads, context)
                                      for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or tokens.shape[1] > self.context:
            raise ValueError("tokens must have shape (batch, length <= context)")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, new_tokens, temperature=1.0):
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        for _ in range(new_tokens):
            logits, _ = self(tokens[:, -self.context:])
            next_logits = logits[:, -1]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial((next_logits / temperature).softmax(dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def sample_batch(data, batch_size, context, device):
    starts = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i:i + context] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + context + 1] for i in starts]).to(device)
    return x, y


def train(args):
    text = (args.text.read_text() if args.text else
            "to learn is to build. to build is to understand.\n" * 100)
    characters = sorted(set(text))
    if len(characters) < 2:
        raise ValueError("training text needs at least two distinct characters")
    stoi = {char: index for index, char in enumerate(characters)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if min(len(train_data), len(val_data)) <= args.context:
        raise ValueError("both text splits must be longer than the context window")
    config = dict(vocab_size=len(characters), context=args.context, width=args.width,
                  heads=args.heads, layers=args.layers)
    torch.manual_seed(args.seed)
    model = GPT(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train_data, args.batch_size, args.context, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 5) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = sample_batch(val_data, args.batch_size, args.context, args.device)
                _, val_loss = model(vx, vy)
            model.train()
            print(f"step {step:4d} | train {loss.item():.3f} | val {val_loss.item():.3f}")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=config, characters=characters), args.checkpoint)
    print(f"saved {args.checkpoint}")
    return model, characters


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GPT(**saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved["characters"]


def show_sample(model, characters, prompt, new_tokens, temperature, device):
    stoi = {char: index for index, char in enumerate(characters)}
    if not prompt:
        prompt = characters[0]
    unknown = set(prompt) - set(stoi)
    if unknown:
        raise ValueError(f"prompt has characters outside the training vocabulary: {sorted(unknown)}")
    tokens = torch.tensor([[stoi[char] for char in prompt]], device=device)
    model.eval()
    output = model.generate(tokens, new_tokens, temperature)
    print("".join(characters[index] for index in output[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, help="Plain-text corpus; omit for a small demo corpus")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/transformer.pt"))
    parser.add_argument("--load", action="store_true", help="Generate from an existing checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    model, characters = load(args.checkpoint, args.device) if args.load else train(args)
    show_sample(model, characters, args.prompt, args.new_tokens, args.temperature, args.device)
