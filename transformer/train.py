"""Train the character-level transformer on tinyShakespeare.

    python train.py
"""

import time
from pathlib import Path

import torch

from model import Transformer
from tokenizer import CharTokenizer


# ------------------------------------------------------------------- settings

DATA = Path(__file__).resolve().parents[1] / "data" / "tinyShakespeare" / "input.txt"
CKPT = Path(__file__).resolve().parent / "ckpt.pt"

block_size = 128          # context window; must be <= 256 (Attention's mask)
batch_size = 32
d_model    = 128
num_heads  = 4
head_dim   = d_model // num_heads

max_iters   = 3000
eval_every  = 250
eval_iters  = 20
lr          = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# ----------------------------------------------------------------------- data

text = DATA.read_text()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}


def get_batch(split):
    """Sample batch_size random windows of block_size (+1 for the targets)."""
    d = splits[split]
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def loss_fn(logits, y):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
    )


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            losses[k] = loss_fn(model(x), y).item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------- model

model = Transformer(
    d_model=d_model,
    head_dim=head_dim,
    num_heads=num_heads,
    vocab_size=tok.vocab_size,
    block_size=block_size,
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)

n_params = sum(p.numel() for p in model.parameters())
print(f"device {device} | vocab {tok.vocab_size} | {n_params/1e6:.2f}M params")
print(f"data {len(data):,} tokens -> {n:,} train / {len(data)-n:,} val")
print(f"expected loss at init: ~{torch.tensor(float(tok.vocab_size)).log():.3f}\n")


# ------------------------------------------------------------------- training

t0 = time.time()
for it in range(1, max_iters + 1):
    x, y = get_batch("train")

    loss = loss_fn(model(x), y)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if it % eval_every == 0 or it == 1:
        losses = estimate_loss()
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| {time.time()-t0:6.1f}s")

torch.save({"model": model.state_dict(), "chars": tok.chars}, CKPT)
print(f"\nsaved {CKPT}")


# ------------------------------------------------------------------- sampling

start = torch.zeros((1, 1), dtype=torch.long, device=device)   # newline
out = model.generate(start, max_new_t=500, temp=0.8)
print("\n" + "-" * 60)
print(tok.decode(out[0].tolist()))
