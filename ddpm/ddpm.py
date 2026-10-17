import math

import torch


betas = torch.linspace(0.0001, 0.02, steps=1000)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t):
    eps = torch.randn_like(x0) # 64, 1, 28, 28

    xt = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1) * x0 + torch.sqrt(1 - alpha_bars[t]).view(-1, 1, 1, 1) * eps

    return xt, eps


def timestep_embedding(t, dim=128):
    half_dim = dim // 2 # 64

    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim)
    ).to(t.device) # 64

    args = t.float().unsqueeze(1) * freqs # t, 1 x 64 -> t, 64

    f1 = torch.cos(args) # t, 64
    f2 = torch.sin(args) # t, 64

    e = torch.stack((f1, f2), dim=2).flatten(1) # t, 128

    return e