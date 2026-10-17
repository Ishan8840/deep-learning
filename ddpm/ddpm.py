import torch


betas = torch.linspace(0.0001, 0.02, steps=1000)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t):
    eps = torch.randn_like(x0)

    return torch.sqrt(alpha_bars[t]) * x0 + torch.sqrt(1 - alpha_bars[t]) * eps

