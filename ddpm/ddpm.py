import torch


betas = torch.linspace(0.0001, 0.02, steps=1000)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t):
    eps = torch.randn_like(x0) # 64, 1, 28, 28

    xt = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1) * x0 + torch.sqrt(1 - alpha_bars[t]).view(-1, 1, 1, 1) * eps

    return xt, eps