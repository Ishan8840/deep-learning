import torch
import torch.nn as nn


betas = torch.linspace(0.0001, 0.02, steps=1000)
alphas  = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)