import torch
import torch.nn as nn


def kl(p, q):
    r = p / q
    r = p * torch.log(r)
    return r.sum(dim=1)



p = torch.softmax(torch.randn(2, 3), dim=1)
q = torch.softmax(torch.randn(2, 3), dim=1)

print(kl(p, q))


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    return (
        torch.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
        - 0.5
    )

mu1 = torch.tensor(10.0)
sigma1 = torch.tensor(2.0)

mu2 = torch.tensor(10.0)
sigma2 = torch.tensor(5.0)

print(gaussian_kl(mu1, sigma1, mu2, sigma2))