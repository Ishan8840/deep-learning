import torch
import torch.nn as nn


def kl(p, q):
    r = p / q
    r = p * torch.log(r)
    return r.sum(dim=1)



p = torch.softmax(torch.randn(2, 3), dim=1)
q = torch.softmax(torch.randn(2, 3), dim=1)

print(kl(p, q))