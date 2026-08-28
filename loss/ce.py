import torch
import torch.nn as nn


def softmax(logits):
    l_e = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)
    p = l_e / l_e.sum(dim=1).unsqueeze(1)

    return p

def ce(p, y):
    rows = torch.arange(p.size(0))
    ce = p[rows, y]
    ce = -torch.log(ce)
    ce = ce.mean()

    return ce

def c(logits, y):
    m = torch.max(logits, dim=1, keepdim=True).values
    exp_val = torch.exp(logits - m)
    lse = m + torch.log(exp_val.sum(dim=1).unsqueeze(1))

    rows = torch.arange(logits.size(0))
    a_c = logits[rows, y]

    return (-a_c + lse).mean()




logits = torch.tensor([[1000., 0., 0.], [1000., 0., 0.]])
y = torch.tensor([0, 1])

print(c(logits, y))
print(nn.CrossEntropyLoss()(logits, y))