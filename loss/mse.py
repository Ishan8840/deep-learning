import torch
import torch.nn as nn

def mse(y, pred):
    res = (y - pred) ** 2
    res = torch.sum(res, dim=1) / y.shape[1]

    return res


y = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
pred = torch.zeros(2, 3)

loss = mse(y, pred)

print(loss)