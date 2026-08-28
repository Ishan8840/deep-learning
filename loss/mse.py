import torch
import torch.nn as nn

def mse(y, pred):
    res = ((y - pred) ** 2).mean()

    return res

y = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
pred = torch.zeros(2, 3, requires_grad=True)

loss = mse(y, pred)
loss.backward()

print(pred.grad)
print(nn.MSELoss()(pred, y))

