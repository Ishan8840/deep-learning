import torch
import torch.nn as nn


# Single Head of Attention
class Attention(nn.Module):
  def __init__(self, input_size, k_size, v_size):
    super().__init__()

    self.k_dim = k_size

    self.W_q = nn.Parameter(torch.randn(input_size, k_size)) # (2, 3)
    self.W_k = nn.Parameter(torch.randn(input_size, k_size)) # (2, 3)
    self.W_v = nn.Parameter(torch.randn(input_size, v_size)) # (2, 5)

  def forward(self, x):
    Q = x @ self.W_q # (4, 2) (2, 3) -> (4, 3)
    K = x @ self.W_k # (4, 2) (2, 3) -> (4, 3)
    V = x @ self.W_v # (4, 2) (2, 5) -> (4, 5)

    att = Q @ K.T
    mask = torch.tril(torch.ones(att.shape))
    att = att.masked_fill(mask == 0, float('-inf'))
    att = torch.softmax(att / torch.sqrt(torch.tensor(self.k_dim)), -1) @ V

    return att

x = torch.randn(4, 2)
attention = Attention(2, 3, 5)

out = attention(x)

print(out)