import torch
import torch.nn as nn


class Attention(nn.Module):
  def __init__(self, input_size, k_size, v_size):
    super().__init__()

    self.k_dim = k_size

    self.W_q = nn.Parameter(torch.randn(input_size, k_size))
    self.W_k = nn.Parameter(torch.randn(input_size, k_size))
    self.W_v = nn.Parameter(torch.randn(input_size, v_size))

  def forward(self, x):
    Q = x @ self.W_q
    K = x @ self.W_k
    V = x @ self.W_v

    att = torch.softmax(Q @ K.T / torch.sqrt(torch.tensor(self.k_dim)), -1) @ V

    return att

attention = Attention(4, 3, 5)

att = attention(torch.randn(8, 4))


print(att)