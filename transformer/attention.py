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


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, input_size, k_size, v_size, model_size):
    super().__init__()

    self.W_o = nn.Parameter(torch.randn(model_size, input_size))
    self.heads = nn.ModuleList([Attention(input_size, k_size, v_size) for _ in range(num_heads)])

  def forward(self, x):
    att = [head(x) for head in self.heads]
    att = torch.cat(att, dim=2)
    att = att @ self.W_o

    return att
    
