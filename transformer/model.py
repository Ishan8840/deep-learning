import torch
import torch.nn as nn



class TokenEmbedding(nn.Module):
  def __init__(self, input_size, d_model):
    super().__init__()

    self.w = nn.Parameter(torch.randn(input_size, d_model))

  def forward(self, x):
    return self.w[x]


class PositionalEncoding(nn.Module):
  def __init__(self, block_size, d_model):
    super().__init__()

    pe = torch.zeros(block_size, d_model)

    position = torch.arange(block_size).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2)
        * (-torch.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer("pe", pe)

  def forward(self, x):
    T = x.shape[1]
    return self.pe[:T]


# Single Head of Attention
class Attention(nn.Module):
  def __init__(self, input_size, k_size, v_size):
    super().__init__()

    self.k_dim = k_size

    self.W_q = nn.Parameter(torch.randn(input_size, k_size)) # (2, 3)
    self.W_k = nn.Parameter(torch.randn(input_size, k_size)) # (2, 3)
    self.W_v = nn.Parameter(torch.randn(input_size, v_size)) # (2, 5)

  def forward(self, x, causal):
    Q = x @ self.W_q # (4, 2) (2, 3) -> (4, 3)
    K = x @ self.W_k # (4, 2) (2, 3) -> (4, 3)
    V = x @ self.W_v # (4, 2) (2, 5) -> (4, 5)

    att = Q @ torch.transpose(K, 1, 2)
    if causal:
      mask = torch.tril(torch.ones(att.shape[1], att.shape[2]))
      att = att.masked_fill(mask == 0, float('-inf'))
    
    att = torch.softmax(att / torch.sqrt(torch.tensor(self.k_dim)), -1) @ V

    return att


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, input_size, k_size, v_size, model_size):
    super().__init__()

    self.W_o = nn.Parameter(torch.randn(model_size, input_size))
    self.heads = nn.ModuleList([Attention(input_size, k_size, v_size) for _ in range(num_heads)])

  def forward(self, x):
    att = [head(x, causal=True) for head in self.heads]
    att = torch.cat(att, dim=2)
    att = att @ self.W_o

    return att


class FFN(nn.Module):
  def __init__(self, d_model):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(d_model, 4 * d_model),
      nn.ReLU(),
      nn.Linear(4 * d_model, d_model)
    )

  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  def __init__(self, d_model):
    super().__init__()

    self.ln1 = nn.LayerNorm(d_model)
    self.mha = MultiHeadAttention(
            num_heads=4,
            input_size=d_model,
            k_size=d_model // 4,
            v_size=d_model // 4,
            model_size=d_model
          )
    self.ln2 = nn.LayerNorm(d_model)
    self.ffn = FFN(d_model)

  def forward(self, x):
    x = x + self.mha(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x