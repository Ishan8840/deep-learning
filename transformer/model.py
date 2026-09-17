import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_in, d_k, d_v):
        super().__init__()

        self.W_q = nn.Parameter(torch.randn(d_in, d_k) * 0.01)
        self.W_k = nn.Parameter(torch.randn(d_in, d_k) * 0.01)
        self.W_v = nn.Parameter(torch.randn(d_in, d_v) * 0.01)

        self.register_buffer('mask', torch.tril(torch.ones(256, 256)))
        self.scale = d_k ** -0.5

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        T = x.size(1)

        wei = Q @ torch.transpose(K, 1, 2) * self.scale
        masked_wei = wei.masked_fill(self.mask[:T, :T]==0, float('-inf'))
        scores = torch.softmax(masked_wei, dim=-1) @ V

        return scores


class MHA(nn.Module):
    def __init__(self, d_model, head_dim, num_heads):
        super().__init__()

        self.heads = nn.ModuleList([Attention(d_model, head_dim, head_dim) for _ in range(num_heads)])
        self.W_o   = nn.Parameter(torch.randn(num_heads * head_dim, d_model) * 0.02)

    def forward(self, x):
        scores = torch.cat([att(x) for att in self.heads], dim=-1)

        return scores @ self.W_o


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.w = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
            )


    def forward(self, x):
        return self.w(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_dim, num_heads):
        super().__init__()

        self.mha = MHA(d_model, head_dim, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.mha(x))
        x = self.ln2(x + self.ffn(x))

        return x


class Transformer(nn.Module):
    def __init__(self, d_model, head_dim, num_heads, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, head_dim, num_heads) for _ in range(6))
        self.lm_head   = nn.Parameter(torch.randn(d_model, vocab_size) * 0.02)

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        return x @ self.lm_head

    @torch.no_grad()
    def generate(self, idx, max_new_t, temp=1.0):
        for _ in range(max_new_t):
            idx_cond = idx[:, -self.block_size:]          # 1. crop
            logits = self(idx_cond)                  # 2. (B, T, vocab)
            logits = logits[:, -1, :] / temp  # 3. last position only
            probs  = torch.softmax(logits, dim=-1)   # 4. (B, vocab)
            nxt    = torch.multinomial(probs, 1)     # 5. (B, 1)
            idx    = torch.cat([idx, nxt], dim=1)    # 6. append
        return idx



if __name__ == "__main__":
    idx = torch.randint(0, 8, (4, 4))
    tra = Transformer(3, 4, 3, 8, 256)
    print(tra(idx).shape)