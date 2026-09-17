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

    def forward(self, x, cache=None, use_cache=False):
        """cache is a (K, V) pair of everything seen so far, or None."""
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        if cache is not None:
            past_K, past_V = cache
            K = torch.cat([past_K, K], dim=1)
            V = torch.cat([past_V, V], dim=1)

        # Query i sits at absolute position off+i, so it may see keys 0..off+i.
        # off is 0 while training (full sequence) and t while generating token t.
        q_len, k_len = Q.size(1), K.size(1)
        off = k_len - q_len

        wei = Q @ torch.transpose(K, 1, 2) * self.scale
        masked_wei = wei.masked_fill(self.mask[off:k_len, :k_len]==0, float('-inf'))
        scores = torch.softmax(masked_wei, dim=-1) @ V

        return scores, ((K, V) if use_cache else None)


class MHA(nn.Module):
    def __init__(self, d_model, head_dim, num_heads):
        super().__init__()

        self.heads = nn.ModuleList([Attention(d_model, head_dim, head_dim) for _ in range(num_heads)])
        self.W_o   = nn.Parameter(torch.randn(num_heads * head_dim, d_model) * 0.02)

    def forward(self, x, cache=None, use_cache=False):
        outs, new_cache = [], []

        for i, att in enumerate(self.heads):
            out, kv = att(x, None if cache is None else cache[i], use_cache)
            outs.append(out)
            new_cache.append(kv)

        scores = torch.cat(outs, dim=-1)

        return scores @ self.W_o, (new_cache if use_cache else None)


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

    def forward(self, x, cache=None, use_cache=False):
        att, new_cache = self.mha(x, cache, use_cache)

        x = self.ln1(x + att)
        x = self.ln2(x + self.ffn(x))

        return x, new_cache


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

    @staticmethod
    def cache_len(cache):
        """How many tokens a cache already holds (0 if empty)."""
        return 0 if cache is None else cache[0][0][0].size(1)

    def forward(self, idx, cache=None, use_cache=False):
        """Returns logits, or (logits, cache) when use_cache is set."""
        T = idx.size(1)

        # With a cache these tokens sit AFTER everything cached, so their
        # positional embeddings start at past, not 0.
        past = self.cache_len(cache)
        pos = torch.arange(past, past + T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)

        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache[i], use_cache)
            new_cache.append(kv)

        logits = x @ self.lm_head

        return (logits, new_cache) if use_cache else logits

    @torch.no_grad()
    def generate(self, idx, max_new_t, temp=1.0, use_cache=True):
        if not use_cache:
            for _ in range(max_new_t):
                idx_cond = idx[:, -self.block_size:]      # 1. crop
                logits = self(idx_cond)                   # 2. (B, T, vocab)
                logits = logits[:, -1, :] / temp          # 3. last position only
                probs  = torch.softmax(logits, dim=-1)    # 4. (B, vocab)
                nxt    = torch.multinomial(probs, 1)      # 5. (B, 1)
                idx    = torch.cat([idx, nxt], dim=1)     # 6. append
            return idx

        cache = None
        step_in = idx[:, -self.block_size:]                # first pass: whole prompt

        for _ in range(max_new_t):
            # The cache holds absolute positions, so it cannot slide. Once it
            # fills block_size, drop it and rebuild from the cropped window.
            if self.cache_len(cache) + step_in.size(1) > self.block_size:
                cache = None
                step_in = idx[:, -self.block_size:]

            logits, cache = self(step_in, cache, use_cache=True)

            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            nxt   = torch.multinomial(probs, 1)

            idx = torch.cat([idx, nxt], dim=1)
            step_in = nxt                                  # only the new token

        return idx



if __name__ == "__main__":
    idx = torch.randint(0, 8, (4, 4))
    tra = Transformer(3, 4, 3, 8, 256)
    print(tra(idx).shape)