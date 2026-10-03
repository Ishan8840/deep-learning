import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_in, d_k, d_v):
        super().__init__()

        self.W_q = nn.Parameter(torch.randn(d_in, d_k) * 0.01)
        self.W_k = nn.Parameter(torch.randn(d_in, d_k) * 0.01)
        self.W_v = nn.Parameter(torch.randn(d_in, d_v) * 0.01)

        self.scale = d_k ** -0.5

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        wei = Q @ torch.transpose(K, 1, 2) * self.scale
        scores = torch.softmax(wei, dim=-1) @ V

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
                nn.GELU(),
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

        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class Transformer(nn.Module):
    """Vision Transformer for fixed-size square images; returns class logits."""

    def __init__(self, img_size, patch, in_ch, d_model, head_dim, num_heads, num_classes,
                 depth=6):
        super().__init__()

        dimensions = dict(img_size=img_size, patch=patch, in_ch=in_ch,
                          d_model=d_model, head_dim=head_dim, num_heads=num_heads,
                          num_classes=num_classes, depth=depth)
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if img_size % patch != 0:
            raise ValueError("img_size must be divisible by patch")

        self.img_size = img_size
        self.in_ch = in_ch
        N = (img_size // patch) ** 2

        self.patch_emb = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.cls       = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.randn(1, N + 1, d_model) * 0.02)
        self.blocks    = nn.ModuleList(TransformerBlock(d_model, head_dim, num_heads) for _ in range(depth))
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, num_classes)


    def forward(self, x):
        """Accept images shaped (batch, channels, height, width)."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_ch, self.img_size, self.img_size):
            raise ValueError(
                f"expected images shaped (batch, {self.in_ch}, "
                f"{self.img_size}, {self.img_size}), got {tuple(x.shape)}"
            )
        B = x.size(0)

        x = self.patch_emb(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_emb

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x[:, 0]))

if __name__ == "__main__":
    images = torch.randn(4, 3, 32, 32)
    model = Transformer(img_size=32, patch=4, in_ch=3, d_model=128,
                        head_dim=32, num_heads=4, num_classes=10)
    print(model(images).shape)  # (4, 10)
