import math
import torch.nn as nn
import torch
import torch.nn.functional as F


betas = torch.linspace(0.0001, 0.02, steps=1000)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t):
    eps = torch.randn_like(x0) # 64, 1, 28, 28

    xt = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1) * x0 + torch.sqrt(1 - alpha_bars[t]).view(-1, 1, 1, 1) * eps

    return xt, eps


def timestep_embedding(t, dim=128):
    half_dim = dim // 2 # 64

    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim)
    ).to(t.device) # 64

    args = t.float().unsqueeze(1) * freqs # t, 1 x 64 -> t, 64

    f1 = torch.cos(args) # t, 64
    f2 = torch.sin(args) # t, 64

    e = torch.stack((f1, f2), dim=2).flatten(1) # t, 128

    return e


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )

        self.time = nn.Linear(time_dim, out_channels)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, t):
        x_conv = self.conv(x)
        t_feat = self.time(t)
        t_feat = t_feat[:, :, None, None]

        f = x_conv + t_feat

        x_s = self.shortcut(x)

        return f + x_s


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(1, 32)
        self.down = Downsample(32)
        self.block2 = ConvBlock(32, 64)


    def forward(self, x, t_emb):

        h1 = self.block1(x, t_emb)
        h2 = self.down(h1)
        h3 = self.block2(h2, t_emb)

        return h3, h1


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = Upsample(64)
        self.block = ConvBlock(96, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, h, skip, t_emb):

        up = self.up(h)
        ccat = torch.cat((up, skip), dim=1)
        h2 = self.block(ccat, t_emb)
        out = self.out(h2)

        return out


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = ConvBlock(64, 64)
        self.decoder = Decoder()

    def forward(self, x, t):
        t_emb = timestep_embedding(t)

        encoded, skip = self.encoder(x, t_emb)
        bottleneck = self.bottleneck(encoded, t_emb)
        decoded = self.decoder(bottleneck, skip, t_emb)

        return decoded


def train_step(model, optimizer, x0):

    batch = x0.shape[0]
    t = torch.randint(low=0, high=1000, size=(batch,), device=x0.device)

    x_t, eps = q_sample(x0, t)
    eps_pred = model(x_t, t)

    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
