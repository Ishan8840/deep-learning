import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from diffusers import DDPMScheduler


noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon",
)

betas = noise_scheduler.betas
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t):
    eps = torch.randn_like(x0) # 64, 16, 2

    shape = [x0.shape[0]] + [1] * (x0.ndim - 1)

    alpha_bar = alpha_bars.to(device=x0.device, dtype=x0.dtype)[t]
    xt = alpha_bar.sqrt().view(*shape) * x0 + (1 - alpha_bar).sqrt().view(*shape) * eps

    return xt, eps


def timestep_embedding(t, dim=128):
    if dim < 1:
        raise ValueError("dim must be positive")
    half_dim = dim // 2 # 64

    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=t.device)
        * -(math.log(10000.0) / max(half_dim, 1))
    ).to(t.device) # 64

    args = t.float().unsqueeze(1) * freqs # t, 1 x 64 -> t, 64

    f1 = torch.cos(args) # t, 64
    f2 = torch.sin(args) # t, 64

    e = torch.stack((f1, f2), dim=2).flatten(1) # t, 128

    return F.pad(e, (0, dim % 2))


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, C, H*W]
        x = x.reshape(B, C, H * W)

        probs = F.softmax(x, dim=-1)

        # coordinates from -1 to 1
        pos_x = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        pos_y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)

        grid_y, grid_x = torch.meshgrid(
            pos_y, pos_x, indexing="ij"
        )

        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        expected_x = torch.sum(probs * grid_x, dim=-1)
        expected_y = torch.sum(probs * grid_y, dim=-1)

        # [B, C] + [B, C] -> [B, C, 2]
        coords = torch.stack([expected_x, expected_y], dim=-1)

        # [B, C, 2] -> [B, C*2]
        return coords.flatten(1)


class DiT(nn.Module):
    def __init__(self, action_dim, hidden_dim, horizon, state_dim, num_heads=4, n_obs_steps=1):
        super().__init__()
        if not isinstance(n_obs_steps, int) or isinstance(n_obs_steps, bool) or n_obs_steps < 1:
            raise ValueError("n_obs_steps must be a positive integer")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim

        self.action_emb = nn.Linear(action_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        resnet = resnet18(
            weights=None,
            norm_layer=lambda channels: nn.GroupNorm(
                num_groups=8,
                num_channels=channels
            )
        ) 
        self.image_emb = nn.Sequential(
            *list(resnet.children())[:-2],
            SpatialSoftmax(),
            nn.Linear(1024, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim)

        # adaLN-Zero starts each residual block as the identity and predicts zero noise.
        for layer in (self.adaLN[-1], self.final_adaLN[-1], self.output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Append history projections after the existing modules so their seeded
        # initialization stays unchanged. Identity preserves old checkpoints.
        self.state_history = nn.Linear(n_obs_steps * hidden_dim, hidden_dim) if n_obs_steps > 1 else nn.Identity()
        self.image_history = nn.Linear(n_obs_steps * hidden_dim, hidden_dim) if n_obs_steps > 1 else nn.Identity()


    def forward(self, x, state, t, img):
        """Predict noise for actions [B, T, action_dim], conditioned on state and RGB images."""
        state_emb, img_emb = self.encode_observation(state, img)
        return self.forward_with_observation(x, t, state_emb, img_emb)

    def encode_observation(self, state, img):
        """Cache these features once per action chunk during diffusion sampling."""
        if self.n_obs_steps == 1:
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0]
            if img.ndim == 5 and img.shape[1] == 1:
                img = img[:, 0]
            return self.state_emb(state), self.image_emb(img)
        if state.ndim != 3 or state.shape[1:] != (self.n_obs_steps, self.state_dim):
            raise ValueError("state must have shape [batch, n_obs_steps, state_dim]")
        if img.ndim != 5 or img.shape[:3] != (state.shape[0], self.n_obs_steps, 3):
            raise ValueError("img must have shape [batch, n_obs_steps, 3, height, width]")
        batch = state.shape[0]
        state_features = self.state_emb(state).flatten(1)
        image_features = self.image_emb(img.flatten(0, 1)).reshape(batch, -1)
        # Concatenation preserves oldest-to-newest order instead of averaging.
        return self.state_history(state_features), self.image_history(image_features)

    def forward_with_observation(self, x, t, state_emb, img_emb):
        """Denoise using cached observation features; parameters match forward()."""
        if x.ndim != 3 or not 1 <= x.shape[1] <= self.pos_emb.shape[1]:
            raise ValueError("x must have shape [batch, sequence_length, action_dim] within horizon")
        x = self.action_emb(x)
        x = x + self.pos_emb[:, :x.shape[1]]

        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_emb(t_emb.to(dtype=x.dtype))

        c = state_emb + t_emb + img_emb

        params = self.adaLN(c)

        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp
        ) = params.chunk(6, dim=-1)
        
        x_norm = self.norm1(x)

        x_mod = (1 + scale_attn.unsqueeze(1)) * x_norm + shift_attn.unsqueeze(1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out

        x_mod = (1 + scale_mlp.unsqueeze(1)) * self.norm2(x) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = (1 + scale.unsqueeze(1)) * self.final_norm(x) + shift.unsqueeze(1)
        
        return self.output(x)


def train_step(model, optimizer, x0, state, img):
    """Run one noise-prediction training step on a batch of clean actions."""
    model.train()

    batch = x0.shape[0]
    t = torch.randint(low=0, high=len(betas), size=(batch,), device=x0.device)

    x_t, eps = q_sample(x0, t)
    eps_pred = model(x_t, state, t, img)

    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()
