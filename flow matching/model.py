import math

import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, dtype):
    # Sinusoidal positions
    step = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000) / width)
    )
    result = torch.zeros(length, width, device=device, dtype=dtype)
    result[:, 0::2] = (step * frequency).sin()
    result[:, 1::2] = (step * frequency).cos()
    return result

def timestep_embedding(t, dim=128):
    half = dim // 2

    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / half
    )

    args = t[:, None] * freqs[None, :]

    return torch.cat([
        torch.sin(args),
        torch.cos(args)
    ], dim=-1)


def flow_sample(x1, t):
    x0 = torch.randn_like(x1)

    t_view = t[:, None, None]

    x_t = (1 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0

    return x_t, target_velocity


class FM(nn.Module):
    def __init__(self, state_dim=14, action_dim=14, chunk_size=50,
                 num_cameras=1, hidden_dim=128,
                 nheads=4, encoder_layers=2, decoder_layers=2, dropout=0.1):
        super().__init__()

        dimensions = (state_dim, action_dim, chunk_size, num_cameras, hidden_dim,
                        nheads, encoder_layers, decoder_layers)

        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in dimensions):
            raise ValueError("Model dimensions must be positive integers")

        if hidden_dim % nheads or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by nheads and 4")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras

        # Image backbone: reduce each camera image to a grid of features.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
        )

        # Project joints/actions into the same token dimension as image features.
        self.joint_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.camera_embed = nn.Parameter(torch.randn(num_cameras, hidden_dim) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Each encoder layer contains self-attention followed by a feedforward
        def make_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            return nn.TransformerEncoder(layer, encoder_layers, enable_nested_tensor=False)

        # Observation encoder: let all context and image tokens attend together.
        self.encoder = make_encoder()

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, qpos, images, x_t, t):
        batch_size = qpos.shape[0]

        if qpos.shape != (batch_size, self.state_dim):
            raise ValueError("qpos must have shape (B, state_dim)")

        if images.ndim != 5 or images.shape[:3] != (batch_size, self.num_cameras, 3):
            raise ValueError("images must have shape (B, num_cameras, 3, H, W)")

        joint_token = self.joint_embed(qpos)
        visual_tokens = self._encode_images(images)
        t_emb = timestep_embedding(t, 128)
        t_emb = self.time_embed(t_emb)

        observation_tokens = torch.cat(
            [joint_token.unsqueeze(1), visual_tokens],
            dim=1
        )

        memory = self.encoder(observation_tokens)
        action_tokens = self.action_embed(x_t)
        action_tokens = action_tokens + positions(self.chunk_size, action_tokens.shape[-1], action_tokens.device, action_tokens.dtype).unsqueeze(0) + t_emb.unsqueeze(1)

        decoded_tokens = self.decoder(action_tokens, memory)
        velocity = self.action_head(decoded_tokens)

        return velocity


    def _encode_images(self, images):
        batch_size = images.shape[0]

        # Run the same CNN on every camera by combining batch and camera axes
        camera_images = images.flatten(0, 1)
        features = self.backbone(camera_images)
        _, hidden_dim, grid_height, grid_width = features.shape

        # Half the positional channels encode row, half encode column.
        row_positions = positions(grid_height, hidden_dim // 2, features.device, features.dtype)
        column_positions = positions(grid_width, hidden_dim // 2, features.device, features.dtype)

        row_grid = row_positions[:, None].expand(-1, grid_width, -1)
        column_grid = column_positions[None].expand(grid_height, -1, -1)

        spatial_positions = torch.cat((row_grid, column_grid), dim=-1)
        spatial_positions = spatial_positions.reshape(grid_height * grid_width, hidden_dim)

        # (B * cameras, D, H, W) -> (B, cameras, H * W, D).
        visual_tokens = features.flatten(2).transpose(1, 2)
        visual_tokens = visual_tokens.reshape(
            batch_size, self.num_cameras, grid_height * grid_width, hidden_dim,
        )
        # Identify both where a feature is and which camera it came from.
        camera_positions = self.camera_embed[None, :, None]
        visual_tokens = visual_tokens + spatial_positions + camera_positions

        return visual_tokens.flatten(1, 2)

    def fm_loss(self, model, x1, qpos, images):
        t = torch.rand(x1.shape[0], device=x1.device)
        
        x_t, target_velocity = flow_sample(x1, t)

        pred_velocity = model(qpos, images, x_t, t)

        return F.mse_loss(pred_velocity, target_velocity)
