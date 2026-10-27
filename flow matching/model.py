"""Visual flow matching for action sequences."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, dtype):
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
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / half
    )

    angles = t[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def flow_sample(x1, t):
    x0 = torch.randn_like(x1)
    time = t[:, None, None]
    return (1 - time) * x0 + time * x1, x1 - x0


def make_backbone(name, hidden_dim):
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
    if name == "resnet18":
        from torchvision.models import resnet18

        network = resnet18(weights=None, norm_layer=lambda width: nn.GroupNorm(32, width))
        return nn.Sequential(
            *list(network.children())[:-2],
            nn.Conv2d(512, hidden_dim, kernel_size=1),
        )
    raise ValueError("backbone must be cnn or resnet18")


class FM(nn.Module):
    def __init__(self, state_dim=14, action_dim=14, chunk_size=50,
                 num_cameras=1, hidden_dim=128,
                 nheads=4, encoder_layers=2, decoder_layers=2, dropout=0.1,
                 backbone="cnn", obs_steps=1):
        super().__init__()

        dimensions = (state_dim, action_dim, chunk_size, num_cameras, hidden_dim,
                        nheads, encoder_layers, decoder_layers, obs_steps)

        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in dimensions):
            raise ValueError("Model dimensions must be positive integers")

        if hidden_dim % nheads or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by nheads and 4")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras
        self.obs_steps = obs_steps
        self.backbone_name = backbone

        self.backbone = make_backbone(backbone, hidden_dim)
        self.joint_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.camera_embed = nn.Parameter(torch.randn(num_cameras, hidden_dim) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, encoder_layers, enable_nested_tensor=False,
        )

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
        memory = self.encode_observations(qpos, images)
        return self.predict_velocity(memory, x_t, t)

    def encode_observations(self, qpos, images):
        batch_size = qpos.shape[0]

        if self.obs_steps > 1:
            if qpos.shape != (batch_size, self.obs_steps, self.state_dim):
                raise ValueError("qpos must have shape (B, obs_steps, state_dim)")
            if images.ndim != 6 or images.shape[:4] != (batch_size, self.obs_steps, self.num_cameras, 3):
                raise ValueError("images must have shape (B, obs_steps, cameras, 3, H, W)")
            joint_tokens = self.joint_embed(qpos).unsqueeze(2)
            visual_tokens = self._encode_images(images.flatten(0, 1))
            visual_tokens = visual_tokens.reshape(batch_size, self.obs_steps, -1, joint_tokens.shape[-1])
            tokens = torch.cat([joint_tokens, visual_tokens], dim=2)
            temporal = positions(self.obs_steps, tokens.shape[-1], tokens.device, tokens.dtype)
            tokens = tokens + temporal[None, :, None, :]
            return self.encoder(tokens.flatten(1, 2))

        if qpos.shape != (batch_size, self.state_dim):
            raise ValueError("qpos must have shape (B, state_dim)")

        if images.ndim != 5 or images.shape[:3] != (batch_size, self.num_cameras, 3):
            raise ValueError("images must have shape (B, num_cameras, 3, H, W)")

        joint_token = self.joint_embed(qpos)
        visual_tokens = self._encode_images(images)

        observation_tokens = torch.cat([joint_token.unsqueeze(1), visual_tokens], dim=1)
        return self.encoder(observation_tokens)

    def predict_velocity(self, memory, x_t, t):
        if x_t.shape != (memory.shape[0], self.chunk_size, self.action_dim):
            raise ValueError("x_t must have shape (B, chunk_size, action_dim)")
        if t.shape != (memory.shape[0],):
            raise ValueError("t must have shape (B,)")
        time_tokens = self.time_embed(timestep_embedding(t, 128)).unsqueeze(1)
        action_tokens = self.action_embed(x_t)
        action_positions = positions(
            self.chunk_size, action_tokens.shape[-1], action_tokens.device, action_tokens.dtype,
        ).unsqueeze(0)
        action_tokens = action_tokens + action_positions + time_tokens
        return self.action_head(self.decoder(action_tokens, memory))

    def _encode_images(self, images):
        batch_size = images.shape[0]

        camera_images = images.flatten(0, 1)
        features = self.backbone(camera_images)
        _, hidden_dim, grid_height, grid_width = features.shape

        row_positions = positions(grid_height, hidden_dim // 2, features.device, features.dtype)
        column_positions = positions(grid_width, hidden_dim // 2, features.device, features.dtype)

        row_grid = row_positions[:, None].expand(-1, grid_width, -1)
        column_grid = column_positions[None].expand(grid_height, -1, -1)

        spatial_positions = torch.cat((row_grid, column_grid), dim=-1)
        spatial_positions = spatial_positions.reshape(grid_height * grid_width, hidden_dim)

        visual_tokens = features.flatten(2).transpose(1, 2)
        visual_tokens = visual_tokens.reshape(
            batch_size, self.num_cameras, grid_height * grid_width, hidden_dim,
        )
        camera_positions = self.camera_embed[None, :, None]
        visual_tokens = visual_tokens + spatial_positions + camera_positions

        return visual_tokens.flatten(1, 2)

    def fm_loss(self, x1, qpos, images, is_pad=None):
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x_t, target_velocity = flow_sample(x1, t)
        pred_velocity = self(qpos, images, x_t, t)
        return masked_mse(pred_velocity, target_velocity, is_pad)


def masked_mse(prediction, target, is_pad=None):
    """Average over valid action coordinates, excluding episode-end padding."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if is_pad is None:
        return F.mse_loss(prediction, target)
    if is_pad.shape != target.shape[:2] or is_pad.dtype != torch.bool:
        raise ValueError("is_pad must be boolean with shape (B, chunk_size)")
    if not (~is_pad).any(dim=1).all():
        raise ValueError("Every sample must contain a valid action")
    return F.mse_loss(prediction[~is_pad], target[~is_pad])
