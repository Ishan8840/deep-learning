"""Action Chunking with Transformers (ACT).

Read ACT.forward first, then follow its helper methods:
    demonstrated actions -> latent sample (training only)
    camera images        -> visual tokens
    latent + joints + visual tokens -> encoder -> decoder -> action chunk

Attention lives inside PyTorch's TransformerEncoderLayer and
TransformerDecoderLayer below. This file defines how ACT connects them.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, dtype):
    """Sinusoidal positions for an ordered sequence of tokens."""
    step = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype)
        * (-math.log(10000) / width)
    )
    result = torch.zeros(length, width, device=device, dtype=dtype)
    result[:, 0::2] = (step * frequency).sin()
    result[:, 1::2] = (step * frequency).cos()
    return result


class ACT(nn.Module):
    """Predict a chunk of actions from camera images and current joint positions.

    images: (B, cameras, 3, H, W), floating point in [0, 1]
    qpos: (B, state_dim), normalized using training-set statistics
    actions: (B, chunk_size, action_dim), normalized targets, optional
    is_pad: (B, chunk_size), True for padded targets

    Training infers z from the demonstrated actions. Inference uses z=0.
    A small CNN keeps this learning implementation runnable on a CPU.
    """

    def __init__(self, state_dim=14, action_dim=14, chunk_size=50,
                 num_cameras=1, hidden_dim=128, latent_dim=32,
                 nheads=4, encoder_layers=2, decoder_layers=2, dropout=0.1):
        super().__init__()
        dimensions = (state_dim, action_dim, chunk_size, num_cameras, hidden_dim,
                      latent_dim, nheads, encoder_layers, decoder_layers)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if hidden_dim % nheads or hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by nheads and 4")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras
        self.latent_dim = latent_dim
        # Image backbone: reduce each camera image to a grid of features.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
        # Project joints/actions into the same token dimension as image features.
        self.joint_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.camera_embed = nn.Parameter(torch.randn(num_cameras, hidden_dim) * 0.02)
        self.context_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)

        # Each encoder layer contains self-attention followed by a feedforward
        # network, with residual connections and layer normalization.
        def make_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            return nn.TransformerEncoder(layer, encoder_layers, enable_nested_tensor=False)

        # Training-only encoder: summarize [CLS, joints, demonstrated actions].
        self.posterior = make_encoder()
        self.to_distribution = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_embed = nn.Linear(latent_dim, hidden_dim)
        # Observation encoder: let all context and image tokens attend together.
        self.encoder = make_encoder()
        # Decoder: self-attention between action queries, then cross-attention
        # to the observation encoder's output. All actions are predicted at once.
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, qpos, images, actions=None, is_pad=None):
        """Return (predicted_actions, latent_mean, latent_log_variance).

        Supplying actions selects the training path. Without actions, the latent
        is zero and its distribution parameters are returned as None.
        """
        batch_size = qpos.shape[0]
        if qpos.shape != (batch_size, self.state_dim):
            raise ValueError("qpos must have shape (B, state_dim)")
        if images.ndim != 5 or images.shape[:3] != (batch_size, self.num_cameras, 3):
            raise ValueError("images must have shape (B, num_cameras, 3, H, W)")

        # 1. Represent the current robot state as one token: (B, hidden_dim).
        joint_token = self.joint_embed(qpos)

        # 2. During training, infer the latent from the demonstrated action chunk.
        #    At inference, no future actions are available, so use the prior mean.
        if actions is not None:
            latent, mu, logvar = self._encode_actions(joint_token, actions, is_pad)
        else:
            latent = qpos.new_zeros(batch_size, self.latent_dim)
            mu = logvar = None

        # 3. Turn camera images into a sequence of spatially positioned tokens.
        visual_tokens = self._encode_images(images)

        # 4. Self-attention mixes [latent, joints, image tokens] into memory.
        latent_token = self.latent_embed(latent)
        context_tokens = torch.stack((latent_token, joint_token), dim=1)
        context_tokens = context_tokens + self.context_embed
        observation_tokens = torch.cat((context_tokens, visual_tokens), dim=1)
        memory = self.encoder(observation_tokens)

        # 5. One learned query per future timestep reads from that memory.
        #    No causal mask: the entire action chunk is predicted in parallel.
        action_queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded_tokens = self.decoder(action_queries, memory)
        predicted_actions = self.action_head(decoded_tokens)
        return predicted_actions, mu, logvar

    def _encode_actions(self, joint_token, actions, is_pad):
        """Infer q(z | joints, actions) and sample z with reparameterization."""
        batch_size = joint_token.shape[0]
        if actions.shape != (batch_size, self.chunk_size, self.action_dim):
            raise ValueError("actions must have shape (B, chunk_size, action_dim)")
        if is_pad is None:
            is_pad = torch.zeros(actions.shape[:2], device=actions.device, dtype=torch.bool)
        if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
            raise ValueError("is_pad must be a boolean (B, chunk_size) tensor")

        # Clear padding before projection, including any NaNs in padded entries.
        valid_actions = actions.masked_fill(is_pad.unsqueeze(-1), 0)
        action_tokens = self.action_embed(valid_actions)
        summary_token = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat((summary_token, joint_token.unsqueeze(1), action_tokens), dim=1)
        sequence_length, hidden_dim = tokens.shape[1:]
        tokens = tokens + positions(sequence_length, hidden_dim, tokens.device, tokens.dtype)

        # The summary and joint tokens are always valid; only actions are padded.
        padding_mask = F.pad(is_pad, (2, 0), value=False)
        encoded_tokens = self.posterior(tokens, src_key_padding_mask=padding_mask)
        summary = encoded_tokens[:, 0]
        mu, logvar = self.to_distribution(summary).chunk(2, dim=-1)

        # z = mean + standard_deviation * noise lets gradients reach both heads.
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mu)
        latent = mu + standard_deviation * noise
        return latent, mu, logvar

    def _encode_images(self, images):
        """Return (B, cameras * grid_height * grid_width, hidden_dim) tokens."""
        batch_size = images.shape[0]

        # Run the same CNN on every camera by combining batch and camera axes.
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


def act_loss(predicted, actions, mu, logvar, is_pad=None, kl_weight=10.0):
    """L1 averaged over valid action entries, plus summed latent KL per sample."""
    if is_pad is None:
        is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
    valid = ~is_pad
    if not valid.any():
        raise ValueError("A batch must contain at least one valid target action")
    l1 = F.l1_loss(predicted[valid], actions[valid])
    # KL(q(z) || N(0, I)): sum over latent dimensions, average over the batch.
    kl_per_dimension = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl_per_dimension.sum(-1).mean()
    total_loss = l1 + kl_weight * kl
    return {"loss": total_loss, "l1": l1, "kl": kl}


class TemporalEnsemble:
    """Combine overlapping predictions; positive decay favors older predictions.

    Call once per control step with (chunk_size, action_dim) predictions.
    Construct a new instance or reset between episodes. Zero actions are valid.
    """

    def __init__(self, decay=0.01):
        if decay < 0:
            raise ValueError("decay must be nonnegative")
        self.decay = decay
        self.reset()

    def reset(self):
        self.chunks = []

    @torch.no_grad()
    def __call__(self, chunk):
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ValueError("chunk must have shape (positive chunk_size, action_dim)")
        self.chunks.append(chunk.detach())
        # Every stored chunk starts at the current control step. Older chunks
        # have already had their executed actions removed on previous calls.
        current_predictions = torch.stack([pending[0] for pending in self.chunks])
        prediction_order = torch.arange(
            len(current_predictions),
            device=current_predictions.device,
            dtype=current_predictions.dtype,
        )
        weights = torch.exp(-self.decay * prediction_order)
        weights = weights / weights.sum()
        action = (current_predictions * weights[:, None]).sum(0)

        # Advance all predictions one step and drop exhausted chunks.
        self.chunks = [pending[1:] for pending in self.chunks if len(pending) > 1]
        return action
