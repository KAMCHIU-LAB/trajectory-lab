"""Independent, compact implementations of public model architectures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim: int, output_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, output_dim),
    )


class GaussianPolicy(nn.Module):
    """Gaussian likelihood with a bounded mean; evaluation uses that mean.

    BC and IQL share this exact actor. The Gaussian has state-independent scale;
    it is not a tanh-transformed distribution. No stochastic clipped-Gaussian
    likelihood or Jacobian approximation is used.
    """

    def __init__(self, observation_dim: int, action_dim: int, width: int):
        super().__init__()
        self.network = mlp(observation_dim, action_dim, width)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observations))

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        distribution = torch.distributions.Normal(
            self(observations), self.log_std.clamp(-5, 2).exp()
        )
        return distribution.log_prob(actions).sum(-1)


class TwinQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, width: int):
        super().__init__()
        self.q1 = mlp(observation_dim + action_dim, 1, width)
        self.q2 = mlp(observation_dim + action_dim, 1, width)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple:
        inputs = torch.cat([observations, actions], -1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("Transformer width must be divisible by attention heads")
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        # True means allowed for scaled_dot_product_attention (unlike MultiheadAttention).
        allowed = causal[None, None] & token_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.projection(attended)
        return x + self.feedforward(self.norm2(x))


class DecisionTransformer(nn.Module):
    """Return/state/action tokenization with strictly causal action prediction.

    A compact DT-style baseline, not a claim to reproduce the paper's tuning.
    Right padding ensures each valid query has at least one valid causal key.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_timestep: int = 8192,
    ):
        super().__init__()
        self.max_timestep = max_timestep
        self.return_embedding = nn.Linear(1, width)
        self.state_embedding = nn.Linear(observation_dim, width)
        self.action_embedding = nn.Linear(action_dim, width)
        self.time_embedding = nn.Embedding(max_timestep, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.action_head = nn.Linear(width, action_dim)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time_embedding(timesteps)
        tokens = torch.stack(
            [
                self.return_embedding(returns.unsqueeze(-1)) + time,
                self.state_embedding(observations) + time,
                self.action_embedding(actions) + time,
            ],
            dim=2,
        )
        batch, length, _, width = tokens.shape
        x = self.input_norm(tokens.reshape(batch, 3 * length, width))
        token_mask = mask.repeat_interleave(3, dim=1)
        for block in self.blocks:
            x = block(x, token_mask)
        # State token t precedes action token t; its output cannot read the label.
        states = self.output_norm(x.reshape(batch, length, 3, width)[:, :, 1])
        return torch.tanh(self.action_head(states))
