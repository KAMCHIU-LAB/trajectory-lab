"""Frozen-encoder action heads for the registered three-arm feasibility study.

Inputs to the encoder are already normalized with the original checkpoint.
Each supplied history must be its deployment window of at most 24 ticks, with
the original absolute timesteps; this module never fits a new normalizer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_ACTIONS = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3])


class ActionHead(nn.Module):
    """Indexed I or interval-shared A/P monotone four-response decoder."""

    def __init__(self, arm: str, actions: np.ndarray):
        super().__init__()
        actions = np.asarray(actions, dtype=np.float64)
        if arm not in ("I", "A", "P") or not np.array_equal(actions, _ACTIONS):
            raise ValueError("Expected registered arm I/A/P and the frozen 11-action grid")
        self.arm = arm
        self.register_buffer("midpoints", torch.tensor((actions[:-1] + actions[1:]) / 6,
                                                      dtype=torch.float32))
        self.register_buffer("widths", torch.tensor(np.diff(actions) / 3, dtype=torch.float32))
        dimension, width, outputs = (141, 57, 40) if arm == "I" else (144, 64, 4)
        # A and P execute exactly the same initialization operations and RNG draws.
        self.hidden = nn.Sequential(nn.Linear(dimension, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.decoder = nn.Linear(width, outputs)
        with torch.no_grad():
            bias = self.decoder.bias.reshape(4, 10 if arm == "I" else 1)
            bias[:2].fill_(-5.0)
            bias[2:].fill_(-2.0)

    def interval_inputs(self, common: torch.Tensor, pmean: torch.Tensor) -> torch.Tensor:
        """Expose the shared common141/midpoint/width inputs for identity evidence."""
        if common.shape[-1] != 141 or pmean.shape != common.shape[:-1]:
            raise ValueError("Expected common [...,141] and matching original-unit pmean")
        shape = (*common.shape[:-1], 10, 1)
        midpoints = self.midpoints.to(dtype=common.dtype).expand(shape[:-1]).unsqueeze(-1)
        widths = self.widths.to(dtype=common.dtype).expand(shape[:-1]).unsqueeze(-1)
        slot = pmean[..., None, None] * midpoints if self.arm == "P" else torch.zeros_like(midpoints)
        return torch.cat((common.unsqueeze(-2).expand(*shape[:-1], 141),
                          midpoints, widths, slot), dim=-1)

    def forward(self, common: torch.Tensor, pmean: torch.Tensor) -> torch.Tensor:
        if common.shape[-1] != 141 or pmean.shape != common.shape[:-1]:
            raise ValueError("Expected common [...,141] and matching original-unit pmean")
        if self.arm == "I":
            raw = self.decoder(self.hidden(common)).reshape(*common.shape[:-1], 4, 10)
        else:
            raw = self.decoder(self.hidden(self.interval_inputs(common, pmean))).transpose(-1, -2)
        return F.pad(F.softplus(raw).cumsum(dim=-1), (1, 0), value=0.0)


@torch.no_grad()
def encoder_features(encoder: nn.Module, states: torch.Tensor, steps: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Return original norm-layer output concatenated with normalized state13."""
    if (states.ndim != 3 or states.shape[-1] != 13 or not 1 <= states.shape[1] <= 24
            or steps.shape != states.shape[:-1] or mask.shape != steps.shape
            or mask.dtype != torch.bool):
        raise ValueError("Expected normalized [batch,1..24,13] histories, timesteps and bool mask")
    encoder.requires_grad_(False)
    encoder.eval()
    hidden = encoder.embedding(states) + encoder.time(steps)
    for block in encoder.blocks:
        hidden = block(hidden, mask)
    hidden = encoder.norm(hidden)
    if hidden.shape[-1] != 128:
        raise ValueError("Expected the registered 128-dimensional encoder")
    return torch.cat((hidden, states), dim=-1)


class DecoderModel(nn.Module):
    """Native Learned-compatible response model with a permanently frozen encoder."""

    def __init__(self, encoder: nn.Module, head: ActionHead, mean, std):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clone())
        if (self.mean.shape != (13,) or self.std.shape != (13,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("Expected the frozen checkpoint's finite 13-column normalizer")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, states: torch.Tensor, steps: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        common = encoder_features(self.encoder, states, steps, mask)
        pmean = states[..., 9] * self.std[9] + self.mean[9]
        return self.head(common, pmean)
