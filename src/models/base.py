"""Abstract base class and core implementations for plant surrogate neural models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePlantSurrogate(nn.Module, ABC):
    """Abstract base class for neural plant telemetry surrogate models."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting the target scalar sensor value from companion sequences.

        Args:
            x: Input tensor of shape [batch_size, seq_len, in_channels].

        Returns:
            Tensor of shape [batch_size, 1].
        """

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step causal prediction wrapper.

        Args:
            x: Input tensor of shape [batch_size, seq_len, in_channels].

        Returns:
            Tensor of shape [batch_size, 1].
        """
        return self.forward(x)

    @abstractmethod
    def get_receptive_field(self) -> int:
        """Return the effective causal receptive field (timesteps) of the architecture."""


class CausalConv1dBlock(nn.Module):
    """Residual 1D convolutional block with strictly causal left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        # Left-pad to preserve strict causality
        x1 = F.pad(x, (self.pad, 0))
        out = self.drop1(F.silu(self.norm1(self.conv1(x1))))
        x2 = F.pad(out, (self.pad, 0))
        out = self.drop2(self.norm2(self.conv2(x2)))
        return F.silu(out + res)


class CausalTCN(BasePlantSurrogate):
    """Causal Dilated Temporal Convolutional Network surrogate model.

    Compatible with existing champion checkpoints from results/surrogates/.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers: list[nn.Module] = []
        curr_in = in_channels
        for d in dilations:
            layers.append(
                CausalConv1dBlock(
                    in_channels=curr_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            curr_in = hidden_channels

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch, seq_len, in_channels]
        # Transpose to [batch, in_channels, seq_len] for Conv1d
        if x.dim() == 3 and x.shape[1] != self.in_channels:
            x = x.transpose(1, 2)
        h = self.network(x)
        # Representation at final position in context window
        h_last = h[:, :, -1]
        return self.head(h_last)

    def get_receptive_field(self) -> int:
        """Calculate exact theoretical receptive field."""
        # Each block has 2 conv layers with padding (K - 1) * d
        return 1 + 2 * sum((self.kernel_size - 1) * d for d in self.dilations)
