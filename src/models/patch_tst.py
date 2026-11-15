"""Causal Patch Time Series Transformer (PatchTST) for industrial SCADA telemetry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePlantSurrogate


class CausalPatchTST(BasePlantSurrogate):
    """Causal PatchTST surrogate model with strict lower-triangular attention mask.

    Performs subseries patching, linear token projection, causal multi-head
    self-attention (zero future lookahead), and cross-channel projection to a target scalar.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        patch_len: int = 16,
        stride: int = 8,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_patches: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_patches = max_patches

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, max_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.Linear(in_channels * d_model, 64),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(64, 1),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract subseries patches of shape [batch, in_channels, num_patches, patch_len]."""
        # Input shape: [batch, seq_len, in_channels] -> transpose to [batch, in_channels, seq_len]
        if x.dim() == 3 and x.shape[1] != self.in_channels:
            x = x.transpose(1, 2)

        batch, channels, seq_len = x.shape
        if seq_len < self.patch_len:
            pad = self.patch_len - seq_len
            x = F.pad(x, (pad, 0))
            seq_len = x.shape[-1]

        # Unfold along time dimension
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Encode subseries patches with causal self-attention.

        Returns:
            Tensor of shape [batch, in_channels, num_patches, d_model].
        """
        patches = self._extract_patches(x)
        batch, channels, num_patches, patch_len = patches.shape

        if num_patches > self.max_patches:
            raise ValueError(
                f"num_patches ({num_patches}) exceeds max_patches ({self.max_patches})"
            )

        # Linear projection to d_model
        h = self.patch_proj(patches)  # [batch, in_channels, num_patches, d_model]
        h = h + self.pos_embed[:, :, :num_patches, :]
        h = self.dropout(h)

        # Merge batch and channels for channel-independent self-attention
        h_flat = h.view(batch * channels, num_patches, self.d_model)

        # Strict lower-triangular causal attention mask: patch i attends only to patch j <= i
        causal_mask = torch.triu(
            torch.full((num_patches, num_patches), float("-inf"), device=x.device),
            diagonal=1,
        )

        encoded = self.transformer_encoder(h_flat, mask=causal_mask, is_causal=False)
        return encoded.view(batch, channels, num_patches, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_patches(x)
        # Extract representation at final patch position (latest temporal context)
        h_last = encoded[:, :, -1, :]  # [batch, in_channels, d_model]
        batch = h_last.shape[0]
        h_flat = h_last.reshape(batch, self.in_channels * self.d_model)
        return self.head(h_flat)

    def get_receptive_field(self) -> int:
        """Effective causal context span of the patch hierarchy."""
        return self.patch_len + (self.max_patches - 1) * self.stride
