"""Locked unit tests for BasePlantSurrogate, CausalTCN, and CausalPatchTST.

Verifies:
1. Abstract base class contract for BasePlantSurrogate.
2. Tensor input/output shapes for CausalTCN and CausalPatchTST.
3. Strict causal invariance (future perturbations do not alter past representations).
4. Receptive field calculations.
5. Compatibility with existing champion TCN checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.models.base import BasePlantSurrogate, CausalTCN
from src.models.patch_tst import CausalPatchTST


def test_base_surrogate_is_abstract() -> None:
    """Verify BasePlantSurrogate cannot be instantiated directly without implementations."""
    with pytest.raises(TypeError):
        BasePlantSurrogate()  # type: ignore[abstract]


def test_causal_tcn_shapes_and_receptive_field() -> None:
    """Verify CausalTCN tensor shapes, forward execution, and receptive field calculation."""
    in_channels = 8
    seq_len = 48
    batch_size = 4

    model = CausalTCN(
        in_channels=in_channels,
        hidden_channels=32,
        dilations=(1, 2, 4, 8),
        kernel_size=3,
        dropout=0.0,
    )
    model.eval()

    rf = model.get_receptive_field()
    assert isinstance(rf, int)
    assert rf > 0
    # Receptive field formula: 1 + 2 * sum((K - 1) * d) = 1 + 2 * 2 * (1 + 2 + 4 + 8) = 61
    assert rf == 61

    x = torch.randn(batch_size, seq_len, in_channels)
    out = model(x)
    assert out.shape == (batch_size, 1)

    step_out = model.predict_step(x)
    assert torch.allclose(out, step_out)


def test_causal_patch_tst_shapes_and_receptive_field() -> None:
    """Verify CausalPatchTST patching, multi-head self-attention, and output dimensions."""
    in_channels = 6
    seq_len = 48
    batch_size = 4
    patch_len = 16
    stride = 8

    model = CausalPatchTST(
        in_channels=in_channels,
        d_model=32,
        patch_len=patch_len,
        stride=stride,
        nhead=4,
        num_layers=2,
        dropout=0.0,
        max_patches=16,
    )
    model.eval()

    rf = model.get_receptive_field()
    assert isinstance(rf, int)
    assert rf > 0

    x = torch.randn(batch_size, seq_len, in_channels)
    out = model(x)
    assert out.shape == (batch_size, 1)

    step_out = model.predict_step(x)
    assert torch.allclose(out, step_out)


def test_causal_tcn_strict_causal_invariance() -> None:
    """Verify perturbations at future timesteps do not affect past convolutional activations."""
    in_channels = 4
    seq_len = 48
    model = CausalTCN(
        in_channels=in_channels,
        hidden_channels=16,
        dilations=(1, 2, 4),
        kernel_size=3,
        dropout=0.0,
    )
    model.eval()

    # Original sequence
    x_orig = torch.randn(1, seq_len, in_channels)
    # Perturbed sequence: perturb timesteps 30 to 47
    x_pert = x_orig.clone()
    x_pert[:, 30:, :] += 10.0

    # Hook causal conv layer activations across time
    activations: list[torch.Tensor] = []

    def hook_fn(
        module: torch.nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        activations.append(out.detach().clone())

    handle = model.network[0].conv1.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(x_orig)
        model(x_pert)
    handle.remove()

    act_orig, act_pert = activations[0], activations[1]
    # At any past timestep t <= 29, activations must be mathematically identical
    # In Conv1d, format is [batch, channels, seq_len]
    assert torch.allclose(act_orig[:, :, :30], act_pert[:, :, :30], atol=1e-6)


def test_causal_patch_tst_strict_causal_invariance() -> None:
    """Verify causal attention mask prevents future patches from influencing earlier patches."""
    in_channels = 4
    seq_len = 48
    patch_len = 16
    stride = 8

    model = CausalPatchTST(
        in_channels=in_channels,
        d_model=32,
        patch_len=patch_len,
        stride=stride,
        nhead=4,
        num_layers=2,
        dropout=0.0,
        max_patches=16,
    )
    model.eval()

    # Base sequence
    x_orig = torch.randn(1, seq_len, in_channels)
    # Perturb the last patch (e.g. timesteps 40 to 47)
    x_pert = x_orig.clone()
    x_pert[:, 40:, :] += 50.0

    with torch.no_grad():
        encoded_orig = model.encode_patches(
            x_orig
        )  # [batch, in_channels, num_patches, d_model]
        encoded_pert = model.encode_patches(x_pert)

    num_patches = encoded_orig.shape[2]
    assert num_patches >= 3

    # All patches strictly before the perturbed patch must remain identical
    assert torch.allclose(
        encoded_orig[:, :, : num_patches - 1, :],
        encoded_pert[:, :, : num_patches - 1, :],
        atol=1e-5,
    )


def test_champion_tcn_checkpoint_compatibility() -> None:
    """Verify CausalTCN can load weights from trained champion checkpoint results/surrogates/vista_tcn_48h_150ep/tcn.pt."""
    ckpt_path = Path("results/surrogates/vista_tcn_48h_150ep/tcn.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Champion checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = CausalTCN(
        in_channels=len(ckpt["companion_channels"]),
        hidden_channels=ckpt["hidden_channels"],
        dilations=tuple(ckpt["dilations"]),
        kernel_size=ckpt["kernel_size"],
        dropout=ckpt["dropout"],
    )

    # Load state dict
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Test forward pass with random batch
    x = torch.randn(2, ckpt["context_length"], len(ckpt["companion_channels"]))
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
