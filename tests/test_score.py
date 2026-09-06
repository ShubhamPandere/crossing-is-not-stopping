"""
tests/test_score.py — Unit tests for atlas.score (UMF computation).
"""

import torch
import pytest
from unittest.mock import MagicMock

from atlas.score import umf, compute_motion_gate


def make_dummy_world_model(grid_size):
    """Mock of the EncPredWM WRAPPER (the object torch.hub.load returns — NOT
    .model): action_dim=10 is the real 10-dim model-chunk contract
    (frameskip=5 * raw_action_dim=2), and .unroll() is the wrapper's own
    canonical rollout entry point (see atlas.score._open_loop_rollout /
    E0_IMPLEMENTATION_PLAN.md T1)."""
    wm = MagicMock()
    wm.grid_size = grid_size
    wm.action_dim = 10
    wm.model = MagicMock()
    wm.model.predictor = MagicMock()

    def _unroll(z_ctxt, act_suffix):
        # z_ctxt: [B=1, tau=1, V=1, H, W, D]; act_suffix: [T, B=1, A].
        # Identity unroll: repeat the single context frame T+1 times, matching
        # unroll()'s documented [T+tau, B, V, H, W, D] output contract.
        T = act_suffix.shape[0]
        frame = z_ctxt[:, 0]  # [B, V, H, W, D] -- drop tau
        return frame.unsqueeze(0).expand(T + 1, *frame.shape).clone()

    wm.unroll.side_effect = _unroll
    return wm

def make_dummy_chart():
    chart = MagicMock()
    chart.apply_ = MagicMock()
    chart.restore_ = MagicMock()
    chart._param_names = []
    return chart


def test_umf_static_chunk_returns_none():
    """Static chunk (no motion) -> denominator 0 -> UMF returns None."""
    grid, D, T = 4, 16, 3
    N = grid * grid
    z0 = torch.randn(N, D)
    encoder_output = z0.unsqueeze(0).expand(T + 1, -1, -1).clone()
    actions = torch.zeros(T, 10)  # model-chunk actions: 10 = frameskip(5) * raw_dim(2)

    chart = make_dummy_chart()
    wm = make_dummy_world_model(grid)

    result = umf(chart, wm, encoder_output, actions, motion_gate=None)
    assert result is None, f"Expected None for static chunk, got {result}"


def test_umf_gated_chunk_returns_none():
    """Chunk below motion_gate -> returns None."""
    grid, D, T = 4, 16, 3
    N = grid * grid
    z0 = torch.randn(N, D)
    z_end = z0 + 0.001  # tiny displacement
    encoder_output = torch.stack([z0] + [z0] * (T - 1) + [z_end])
    actions = torch.zeros(T, 10)  # model-chunk actions

    chart = make_dummy_chart()
    wm = make_dummy_world_model(grid)

    # Gate is larger than the displacement -> should be gated.
    result = umf(chart, wm, encoder_output, actions, motion_gate=1.0)
    assert result is None


def test_umf_shape_validation():
    """Wrong encoder_output shape raises ValueError."""
    chart = make_dummy_chart()
    wm = make_dummy_world_model(4)
    with pytest.raises(ValueError, match="encoder_output must be"):
        umf(chart, wm, torch.randn(5, 16), torch.zeros(3, 2))


def test_umf_frame_count_mismatch():
    """T+1 frame count mismatch raises ValueError."""
    chart = make_dummy_chart()
    wm = make_dummy_world_model(4)
    with pytest.raises(ValueError, match="frames"):
        umf(chart, wm, torch.randn(5, 10, 16), torch.zeros(3, 2))


def test_compute_motion_gate_empty():
    """Empty tensor raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_motion_gate(torch.tensor([]))


def test_compute_motion_gate_percentile():
    """10th percentile of [0..9] is 0.9."""
    displacements = torch.arange(10, dtype=torch.float32)
    gate = compute_motion_gate(displacements, percentile=10.0)
    assert abs(gate - 0.9) < 0.1  # approximate due to linear interpolation
