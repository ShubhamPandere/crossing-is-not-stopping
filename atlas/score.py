"""
score.py — UMF (Unexplained Motion Fraction) and informative-chunk gating.

C3: UMF(c; Q) = Σ_k ‖ẑ^c_k − z_k‖² / Σ_k ‖z_k − z₀‖²

  Numerator:   open-loop latent prediction error under chart c.
  Denominator: total latent motion in chunk Q relative to z₀ (first frame).
  Result:      0 = perfect predictor; ≈1 = no better than predicting stasis.

Prequential invariant (enforced here, not just documented):
  score() is a PURE FORWARD PASS with no_grad. Refinement happens AFTER scoring.

Informative-chunk gate:
  A chunk is informative iff its observed displacement exceeds a threshold
  computed from the training set (10th percentile of training displacement).
  Non-informative chunks return None; callers must not count them as strikes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.chart import Chart


@torch.no_grad()
def umf(
    chart: "Chart",
    world_model,
    encoder_output: torch.Tensor,   # [T_model+1, N_patches, D] — already encoded, one frame per frameskip
    actions: torch.Tensor,          # [T_model, wm.action_dim] — MODEL-chunk actions (10-dim: frameskip*raw_dim)
    motion_gate: float | None = None,
    proprio_ctxt: torch.Tensor | None = None,
) -> float | None:
    """
    Compute UMF for *chart* on chunk Q = (encoder_output, actions).

    Args:
        chart:           The chart to evaluate.
        world_model:     EncPredWM instance (the object torch.hub.load returns — NOT
                         .model). Owns ctxt_window/proprio_mode and is the canonical
                         unroll this checkpoint was validated with. The chart
                         parameters are swapped into world_model.model.predictor
                         (ViTPredictor).
        encoder_output:  Pre-computed visual latent states z_0 … z_{T_model}, sampled
                         every `frameskip` raw frames, shape [T_model+1, N, D].
        actions:         Executed MODEL-chunk actions a_0 … a_{T_model-1}, shape
                         [T_model, 10] (10 = frameskip * raw_action_dim). NOT raw
                         per-env-step actions — see _open_loop_rollout's guard.
        motion_gate:     If not None, return None when observed displacement
                         ||z_T - z_0||_F <= motion_gate (uninformative chunk).
        proprio_ctxt:    Encoded proprio for the FIRST frame, shape [1, 1, P_tok, D]
                         — exactly EncPredWM.encode()'s "proprio" output for a
                         single-frame observation (T=1), no reshaping needed.
                         NOT optional in practice for this checkpoint —
                         empirically, dino_wm_pusht's predictor was trained with
                         proprio concatenated into the token channel width
                         (VideoWM.concat_obs_act(), dim=3), so forward_pred with
                         proprio=None produces a channel-width mismatch, not a
                         graceful proprio-free path. None is accepted only for
                         forward compatibility with a checkpoint that has no
                         proprio_encoder at all.

    Returns:
        UMF value in [0, ∞), or None if chunk is uninformative.

    Raises:
        ValueError: if tensor shapes are inconsistent.
    """
    if encoder_output.ndim != 3:
        raise ValueError(
            f"encoder_output must be [T+1, N_patches, D], got {encoder_output.shape}"
        )
    T = actions.shape[0]
    if encoder_output.shape[0] != T + 1:
        raise ValueError(
            f"encoder_output has {encoder_output.shape[0]} frames but {T} actions imply {T+1} frames"
        )

    z = encoder_output  # [T+1, N, D]
    z0 = z[0]          # [N, D] — first frame = null predictor output

    # ── Informative-chunk gate ────────────────────────────────────────────────
    observed_displacement = (z[-1] - z0).norm(p="fro").item()
    if motion_gate is not None and observed_displacement <= motion_gate:
        return None  # uninformative chunk — caller must not count as a strike

    # ── Open-loop unroll under chart ──────────────────────────────────────────
    # chart.apply_/restore_ swap params on ViTPredictor (world_model.model.predictor).
    predictor = world_model.model.predictor
    z_ctxt = _make_z_ctxt(world_model, z0, proprio_ctxt)
    chart.apply_(predictor)
    try:
        z_hat = _open_loop_rollout(world_model, z_ctxt, actions)   # [T, N, D]
    finally:
        chart.restore_(predictor)

    # ── UMF numerator: Σ_k ||ẑ_k − z_k||² ───────────────────────────────────
    z_targets = z[1:]  # [T, N, D]
    numerator   = (z_hat - z_targets).pow(2).sum().item()

    # ── UMF denominator: Σ_k ||z_k − z_0||² ─────────────────────────────────
    displacement = (z_targets - z0.unsqueeze(0)).pow(2).sum().item()

    if displacement == 0.0:
        # Denominator is exactly 0 → chunk is static; return None (gate should
        # have caught this, but be defensive).
        return None

    return numerator / displacement


@torch.no_grad()
def rollout_umf(
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    proprio_ctxt: torch.Tensor | None = None,
    motion_gate: float | None = None,
) -> float | None:
    """
    Same computation as umf(), for a predictor that is ALREADY in the state to
    be scored — no chart apply_/restore_ here. umf()'s own apply/restore would
    undo a chart a caller needs to stay applied for the rest of an episode
    (e.g. E0 planning's post-hoc per-replan UMF logging, where the chart was
    applied once at episode start and CEM keeps planning against it).
    """
    if encoder_output.ndim != 3:
        raise ValueError(
            f"encoder_output must be [T+1, N_patches, D], got {encoder_output.shape}"
        )
    T = actions.shape[0]
    if encoder_output.shape[0] != T + 1:
        raise ValueError(
            f"encoder_output has {encoder_output.shape[0]} frames but {T} actions imply {T+1} frames"
        )

    z = encoder_output
    z0 = z[0]

    observed_displacement = (z[-1] - z0).norm(p="fro").item()
    if motion_gate is not None and observed_displacement <= motion_gate:
        return None

    z_ctxt = _make_z_ctxt(world_model, z0, proprio_ctxt)
    z_hat = _open_loop_rollout(world_model, z_ctxt, actions)

    z_targets = z[1:]
    numerator = (z_hat - z_targets).pow(2).sum().item()
    displacement = (z_targets - z0.unsqueeze(0)).pow(2).sum().item()
    if displacement == 0.0:
        return None

    return numerator / displacement


def _make_z_ctxt(enc_pred_wm, z0: torch.Tensor, proprio_ctxt: torch.Tensor | None):
    """Build the z_ctxt _open_loop_rollout expects: a TensorDict with real
    proprio when available (required for dino_wm_pusht — see umf()'s
    proprio_ctxt docstring), else a bare visual Tensor."""
    if proprio_ctxt is None:
        return z0
    from tensordict import TensorDict

    grid = enc_pred_wm.grid_size
    D = z0.shape[-1]
    visual = z0.reshape(1, 1, 1, grid, grid, D)
    return TensorDict({"visual": visual, "proprio": proprio_ctxt}, batch_size=[])


def _open_loop_rollout(enc_pred_wm, z_ctxt, actions: torch.Tensor) -> torch.Tensor:
    """
    Roll out the predictor open-loop via EncPredWM.unroll() — the checkpoint's own
    canonical unroll (sliding ctxt_window, real action chunking, real proprio
    propagation when z_ctxt carries a "proprio" key), rather than hand-driving
    VideoWM.forward_pred() with a 1-frame context and a fabricated zero proprio.
    GC_Agent plans through this exact function (gc_agent.py:175).

    Args:
        enc_pred_wm: EncPredWM — the object torch.hub.load(...) returns, NOT
                     the inner .model. Owns ctxt_window/proprio_mode/grid_size.
        z_ctxt:      Context latent features for the FIRST frame of the chunk
                     (tau=1). Either:
                       - a TensorDict/dict with "visual" [1, 1, V, H, W, D] and
                         "proprio" [1, 1, P_tok, D] (once real proprio is
                         threaded through — see T3), or
                       - a bare visual Tensor, either already [1, 1, V, H, W, D]
                         or flat [N, D] (auto-reshaped using grid_size). The
                         Tensor form has no proprio key at all — EncPredWM.unroll
                         then omits proprio from forward_pred rather than
                         substituting a fabricated zero.
        actions:     [T_model, 10] MODEL-chunk actions (normalized). 10 =
                     frameskip * raw_action_dim — NOT raw per-env-step actions.

    Returns:
        Predicted visual latent states ẑ_1 … ẑ_{T_model}, shape [T_model, N_patches, D].

    Raises:
        ValueError: if actions' last dim doesn't match the checkpoint's model
                    action_dim — the guard for the raw-vs-model time-base bug
                    this function replaces (see E0_DIAGNOSIS_AND_PLAN.md).
    """
    from tensordict import TensorDict

    model_action_dim = enc_pred_wm.action_dim  # 10 = frameskip(5) * raw_action_dim(2)
    if actions.shape[-1] != model_action_dim:
        raise ValueError(
            f"_open_loop_rollout got actions with last dim {actions.shape[-1]}, but "
            f"this checkpoint's model action_dim is {model_action_dim} (= frameskip * "
            f"raw_action_dim). This looks like RAW per-env-step actions, not "
            f"model-chunk actions — chunk them to [T_raw // frameskip, {model_action_dim}] "
            f"before calling (see E0_IMPLEMENTATION_PLAN.md T3)."
        )

    grid = enc_pred_wm.grid_size

    if isinstance(z_ctxt, torch.Tensor) and z_ctxt.ndim == 2:
        # Flat [N, D] first-frame visual latent -> [B=1, tau=1, V=1, H, W, D]
        D = z_ctxt.shape[-1]
        z_ctxt = z_ctxt.reshape(1, 1, 1, grid, grid, D)

    T = actions.shape[0]
    act_suffix = actions.unsqueeze(1)  # [T, B=1, A]

    out = enc_pred_wm.unroll(z_ctxt, act_suffix=act_suffix)

    vid = out["visual"] if isinstance(out, (TensorDict, dict)) else out
    # vid: [T + tau, B, V, H, W, D]; tau=1 context frame prepended -> drop it.
    vid = vid[1:].squeeze(1)          # [T, V, H, W, D]
    T_out, V, H, W, D = vid.shape
    return vid.reshape(T_out, V * H * W, D)   # [T, N, D]


def compute_motion_gate(training_displacements: torch.Tensor, percentile: float = 10.0) -> float:
    """
    Compute the informative-chunk gate threshold from training data.

    Args:
        training_displacements: 1-D tensor of per-chunk Frobenius displacements
                                collected from the training dataset.
        percentile:             Gate chunks below this percentile (default 10th).

    Returns:
        Scalar threshold; chunks with displacement ≤ this value return UMF=None.
    """
    if training_displacements.numel() == 0:
        raise ValueError("training_displacements is empty; cannot compute motion gate.")
    return float(torch.quantile(training_displacements.float(), percentile / 100.0).item())
