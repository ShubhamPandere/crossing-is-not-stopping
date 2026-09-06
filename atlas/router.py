"""
router.py — Routing algorithms: select the best chart for a given chunk.

Routers implemented (E1 compares all of them):
  umf       — argmin UMF over the library (ours, C1)
  e1        — one-step prediction error only (ablation of UMF: uses only z₁)
  sdyn      — dynamics fingerprint: cosine similarity of action-conditioned latent delta
  random    — uniform random chart selection (analytic mean SR(c) baseline)
  oracle_id — given the true regime label, returns the matching chart (upper bound)

All routers return a chart index. Hysteresis: the current chart is kept unless
a competitor beats it by > margin m (default 0.05, applied to UMF/e1/sdyn).
"""

from __future__ import annotations

import random as _random
from typing import Literal

import torch

from atlas.library import Library
from atlas.score import umf as compute_umf

RouterKind = Literal["umf", "e1", "sdyn", "random", "oracle_id"]


def route(
    kind: RouterKind,
    library: Library,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    current_idx: int,
    motion_gate: float | None = None,
    hysteresis: float = 0.05,
    *,
    regime_label: int | None = None,   # required for oracle_id
    label_to_chart: dict[int, int] | None = None,  # required for oracle_id
    proprio_ctxt: torch.Tensor | None = None,
    rng: _random.Random | None = None,  # for the "random" router; see _route_random
) -> tuple[int, dict]:
    """
    Select a chart index from *library*.

    Args:
        kind:           Which router to use.
        library:        The current chart library.
        world_model:    EncPredWM instance (the object torch.hub.load returns —
                        NOT .model). Predictor reached via world_model.model.predictor.
        encoder_output: Pre-encoded chunk [T+1, N, D].
        actions:        Executed actions [T, action_dim].
        current_idx:    Index of the currently active chart (for hysteresis).
        motion_gate:    Informative-chunk threshold (None = skip gate).
        hysteresis:     Keep current chart unless competitor beats by this margin.
        regime_label:   True regime ID (oracle only).
        label_to_chart: Map regime_id -> chart_idx (oracle only).
        proprio_ctxt:   Encoded first-frame proprio [1, 1, P_tok, D] — see
                        score.umf()'s proprio_ctxt docstring (required in
                        practice for this checkpoint).
        rng:            random.Random instance for the "random" router — thread
                        the episode seed here for reproducibility (the router
                        previously used the unseeded global `random` module).
                        Falls back to that global module if None.

    Returns:
        (selected_idx, info_dict)
        info_dict keys: 'scores' (list of UMF values or None), 'gated' (bool)
    """
    if kind == "oracle_id":
        return _route_oracle(library, regime_label, label_to_chart)

    if kind == "random":
        return _route_random(library, rng)

    scores = _score_all(kind, library, world_model, encoder_output, actions, motion_gate, proprio_ctxt)

    # If all scores are None the chunk is uninformative — keep current chart.
    valid = {i: s for i, s in enumerate(scores) if s is not None}
    if not valid:
        return current_idx, {"scores": scores, "gated": True}

    best_idx = min(valid, key=valid.__getitem__)
    best_score = valid[best_idx]
    relative_gap: float | None = None
    hysteresis_held = False

    # Hysteresis: only switch if improvement exceeds margin, normalised by the
    # INCUMBENT'S OWN score (FIX_SPEC.md A1) -- the earlier spread normaliser
    # (spread = max(valid) - min(valid)) is inert at K=2: with exactly two
    # charts an incumbent that is not the argmin *is* the max, so
    # relative_gap = (max-min)/(max-min) = 1.0, always > m=0.05, and the router
    # is pure argmin. Normalising by current_score keeps the scale-invariance
    # the spread version was reaching for (umf ~O(1), e1 ~1e4, sdyn in [-1,1])
    # without being degenerate at K=2: m=0.05 now means "switch only if the
    # improvement is at least 5% of the incumbent's own score". m's value
    # (0.05, a CLAUDE.md Sec1.7 non-negotiable) is unchanged -- only the
    # normaliser.
    current_score = valid.get(current_idx)
    if current_score is not None:
        relative_gap = (current_score - best_score) / current_score if current_score > 0 else 0.0
        if relative_gap < hysteresis:
            hysteresis_held = True
            return current_idx, {"scores": scores, "gated": False,
                                 "relative_gap": relative_gap, "hysteresis_held": True}

    return best_idx, {"scores": scores, "gated": False,
                      "relative_gap": relative_gap, "hysteresis_held": hysteresis_held}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _score_all(
    kind: RouterKind,
    library: Library,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    motion_gate: float | None,
    proprio_ctxt: torch.Tensor | None = None,
) -> list[float | None]:
    """Return a score (lower = better) for each chart, or None if gated."""
    scores: list[float | None] = []
    for chart in library:
        if kind == "umf":
            s = compute_umf(chart, world_model, encoder_output, actions, motion_gate,
                             proprio_ctxt=proprio_ctxt)
        elif kind == "e1":
            s = _e1_score(chart, world_model, encoder_output, actions, motion_gate, proprio_ctxt)
        elif kind == "sdyn":
            s = _sdyn_score(chart, world_model, encoder_output, actions, motion_gate, proprio_ctxt)
        else:
            raise ValueError(f"Unknown router kind: {kind!r}")
        scores.append(s)
    return scores


@torch.no_grad()
def _e1_score(chart, world_model, encoder_output, actions, motion_gate, proprio_ctxt=None) -> float | None:
    """
    One-step prediction error: ‖ẑ₁ − z₁‖² (ablation of UMF: no normalisation,
    scores only the first predicted step).

    Runs the FULL open-loop rollout over *actions* (not a 1-action slice) and
    only scores z_hat[0] — VideoWM.encode_act() reshapes actions into chunks of
    `frameskip` raw actions (e.g. 5 raw 2-dim actions -> one 10-dim model
    action), so a single raw action cannot be encoded on its own; the caller's
    full *actions* array is always frameskip-divisible (same array `umf()`
    scores), so this reuses that guarantee instead of re-deriving frameskip.
    proprio_ctxt: see score.umf()'s docstring — required in practice for this
    checkpoint.
    """
    from atlas.score import _open_loop_rollout, _make_z_ctxt

    z = encoder_output
    z0, z1 = z[0], z[1]
    observed_disp = (z[-1] - z[0]).norm(p="fro").item()
    if motion_gate is not None and observed_disp <= motion_gate:
        return None
    predictor = world_model.model.predictor
    z_ctxt = _make_z_ctxt(world_model, z0, proprio_ctxt)
    chart.apply_(predictor)
    try:
        z_hat = _open_loop_rollout(world_model, z_ctxt, actions)
    finally:
        chart.restore_(predictor)
    return (z_hat[0] - z1).pow(2).sum().item()


@torch.no_grad()
def _sdyn_score(chart, world_model, encoder_output, actions, motion_gate, proprio_ctxt=None) -> float | None:
    """
    Dynamics fingerprint routing (S-dyn baseline).
    Score = negative cosine similarity between observed Δz and predicted Δz,
    using the first predicted step of the full open-loop rollout (see
    _e1_score's docstring for why a 1-action slice can't be encoded directly).
    Lower = better (matches UMF convention). proprio_ctxt: see
    score.umf()'s docstring — required in practice for this checkpoint.
    """
    from atlas.score import _open_loop_rollout, _make_z_ctxt

    z = encoder_output
    observed_disp = (z[-1] - z[0]).norm(p="fro").item()
    if motion_gate is not None and observed_disp <= motion_gate:
        return None
    z0, z1_obs = z[0], z[1]
    predictor = world_model.model.predictor
    z_ctxt = _make_z_ctxt(world_model, z0, proprio_ctxt)
    chart.apply_(predictor)
    try:
        z_hat = _open_loop_rollout(world_model, z_ctxt, actions)
    finally:
        chart.restore_(predictor)
    z1_hat = z_hat[0]
    delta_obs = (z1_obs - z0).flatten()
    delta_pred = (z1_hat - z0).flatten()
    cos_sim = torch.nn.functional.cosine_similarity(delta_obs, delta_pred, dim=0).item()
    return -cos_sim  # negate: lower score = more similar = better


def _route_random(library: Library, rng: _random.Random | None = None) -> tuple[int, dict]:
    source = rng if rng is not None else _random
    idx = source.randrange(len(library))
    return idx, {"scores": [None] * len(library), "gated": False}


def _route_oracle(
    library: Library,
    regime_label: int | None,
    label_to_chart: dict[int, int] | None,
) -> tuple[int, dict]:
    if regime_label is None or label_to_chart is None:
        raise ValueError(
            "oracle_id router requires regime_label and label_to_chart to be provided."
        )
    if regime_label not in label_to_chart:
        raise KeyError(
            f"Regime label {regime_label} not in label_to_chart map: {list(label_to_chart.keys())}. "
            "Ensure E0 has been run and charts have been assigned to regimes."
        )
    idx = label_to_chart[regime_label]
    if idx >= len(library):
        raise IndexError(
            f"Oracle chart index {idx} for regime {regime_label} exceeds library size {len(library)}."
        )
    return idx, {"scores": [None] * len(library), "gated": False}
