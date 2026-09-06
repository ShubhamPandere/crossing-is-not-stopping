"""
loop.py — atlas_step(): the ATLAS prequential controller (C1 + C2).

Called once per replan, BEFORE planning. The upstream planning code calls:

    chart_idx = atlas_step(state, library, expander, chunk, cfg)
    world_model.apply_chart(library[chart_idx])
    plan = cem_planner(world_model, ...)
    world_model.restore_chart(library.c0)   # restore baseline
    world_model.apply_chart(library[chart_idx])
    [execute 5 actions, collect new chunk]
    atlas_refine(library[chart_idx], predictor, new_chunk, cfg)

Step order (strict — never re-order):
  1. SCORE    UMF(c; Q) for all c
  2. SELECT   c* = argmin UMF, with hysteresis margin m
  3. EXPAND   if strikes ≥ q, fire probe on NEXT chunk (see expand.py)
  4. (caller) EXECUTE plan under c*
  5. REFINE   1 SGD step on c*      ← atlas_refine(), called by caller AFTER step 4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from atlas.library import Library
from atlas.router import route, RouterKind
from atlas.expand import Expander, ExpansionConfig, ProbeOutcome, _fit_candidate
from atlas.score import umf as compute_umf


@dataclass
class ATLASConfig:
    router: RouterKind = "umf"
    tau: float = 0.5
    q: int = 3
    hysteresis: float = 0.05
    lr: float = 5e-4
    n_probe: int = 20
    motion_gate: float | None = None   # computed from training data; set before use
    k_max: int = 10

    # For oracle router only:
    label_to_chart: dict[int, int] | None = None

    # Expansion mode: 'atlas' | 'detect_only' | 'fixed' | 'none'
    expansion_mode: Literal["atlas", "detect_only", "fixed", "none"] = "atlas"


@dataclass
class StepInfo:
    selected_idx: int
    scores: list[float | None]
    gated: bool
    probe_outcome: ProbeOutcome
    strikes: int


def atlas_step(
    library: Library,
    expander: Expander,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    current_idx: int,
    cfg: ATLASConfig,
    *,
    regime_label: int | None = None,
    # next_chunk is only needed for full ATLAS expansion; pass None otherwise
    next_encoder_output: torch.Tensor | None = None,
    next_actions: torch.Tensor | None = None,
    proprio_ctxt: torch.Tensor | None = None,
    next_proprio_ctxt: torch.Tensor | None = None,
    rng=None,
) -> StepInfo:
    """
    Execute one ATLAS decision step (SCORE + SELECT + EXPAND).
    Refinement (REFINE) is NOT done here — call atlas_refine() after planning.

    Args:
        library:              Current chart library.
        expander:             Stateful expander (strike counter).
        world_model:          EncPredWM instance (the object torch.hub.load
                              returns — NOT .model). Passed through to
                              route()/expander.maybe_expand(), both of which
                              need the wrapper (see E0_IMPLEMENTATION_PLAN.md T1/T2).
        encoder_output:       Current chunk [T+1, N, D].
        actions:              Executed actions [T, action_dim].
        current_idx:          Currently active chart index.
        cfg:                  ATLAS hyperparameters.
        regime_label:         True regime label (oracle router only).
        next_encoder_output:  Next chunk for expansion verification (ATLAS mode).
        next_actions:         Next chunk actions (ATLAS mode).
        proprio_ctxt:         Encoded first-frame proprio for `encoder_output`
                              [1, 1, P_tok, D] — see score.umf()'s docstring
                              (required in practice for this checkpoint).
        next_proprio_ctxt:    Encoded first-frame proprio for
                              `next_encoder_output` (ATLAS mode only).
        rng:                  random.Random instance for the "random" router
                              — thread the episode seed here for
                              reproducibility. Falls back to the unseeded
                              global `random` module if None.

    Returns:
        StepInfo with selected chart index and diagnostics.
    """
    # ── 1. SCORE + 2. SELECT ─────────────────────────────────────────────────
    selected_idx, route_info = route(
        kind=cfg.router,
        library=library,
        world_model=world_model,
        encoder_output=encoder_output,
        actions=actions,
        current_idx=current_idx,
        motion_gate=cfg.motion_gate,
        hysteresis=cfg.hysteresis,
        regime_label=regime_label,
        label_to_chart=cfg.label_to_chart,
        proprio_ctxt=proprio_ctxt,
        rng=rng,
    )

    scores = route_info["scores"]
    gated = route_info["gated"]

    # ── Best UMF for strike counter ───────────────────────────────────────────
    best_umf: float | None = None
    if not gated and scores:
        valid = [s for s in scores if s is not None]
        if valid:
            best_umf = min(valid)

    # ── 3. EXPAND ─────────────────────────────────────────────────────────────
    probe_outcome: ProbeOutcome = "not_ready"

    if cfg.expansion_mode == "atlas":
        expander.record(best_umf, encoder_output, actions, proprio_ctxt)
        if (
            next_encoder_output is not None
            and next_actions is not None
            and expander._strikes >= cfg.q
        ):
            probe_outcome = expander.maybe_expand(
                library, world_model, next_encoder_output, next_actions, cfg.motion_gate,
                next_proprio_ctxt,
            )

    elif cfg.expansion_mode == "detect_only":
        # Detect-and-spawn: commit immediately when strikes ≥ q, no verification.
        expander.record(best_umf, encoder_output, actions, proprio_ctxt)
        if expander._strikes >= cfg.q and not library.is_full():
            best_idx = selected_idx
            new_chart = library.clone_from(best_idx)
            # FIX_SPEC.md B6: previously committed here with NO gradient
            # step -- a byte-identical clone of its parent, which ties the
            # parent on UMF (identical weights => identical scores) and so
            # never wins argmin, and is not what "detect-and-spawn" (as
            # opposed to "detect-and-do-nothing") is supposed to mean: the
            # comparison against 'atlas' (full verification) should differ
            # by EXACTLY "verifies", not also by "never actually adapts".
            # Fit the candidate on the deficit chunks (same call
            # maybe_expand() makes for the 'atlas' mode) before committing
            # it -- still with NO held-out verification, which remains the
            # defining difference from expansion_mode='atlas'.
            if expander._deficit_chunks:
                _fit_candidate(new_chart, world_model, expander._deficit_chunks,
                                cfg.n_probe, cfg.lr)
            library.add(new_chart)
            selected_idx = len(library) - 1
            expander._strikes = 0
            expander._deficit_chunks.clear()
            expander._n_committed += 1
            expander._n_probes_fired += 1
            probe_outcome = "committed"

    # 'fixed' and 'none' modes: no expansion logic.

    return StepInfo(
        selected_idx=selected_idx,
        scores=scores,
        gated=gated,
        probe_outcome=probe_outcome,
        strikes=expander._strikes,
    )


def atlas_refine(
    chart,
    world_model,
    encoder_output: torch.Tensor,
    actions: torch.Tensor,
    lr: float = 5e-4,
    *,
    proprio_ctxt: torch.Tensor | None = None,
    optimizer=None,
) -> float:
    """
    One AdaJEPA gradient step on the selected chart.
    Must be called AFTER scoring and planning (step 5 in the prequential order).

    Rolls out via _open_loop_rollout() -- the same EncPredWM.unroll()-based
    function score.umf() and harness.run_e0_finetune() use, rather than the
    bare predictor(z_cur, a_t) call this used to make, which is not a valid
    call signature for this ViTPredictor (see E0_DIAGNOSIS_AND_PLAN.md).

    Args:
        chart:          The selected chart (c*).
        world_model:    EncPredWM instance (the object torch.hub.load
                        returns — NOT .model). Predictor reached via
                        world_model.model.predictor.
        encoder_output: Current chunk [T+1, N, D].
        actions:        Executed actions [T, action_dim].
        lr:             Learning rate. Ignored if `optimizer` is given.
        proprio_ctxt:   Encoded first-frame proprio [1, 1, P_tok, D] — see
                        score.umf()'s docstring (required in practice for
                        this checkpoint).
        optimizer:      Pre-built Adam optimizer over this chart's params, so
                        the CALLER (E3/E4's ArmState, one Adam per chart
                        index) can retain moment state across replans instead
                        of a fresh Adam being built (and its momentum
                        discarded) on every call -- the same setup AdaJEPA
                        uses. If None, a fresh Adam is built (back-compat for
                        callers with no persistent-optimizer state).

    Returns:
        Scalar loss value for logging.
    """
    import torch.optim as optim
    from atlas.score import _open_loop_rollout, _make_z_ctxt

    predictor = world_model.model.predictor
    chart.apply_(predictor)
    # FIX_SPEC.md C6: for kind="lora4", apply_() replaces the base weight's
    # entry in named_parameters() with a parametrization -- the base name
    # itself (in chart._param_names) no longer appears there, so the naive
    # "n in chart._param_names" filter (correct for ln_act/full) silently
    # selects ZERO parameters for lora4, and optim.Adam([]) raises
    # ValueError("optimizer got an empty parameter list"). Select by name
    # suffix instead, matching harness.py::run_e0_finetune's already-correct
    # offline pattern (harness.py:117-125).
    if chart.kind == "lora4":
        params = [p for n, p in predictor.named_parameters()
                   if "lora_A" in n or "lora_B" in n]
    else:
        params = [p for n, p in predictor.named_parameters() if n in chart._param_names]
    # FIX_SPEC.md B4: scripts/run_e4.py freezes ALL wm.encoder/wm.predictor
    # params (requires_grad_(False)) once, up front, before any chart is
    # ever applied -- but NOT wm.action_encoder/wm.proprio_encoder, which
    # stay requires_grad=True (confirmed empirically). Because
    # _open_loop_rollout's forward pass threads action/proprio embeddings
    # through those still-trainable modules, loss.backward() below does
    # NOT raise even when every chart-selected param has requires_grad=
    # False (contra FIX_SPEC.md's literal "loss.backward() raises"
    # wording) -- it succeeds silently while leaving every actual chart
    # parameter's .grad as None, so Adam's step is a total no-op for them:
    # the chart is NEVER refined, with no error anywhere (verified via
    # scratchpad/assert_b4.py: ln_act chart weights bit-identical before/
    # after atlas_refine() on unfixed code; lora4 unaffected, since its
    # injected lora_A/lora_B are freshly-constructed nn.Parameter objects
    # with torch's default requires_grad=True). Re-enable requires_grad on
    # exactly this chart's own selected parameter surface, every call, so
    # each arm/chart is self-contained regardless of run order or which
    # arms ran before it in this process (run_e4.py runs one arm per fresh
    # subprocess, so there is no "AdaJEPA arm happened to run first"
    # workaround available).
    for p in params:
        p.requires_grad_(True)
    if optimizer is None:
        optimizer = optim.Adam(params, lr=lr)

    optimizer.zero_grad()
    z_ctxt = _make_z_ctxt(world_model, encoder_output[0], proprio_ctxt)
    z_preds = _open_loop_rollout(world_model, z_ctxt, actions)  # [T, N, D]
    loss = (z_preds - encoder_output[1:]).pow(2).mean(dim=-1).mean()
    loss.backward()
    optimizer.step()

    scalar_loss = loss.item()

    # [WandB Logging] Log to active WandB run if initialized
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "loop/probe_refine_loss": scalar_loss,
                "kind": chart.kind,
            })
    except ImportError:
        pass

    chart.update_from_predictor_(predictor)
    # NOT a restore to pretrained weights -- chart._params now holds the
    # just-refined values (update_from_predictor_ above), so this re-applies
    # THIS chart's own (updated) weights, leaving the predictor holding the
    # refined chart, not the frozen baseline. The previous comment here
    # ("restore predictor to chart's baseline weights") was false -- see
    # FIX_SPEC.md C4 / Chart.restore_() and the new Chart.restore_pretrained_()
    # for a method that genuinely restores pretrained weights.
    chart.restore_(predictor)
    return scalar_loss


def atlas_refine_buffered(
    chart,
    world_model,
    buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    lr: float = 5e-4,
    *,
    optimizer=None,
) -> float:
    """
    FIX_SPEC.md B9 (ADDITIVE — atlas_refine() itself is UNCHANGED above,
    so every existing single-chunk caller, e.g. E1/E2/the gates, keeps its
    current behaviour exactly): refine *chart* over a BUFFER of up to
    AdaJEPA.BUFFER_SIZE=5 recent (encoder_output, actions, proprio_ctxt)
    chunks, one gradient step summing per-item losses -- the SAME buffer
    size and per-item-backward-then-single-step pattern
    atlas.adajepa.AdaJEPA.refine() uses. plan §7.6 requires arms 2/3
    (AdaJEPA) and arms 4/5/6 (ATLAS-family) see the same buffer size so the
    3->4 ladder rung differs by "routes/expands", not also by "sees 5x
    less refinement data per step". Callers (harness_e4.py) own the buffer
    itself (one per chart index, incl. the c0-adapted clone — B8), same as
    ArmState.optimizers already owns one Adam per chart index.

    Args:
        chart:      The chart to refine (library chart or a c0 clone).
        world_model: EncPredWM instance.
        buffer:     Up to 5 (encoder_output, actions, proprio_ctxt) tuples,
                    oldest first — caller-maintained sliding window.
        lr:         Learning rate. Ignored if `optimizer` is given.
        optimizer:  Pre-built Adam optimizer over this chart's params (see
                    atlas_refine()'s identical parameter for why).

    Returns:
        Mean scalar loss across the buffer, for logging.
    """
    import torch.optim as optim
    from atlas.score import _open_loop_rollout, _make_z_ctxt

    if not buffer:
        return 0.0

    predictor = world_model.model.predictor
    chart.apply_(predictor)
    # Same C6 (lora4 suffix filter) + B4 (requires_grad re-enable) fixes as
    # atlas_refine() above — see its comments for the full rationale.
    if chart.kind == "lora4":
        params = [p for n, p in predictor.named_parameters()
                   if "lora_A" in n or "lora_B" in n]
    else:
        params = [p for n, p in predictor.named_parameters() if n in chart._param_names]
    for p in params:
        p.requires_grad_(True)
    if optimizer is None:
        optimizer = optim.Adam(params, lr=lr)

    optimizer.zero_grad()
    total = 0.0
    for encoder_output, actions, proprio_ctxt in buffer:
        z_ctxt = _make_z_ctxt(world_model, encoder_output[0], proprio_ctxt)
        z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
        loss = (z_preds - encoder_output[1:]).pow(2).mean(dim=-1).mean()
        (loss / len(buffer)).backward()   # per-item backward = O(1) memory,
        total += loss.item()              # same pattern as AdaJEPA.refine()
    optimizer.step()
    avg_loss = total / len(buffer)

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"loop/probe_refine_loss_buffered": avg_loss, "kind": chart.kind})
    except ImportError:
        pass

    chart.update_from_predictor_(predictor)
    chart.restore_(predictor)
    return avg_loss
