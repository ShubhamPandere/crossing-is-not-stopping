"""
harness.py — Offline evaluation harness for E0, E1, E5.

E0 (adapter capacity):
  - Fine-tune chart kinds offline on regime trajectories.
  - Evaluate UMF and planning success in-regime.

E1 (fitness routing):
  - Charts fixed; 2 warmup replans under c₀, then route and plan.
  - Same seeds across all routers (G5 guaranteed by streams.paired_seed).

E5 (cross-policy):
  - M[i,j] = chart i's UMF on chunks from chart j's plans.

All harness functions write per-episode JSONL to the output directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from atlas.chart import Chart, ChartKind
from atlas.library import Library
from atlas.score import umf as compute_umf
from atlas.router import route, RouterKind
from atlas import LOGS_DIR


def compute_trajectory_loss(world_model, z_preds: torch.Tensor, z_targets: torch.Tensor) -> torch.Tensor:
    """Computes loss on latent predictions following VideoWM.compute_loss configuration."""
    import torch.nn.functional as F
    if getattr(world_model, "cfgs_loss", None):
        cfgs = world_model.cfgs_loss
        cos_w = cfgs.get("cos_loss_weight", 0.0)
        l1_w = cfgs.get("l1_loss_weight", 0.0)
        l2_w = cfgs.get("l2_loss_weight", 1.0)
        smooth_w = cfgs.get("smooth_l1_loss_weight", 0.0)
        
        loss = torch.tensor(0.0, device=z_preds.device)
        if l2_w > 0:
            loss = loss + l2_w * (z_preds - z_targets).pow(2).mean(dim=-1).mean()
        if l1_w > 0:
            loss = loss + l1_w * (z_preds - z_targets).abs().mean(dim=-1).mean()
        if cos_w > 0:
            cos_sim = F.cosine_similarity(z_preds, z_targets, dim=-1)
            loss = loss + cos_w * (-cos_sim).mean()
        if smooth_w > 0:
            loss = loss + smooth_w * F.smooth_l1_loss(z_preds, z_targets, reduction="mean")
        return loss
    else:
        return (z_preds - z_targets).pow(2).mean(dim=-1).mean()


# ── E0 — Adapter capacity ─────────────────────────────────────────────────────

def run_e0_finetune(
    world_model,
    trajectories: list[dict],   # list of {encoder_output, actions} dicts
    kind: ChartKind,
    regime: str,
    n_steps: int,
    lr: float,
    out_dir: Path,
    val_trajectories: list[dict] | None = None,
    eval_every: int = 25,
    patience: int = 5,
) -> Chart:
    """
    Offline fine-tune a chart of the given kind on provided trajectories.

    Args:
        world_model:  EncPredWM instance (the object torch.hub.load returns —
                      NOT .model). _open_loop_rollout needs the wrapper for its
                      canonical unroll(); chart apply/restore and cfgs_loss-based
                      loss still reach the inner VideoWM via world_model.model.
        trajectories: List of trajectory dicts with pre-encoded data.
        kind:         Chart kind to fine-tune.
        regime:       Regime name (for logging).
        n_steps:      Maximum number of gradient steps (upper bound -- early
                      stopping below may stop sooner).
        lr:           Learning rate.
        out_dir:      Directory to save the resulting chart and loss log.
        val_trajectories: Held-out trajectories for early stopping
                      (E0_IMPLEMENTATION_PLAN.md T9 -- the overfitting fix:
                      `full` previously reached train loss 0.0015 over 2000
                      steps on 30 transitions with no validation signal at
                      all). If None, behaves exactly as before: always runs
                      the full n_steps and returns the FINAL step's weights
                      (no regression for any existing caller that doesn't
                      pass this).
        eval_every:   Steps between validation checks.
        patience:     Stop after this many consecutive checks with no
                      improvement in validation loss. The chart returned is
                      the BEST-validation snapshot seen, not necessarily the
                      final step's weights -- training can (and does)
                      overfit past the optimum.

    Returns:
        The fine-tuned Chart.
    """
    import copy
    import torch.optim as optim
    from tqdm import tqdm
    from atlas.score import _open_loop_rollout, _make_z_ctxt

    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = world_model.model.predictor
    chart = Chart(predictor, kind)
    chart.apply_(predictor)

    # Freeze non-chart parameters and enable gradients ONLY on chart parameters
    for n, p in predictor.named_parameters():
        if kind == "lora4" and ("lora_A" in n or "lora_B" in n):
            p.requires_grad_(True)
        elif kind != "lora4" and n in chart._param_names:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    params = [p for n, p in predictor.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    def _val_loss() -> float:
        # Training mutates `predictor`'s live parameters in-place (not
        # chart._params), so predictor already reflects the current trained
        # state here -- no apply_/restore_ needed, just a plain no-grad
        # forward pass. update_from_predictor_ separately snapshots the
        # current weights INTO chart._params, which the caller below uses to
        # keep the best-so-far snapshot.
        chart.update_from_predictor_(predictor)
        losses = []
        with torch.no_grad():
            for traj in val_trajectories:
                enc_out = traj["encoder_output"]
                actions = traj["actions"]
                z_vis = enc_out[0]
                proprio_ctxt = traj.get("proprio")
                if proprio_ctxt is not None:
                    proprio_ctxt = proprio_ctxt[0:1].unsqueeze(0)
                z_ctxt = _make_z_ctxt(world_model, z_vis, proprio_ctxt)
                z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
                loss = compute_trajectory_loss(world_model.model, z_preds, enc_out[1:])
                losses.append(loss.item())
        return sum(losses) / len(losses)

    loss_log: list[float] = []
    val_loss_log: list[dict] = []
    best_val_loss = float("inf")
    best_train_loss: float | None = None  # train loss AT the best-val step -- the
                                          # value that actually describes the chart
                                          # returned/saved, not the final step's
                                          # (possibly more-overfit) train loss.
    best_params: dict[str, torch.Tensor] | None = None
    checks_since_improvement = 0
    stopped_early_at: int | None = None

    pbar = tqdm(range(n_steps), desc=f"{kind}_{regime}", unit="step")
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        # P2a fix (E0_RECOVERY_PLAN.md): backward() per-trajectory instead of
        # summing all trajectories' loss tensors and calling backward() once
        # -- the old version kept every trajectory's autograd graph alive
        # simultaneously, so peak memory scaled O(N trajectories). This is
        # why `lora4` OOM'd at the same 20x25 budget `ln_act` used and had to
        # be retrained at a smaller, confounding 10x15. Gradients are
        # mathematically identical (sum of per-trajectory grads either way);
        # peak memory becomes O(1 trajectory).
        for traj in trajectories:
            enc_out: torch.Tensor = traj["encoder_output"]  # [T+1, N, D]
            actions: torch.Tensor = traj["actions"]         # [T, action_dim]
            z_vis = enc_out[0]
            proprio_ctxt = traj.get("proprio")
            if proprio_ctxt is not None:
                proprio_ctxt = proprio_ctxt[0:1].unsqueeze(0)  # [1, 1, P_tok, D_p]
            z_ctxt = _make_z_ctxt(world_model, z_vis, proprio_ctxt)

            # Predict visual latents open-loop
            z_preds = _open_loop_rollout(world_model, z_ctxt, actions) # [T, N, D]
            loss = compute_trajectory_loss(world_model.model, z_preds, enc_out[1:])
            (loss / len(trajectories)).backward()
            total_loss += loss.item()  # detached scalar, for logging only

        optimizer.step()
        avg_loss = total_loss / len(trajectories)
        loss_log.append(avg_loss)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

        # [WandB Logging] Log step and loss to active WandB run if initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "step": step + 1,
                    "loss": avg_loss,
                    f"loss_{kind}_{regime}": avg_loss,
                    "kind": kind,
                    "regime": regime,
                })
        except ImportError:
            pass

        # [Debug print statement] Print progress every 100 steps and on step 1.
        # pbar.write (not print) so it doesn't corrupt the tqdm bar's line.
        if (step + 1) == 1 or (step + 1) % 100 == 0 or (step + 1) == n_steps:
            pbar.write(f"    [Debug] [{kind}_{regime}] Step {step+1}/{n_steps} - Loss: {avg_loss:.6f}")

        if val_trajectories and ((step + 1) % eval_every == 0 or (step + 1) == n_steps):
            val_loss = _val_loss()
            val_loss_log.append({"step": step + 1, "val_loss": val_loss})
            pbar.write(f"    [Debug] [{kind}_{regime}] Step {step+1}/{n_steps} - Val Loss: {val_loss:.6f}"
                       f"{' (best)' if val_loss < best_val_loss else ''}")

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"val_loss_{kind}_{regime}": val_loss, "step": step + 1})
            except ImportError:
                pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = avg_loss
                best_params = copy.deepcopy(chart._params)
                checks_since_improvement = 0
            else:
                checks_since_improvement += 1
                if checks_since_improvement >= patience:
                    stopped_early_at = step + 1
                    pbar.write(f"    [Debug] [{kind}_{regime}] Early stopping at step {step+1} "
                               f"-- no val improvement for {patience} checks (best val loss "
                               f"{best_val_loss:.6f})")
                    break

    if val_trajectories and best_params is not None:
        # Keep the BEST validation snapshot, not the final (possibly
        # overfit) step's weights.
        chart._params = best_params
    else:
        chart.update_from_predictor_(predictor)
    chart.restore_(predictor)

    chart_path = out_dir / f"chart_{kind}_{regime}.pt"
    chart.save(chart_path)
    (out_dir / f"loss_{kind}_{regime}.json").write_text(json.dumps(loss_log))
    if val_trajectories:
        (out_dir / f"val_loss_{kind}_{regime}.json").write_text(json.dumps({
            "val_loss_log": val_loss_log,
            "best_val_loss": best_val_loss,
            "best_train_loss": best_train_loss,  # train loss AT the checkpoint used
            "stopped_early_at_step": stopped_early_at,
        }))
    return chart


# ── E1 — Fitness routing ──────────────────────────────────────────────────────

def _make_obs_td(visual_hw3_uint8, proprio_vec, device: str):
    """Build a batchless-per-field TensorDict; GC_Agent.set_goal()/.act() add
    their own leading batch dim internally (see scripts/run_e0_planning.py)."""
    import numpy as np
    from tensordict import TensorDict

    visual = torch.from_numpy(visual_hw3_uint8.copy()).permute(2, 0, 1).float().unsqueeze(0)
    proprio = torch.from_numpy(np.asarray(proprio_vec, dtype=np.float32)).unsqueeze(0)
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[]).to(device)


def _prepare_env(base_env, regime, seed: int, state) -> tuple[dict, dict]:
    """
    Reset to a controlled state with physics reapplied.

    Uses regime.reset() (PhysicsRegime.reset(), which calls _apply_physics()
    automatically) rather than PushTWrapper's reset()/step() — PushTWrapper
    discards obs["visual"] (it expects a separate PixelWrapper to re-render),
    which E1 needs for encoding. reset_to_state is set on base_env directly
    (not on the regime wrapper) since gym.Wrapper does not proxy attribute
    writes to the wrapped env.
    """
    base_env.seed(seed)
    base_env.reset_to_state = state
    return regime.reset()


def run_e1_episode(
    library: Library,
    agent,                      # GC_Agent, pre-configured with E1's CEM hyperparameters
    world_model,                # EncPredWM wrapper: .model.predictor, .encode_obs, .unroll
    base_env,                   # raw PushTEnv
    regime,                     # PhysicsRegime(base_env, ...) — physics fixed for the whole episode
    goal_utils,                 # PushTWrapper(base_env) — only .sample_random_init_goal_states/.eval_state used
    router: RouterKind,
    episode_seed: int,
    n_warmup_replans: int,
    n_replans_target: int,
    frameskip: int,
    num_act_stepped: int,       # model-chunk units -- matches agent's CEMPlanner.horizon units
    motion_gate: float | None,
    hysteresis: float,
    out_dir: Path,
    episode_id: str,
    *,
    regime_label: int | None = None,
    label_to_chart: dict[int, int] | None = None,
) -> dict[str, Any]:
    """
    Run one E1 episode with a specified router.

    Prequential order per replan (never reorder — see loop.py's module docstring
    for the same invariant in the full ATLAS loop): SCORE the chunk executed by
    the PREVIOUS replan -> SELECT a chart -> EXECUTE this replan with it. The
    first `n_warmup_replans` replans skip scoring/selection entirely and stay on
    library[0] (c0) — this is deliberate: E1 tests whether the router can find
    the right chart using only c0's warmup data, not whether c0 itself is a good
    fit. Charts are frozen throughout (no atlas_refine call) — E1's charts come
    fixed from E0 ("Charts from E0, fixed" per the proposal); this also means
    E1 never touches atlas.loop.atlas_step()/atlas_refine() at all.

    Regime is FIXED for the whole episode (set via `regime` before this is
    called) — E1 does not switch regimes mid-episode; that's E3/E4's stream.
    """
    from atlas.regimes import assert_no_bypassed_corruption
    assert_no_bypassed_corruption(regime)  # FIX_SPEC.md C5 -- see docstring

    device = agent.device
    predictor = world_model.model.predictor
    # Seeds the "random" router for this episode -- was the unseeded global
    # `random` module (E0_IMPLEMENTATION_PLAN.md T12 #6).
    import random as _random
    router_rng = _random.Random(episode_seed)

    init_state, goal_state = goal_utils.sample_random_init_goal_states(episode_seed)

    goal_obs, _ = _prepare_env(base_env, regime, episode_seed, goal_state)
    agent.set_goal(_make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))

    obs, _ = _prepare_env(base_env, regime, episode_seed, init_state)

    current_idx = 0  # c0, always the starting chart
    elapsed = 0
    success = False
    selected_trace: list[int] = []
    umf_trace: list[list[float | None]] = []
    raw_steps_per_replan: list[int] = []
    # (encoder_output [T_model+1,N,D], actions [T_model,10], proprio_ctxt [1,1,P_tok,D])
    prev_chunk: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    for replan_idx in range(n_replans_target):
        if replan_idx >= n_warmup_replans and prev_chunk is not None:
            enc_out, acts, proprio_ctxt = prev_chunk
            current_idx, route_info = route(
                kind=router, library=library, world_model=world_model,
                encoder_output=enc_out, actions=acts, current_idx=current_idx,
                motion_gate=motion_gate, hysteresis=hysteresis,
                regime_label=regime_label, label_to_chart=label_to_chart,
                proprio_ctxt=proprio_ctxt, rng=router_rng,
            )
            umf_trace.append(route_info["scores"])
        else:
            umf_trace.append([None] * len(library))  # warmup: no scoring, stays on c0
        selected_trace.append(current_idx)

        chart = library[current_idx]
        chart.apply_(predictor)
        try:
            obs_td = _make_obs_td(obs["visual"], obs["proprio"], device)
            # steps_left must be in MODEL-chunk units -- CEMPlanner.plan() does
            # plan_length = min(self.horizon, steps_left), and self.horizon is
            # unambiguously model-chunk units. Multiplying by frameskip here
            # (an earlier version of this code did) mixes units with horizon --
            # fixed to match num_act_stepped's units instead.
            steps_left_model = (n_replans_target - replan_idx) * num_act_stepped
            action = agent.act(obs_td, steps_left=max(steps_left_model, 1))
        finally:
            chart.restore_(predictor)  # chart only needs to be applied during planning

        from einops import rearrange
        raw_actions = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
        raw_actions = agent.preprocessor.denormalize_actions(raw_actions).numpy()

        imgs = [obs["visual"]]
        proprios = [obs["proprio"]]
        step_actions = []
        for a in raw_actions:
            obs, reward, done, info = base_env.step(a)
            imgs.append(obs["visual"])
            proprios.append(obs["proprio"])
            step_actions.append(a)
            elapsed += 1
            if goal_utils.eval_state(goal_state, info["state"])["success"]:
                success = True
                break
        raw_steps_per_replan.append(len(step_actions))

        # Encode the just-executed chunk so the NEXT replan can score against
        # it — subsampled to the model time base and chunked to model actions
        # (E0_IMPLEMENTATION_PLAN.md T4), using world_model.encode() (not
        # preprocessor.transform_obs_visual + encode_obs) so real proprio is
        # captured -- this checkpoint's predictor requires it (see T1's
        # finding: forward_pred(proprio=None) is a channel-width mismatch,
        # not a graceful no-proprio path). If success cut this replan short
        # mid-frameskip-group, truncate to the largest prefix divisible by
        # frameskip -- prev_chunk from a successful replan is never consumed
        # again (the outer loop breaks right after this block).
        import numpy as np
        n_raw = (len(step_actions) // frameskip) * frameskip
        if n_raw == 0:
            if not success:
                prev_chunk = None
        else:
            keep_idx = list(range(0, n_raw + 1, frameskip))
            imgs_sub = np.stack([imgs[i] for i in keep_idx], axis=0)       # [T_model+1, H, W, 3]
            proprios_sub = np.stack([proprios[i] for i in keep_idx], axis=0)  # [T_model+1, P]
            visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
            proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
            # FIX_SPEC.md D10 (SUBMISSION_PLAN.md Part A-vi): .squeeze(0) below
            # removes the batch dim and is only safe because this call site
            # never batches multiple episodes -- assert the invariant rather
            # than rely on it silently. A batch size > 1 here would make
            # .squeeze(0) a silent no-op instead of removing the intended
            # axis, corrupting enc_out's shape without raising.
            assert visual_t.shape[0] == 1 and proprio_t.shape[0] == 1, (
                f"D10: expected a single-episode batch (dim0=1), got "
                f"visual={visual_t.shape}, proprio={proprio_t.shape}."
            )
            with torch.no_grad():
                enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
                enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)   # [T_model+1, N, D]
                proprio_enc = enc["proprio"]                                  # [1, T_model+1, P_tok, D_p]

            acts_np = np.stack(step_actions[:n_raw], axis=0)               # [n_raw, 2]
            act_norm = agent.preprocessor.normalize_actions(
                torch.from_numpy(acts_np).float().unsqueeze(0)
            ).squeeze(0)                                                    # [n_raw, 2]
            act_model = act_norm.reshape(n_raw // frameskip, frameskip * 2).to(device)  # [T_model, 10]

            prev_chunk = (enc_out, act_model, proprio_enc[:, 0:1])  # proprio_ctxt: [1,1,P_tok,D_p]

        if success:
            break

    record = {
        "episode_id": episode_id,
        "router": router,
        "seed": episode_seed,
        "success": success,
        "elapsed_raw_steps": elapsed,
        "n_replans": len(selected_trace),
        "raw_steps_per_replan": raw_steps_per_replan,
        "selected_trace": selected_trace,
        "umf_trace": umf_trace,
        "regime_label": regime_label,
    }
    log_episode(out_dir, record)
    return record


# ── E5 — Cross-policy matrix ──────────────────────────────────────────────────

def build_cross_policy_matrix(
    library: Library,
    world_model,
    chunks_per_chart: list[list[dict]],   # chunks_per_chart[j] = chunks from chart j's plans
    motion_gate: float | None,
) -> torch.Tensor:
    """
    Compute M[i,j] = chart i's mean UMF on chunks generated by chart j's plans.

    Args:
        library:          Chart library with K charts.
        world_model:      VideoWM instance (EncPredWM.model from torch.hub).
        chunks_per_chart: List of K lists; chunks_per_chart[j] are the encoded
                          chunks collected while chart j was active.
        motion_gate:      Informative-chunk gate.

    Returns:
        [K, K] matrix of mean UMF values (NaN where all chunks are gated).
    """
    K = len(library)
    if len(chunks_per_chart) != K:
        raise ValueError(
            f"chunks_per_chart has {len(chunks_per_chart)} entries but library has {K} charts."
        )
    M = torch.full((K, K), float("nan"))
    for i, chart_i in enumerate(library):
        for j, chunk_list in enumerate(chunks_per_chart):
            scores = []
            for chunk in chunk_list:
                s = compute_umf(chart_i, world_model,
                                chunk["encoder_output"], chunk["actions"], motion_gate)
                if s is not None:
                    scores.append(s)
            if scores:
                M[i, j] = sum(scores) / len(scores)
    return M


# ── JSONL logging ─────────────────────────────────────────────────────────────

def log_episode(out_dir: Path, record: dict) -> None:
    """Append one episode record to a JSONL log file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
