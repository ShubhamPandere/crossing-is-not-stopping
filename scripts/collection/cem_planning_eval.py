"""
scripts/run_e0_planning.py — E0's missing "planning success" half (Table T5's
Success column). Runs N CEM-driven episodes for one fixed, pre-trained E0
chart and reports success rate.

Does NOT use jepa-wms's own eval.py/PlanEvaluator — that's a heavyweight
distributed/dataset-dependent framework with confirmed Push-T bugs (see
code-review.md Bug #7: PlanEvaluator.eval() reads a Metaworld-only attribute
that doesn't exist on PushTEnv; TensorWrapper.step() assumes an info["success"]
key Push-T never sets). Reuses only GC_Agent/CEMPlanner (generic, no
task-specific code) plus PushTWrapper's two Push-T utility methods
(sample_random_init_goal_states, eval_state), with its own minimal episode loop.

Planner config defaults to the SUBSTRATE's own validated Push-T config
(vendor/jepa-wms/configs/evals/simu_env_planning/pt/dino-wm/
pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml:200-205):
num_samples=300, iterations=30, num_elites=10, horizon=6, num_act_stepped=6,
var_scale=1.0, frameskip=5 -> 30 raw steps/episode, 1 replan. This is the
config dino_wm_pusht reports ~90% Push-T SR under -- a DELIBERATE, DOCUMENTED
DEVIATION from implementation-plan Sec7.0's "CEM 200x10, horizon 25, 5
executed actions/replan" budget, justified as substrate fidelity: Sec7.0's
numbers are AdaJEPA's (a different substrate -- see ATLAS_implementation_plan_v2.md
Sec7.0a). Applied here, horizon=25 means 125 raw steps of lookahead for a task
DINO-WM samples to be feasible within 25. This RESOLVES E0_HANDOFF.md's open
num_act_stepped 1-vs-5 ambiguity: at nas=6, one replan covers the whole
30-step episode, so there is no ambiguity left to resolve (see
E0_IMPLEMENTATION_PLAN.md T6). An earlier version of this script used the
same jepa-wms shipped-config numbers but mislabeled them "the published spec"
without cross-checking against the plan's own Sec7.0/7.6 budget -- that
mislabeling was the real error (not the numbers themselves, which are correct
here); a later revision then switched to the AdaJEPA-derived
num_samples=200/horizon=25/nas=1 reading before this file settled on the
substrate config as the user's explicit final decision. All results computed
under either earlier config should be treated as invalid until re-run under
this one. See code-review.md Bug #7 for the memory numbers this config was
originally reported to OOM at on a 6GB GPU (needs Modal/a bigger local GPU).

Usage:
    python scripts/run_e0_planning.py --kind ln_act --regime R1 --episodes 1 \\
        --num-samples 16 --iterations 5 --horizon 3   # local smoke test
    python scripts/run_e0_planning.py --kind ln_act --regime R1 --episodes 20  # full spec, needs Modal
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import atlas
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

# One model step's action_dim=10 is FRAMESKIP=5 raw 2-dim actions concatenated
# (data.custom.frameskip=5 in the shipped config). Must unchunk before
# denormalize_actions/env.step(), which expect the raw 2-dim space.
FRAMESKIP = 5

# frameskip * goal_H + 1 (plan_evaluator.py's own goal_source=dset formula).
# goal_H here = max_steps/frameskip = 30/5 = 6 -- the EPISODE's raw-step
# budget. Numerically the same as the CEM planner's own horizon (also 6,
# post-T6), but conceptually independent -- this is the goal-sampling
# trajectory length, not the planner's lookahead.
GOAL_TRAJ_LEN = FRAMESKIP * 6 + 1

# Raw ATLAS_HOME env var, NOT atlas.ATLAS_HOME: that's .resolve()'d, which on
# Modal follows the volume mount (/atlas_root) down to its internal storage
# path (/__modal/volumes/vo-xxx/...) -- a path that isn't itself mounted
# anywhere, so torch.hub's file access on the resolved path 404s even though
# the file is really there under /atlas_root.
_atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
HUB_PATH = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402

import atlas.score as score  # noqa: E402
from atlas.chart import Chart  # noqa: E402
from atlas.regimes import PhysicsRegime, REGIME_CONFIGS, set_regime_config  # noqa: E402


def build_cfg(num_samples: int, iterations: int, horizon: int, num_act_stepped: int,
               objective_alpha: float = 0.1) -> OmegaConf:
    return OmegaConf.create({
        "local_seed": 0,
        "task_specification": {"obs": "rgb_state"},
        "planner": {
            "planner_name": "cem",
            "iterations": iterations,
            "num_samples": num_samples,
            "num_elites": min(10, num_samples),
            "horizon": horizon,
            "var_scale": 1.0,
            "num_act_stepped": num_act_stepped,
            "repeat_actskip": False,
            "decode_each_iteration": False,
            "distribute_planner": False,
            # alpha weights the PROPRIO term in the planner cost
            # (objectives.py: diff = diff_visual + alpha * diff_proprio).
            # Default 0.1 = the substrate's own validated value, unchanged for every
            # existing caller. Exposed only so C2_FAILURE_DIAGNOSIS.md §3.2 can be
            # tested: the chart's fine-tune loss is VISUAL-ONLY (harness.py:183), so
            # nothing constrains the proprio head it degrades here.
            "planning_objective": {"objective_type": "L2", "sum_all_diffs": False,
                                    "alpha": objective_alpha},
        },
    })


def prepare_with_visual(base_env: PushTEnv, regime: PhysicsRegime, seed: int, state: np.ndarray):
    # PushTWrapper.prepare() discards obs["visual"] (expects a PixelWrapper to
    # re-render it); we skip PixelWrapper, so reset the raw env directly.
    # Routed through regime.reset() (not base_env.reset()) so physics reapplies
    # in the one place PhysicsRegime.reset() already does it -- matches
    # atlas/harness.py::_prepare_env exactly (E0_IMPLEMENTATION_PLAN.md T6).
    # reset_to_state is set on base_env directly, not the regime wrapper: since
    # gym.Wrapper does not proxy attribute WRITES to the wrapped env.
    from atlas.regimes import assert_no_bypassed_corruption
    assert_no_bypassed_corruption(regime)  # FIX_SPEC.md C5 -- see docstring
    base_env.seed(seed)
    base_env.reset_to_state = state
    return regime.reset()


def make_obs_td(visual_hw3_uint8: np.ndarray, proprio_vec: np.ndarray, device: str) -> TensorDict:
    # No leading batch dim: GC_Agent.set_goal()/.act() add it themselves.
    visual = torch.from_numpy(visual_hw3_uint8.copy()).permute(2, 0, 1).float().unsqueeze(0)
    proprio = torch.from_numpy(np.asarray(proprio_vec, dtype=np.float32)).unsqueeze(0)
    return TensorDict({"visual": visual, "proprio": proprio}, batch_size=[]).to(device)


def load_dataset_states(split: str = "train") -> tuple[np.ndarray, list[int]]:
    # Raw recorded [agent_x, agent_y, T_x, T_y, angle] trajectories -- only
    # states.pth is needed (not visual/action/token files) to sample reachable
    # (init, goal) pairs; PushTEnv regenerates observations from state directly.
    d = atlas.DATA_DIR / "pusht_noise" / split
    states = torch.load(d / "states.pth", weights_only=False).numpy()
    with open(d / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    return states, seq_lengths


# P2d (E0_RECOVERY_PLAN.md): reachability cutoff, derived from data, not guessed.
# In P1 Step-4's 15-episode calibration cells, episode 2's agent-block distance
# was 184.37px -- the agent never made a single contact in 30 raw steps, in ALL
# THREE cells (R0, friction 2.0, damping 0.5): total_contacts==0 and
# block_pos_diff exactly equal to init_block_pos_diff in every cell. The
# next-highest distance among the OTHER 14 episodes (all of which made contact)
# was ep0 at 150.48px. 160.0 sits in the ~34px gap between the largest reachable
# distance observed (150.48) and the one confirmed-unreachable distance (184.37).
DEFAULT_MAX_AGENT_BLOCK_DIST = 160.0


def sample_dataset_init_goal(states: np.ndarray, seq_lengths: list[int], rs: np.random.RandomState,
                              traj_len: int = GOAL_TRAJ_LEN, min_block_pos_diff: float = 40.0,
                              max_agent_block_dist: float = DEFAULT_MAX_AGENT_BLOCK_DIST,
                              max_tries: int = 20, return_indices: bool = False):
    """
    Samples a real (init, goal) pair from the same real demo episode, `traj_len`
    raw timesteps apart.

    Real demos have idle/repositioning stretches where the block barely moves
    over that window -- sampling blindly can draw a pair already within (or
    very close to) block_success()'s own 20px/pi-9 threshold, making "success"
    trivial without any real pushing. Confirmed empirically: 5/10 baseline R0
    episodes (uniform sampling, no filter) finished in <=8 raw steps, one in a
    single action -- inflating the measured success rate. Retries (up to
    max_tries) until the block has moved at least min_block_pos_diff px
    between init and goal, so a real push is actually required.

    max_agent_block_dist (P2d): also rejects pairs where the sampled init
    state's agent-to-block distance exceeds this -- such pairs are
    unreachable within the episode's step budget regardless of policy
    quality (see DEFAULT_MAX_AGENT_BLOCK_DIST's derivation above), and burn
    episode budget on a case that can never discriminate between arms.

    Falls back to the last-drawn pair (with a warning) if nothing satisfies
    both conditions within max_tries, rather than looping forever -- some
    regions of the dataset may just not have long-range, reachable pushes at
    this traj_len.
    """
    valid_eps = [i for i, l in enumerate(seq_lengths) if l >= traj_len]
    init_state = goal_state = None
    for attempt in range(max_tries):
        ep_idx = valid_eps[rs.randint(len(valid_eps))]
        max_offset = seq_lengths[ep_idx] - traj_len
        offset = rs.randint(max_offset + 1) if max_offset > 0 else 0
        init_state = states[ep_idx, offset]
        goal_state = states[ep_idx, offset + traj_len - 1]
        block_pos_diff = np.linalg.norm(goal_state[2:4] - init_state[2:4])
        agent_block_dist = np.linalg.norm(init_state[0:2] - init_state[2:4])
        if block_pos_diff >= min_block_pos_diff and agent_block_dist <= max_agent_block_dist:
            break
        if attempt == max_tries - 1:
            print(f"    WARNING: sample_dataset_init_goal exhausted {max_tries} tries -- "
                  f"accepting a pair with block_pos_diff={block_pos_diff:.1f} "
                  f"(min {min_block_pos_diff}), agent_block_dist={agent_block_dist:.1f} "
                  f"(max {max_agent_block_dist})")
    # dataset states are 5-dim (no velocity); pad to match with_velocity=True's
    # 7-dim format -- generate_state() itself hardcodes agent velocity to 0
    # regardless, so zero-padding matches the existing convention exactly.
    init_state = np.concatenate([init_state, [0.0, 0.0]])
    goal_state = np.concatenate([goal_state, [0.0, 0.0]])
    if return_indices:
        # P18: so closed_loop collection can record which real demo episode +
        # offset it drew (episode_idx is otherwise null for that source).
        return init_state, goal_state, int(ep_idx), int(offset)
    return init_state, goal_state


def block_success(goal_state: np.ndarray, cur_state: np.ndarray) -> dict:
    # wrapper.eval_state() compares state[:4] = [agent_x, agent_y, T_x, T_y] together,
    # which is only meaningful when goal/init come from a correlated real trajectory
    # (goal_source=dset upstream). Our goals are independently random (no dataset
    # dependency by design), so the agent-position term is pure noise -- success
    # would require the pusher to land within 20px of an unrelated random point.
    # Push-T's actual objective (and IBC/Diffusion Policy's own metric) is block
    # position/orientation only -- state indices [T_x, T_y, angle] = [2, 3, 4].
    pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
    # PushTWrapper.generate_state() draws the goal angle as randn()*2pi - pi --
    # an unbounded Gaussian, not a wrapped angle -- so a raw difference can span
    # several full rotations. np.minimum(d, 2pi-d) (the original eval_state()
    # formula) only wraps correctly within one rotation; for larger raw
    # differences it produces garbage (even negative) results. Wrap properly
    # via the standard atan2-free formula: fold into (-pi, pi] first.
    raw_diff = goal_state[4] - cur_state[4]
    angle_diff = np.abs((raw_diff + np.pi) % (2 * np.pi) - np.pi)
    success = pos_diff < 20 and angle_diff < np.pi / 9
    return {"success": success, "block_pos_diff": float(pos_diff), "block_angle_diff": float(angle_diff)}


def run_episode(agent: GC_Agent, base_env: PushTEnv, wrapper: PushTWrapper, regime: PhysicsRegime,
                 seed: int, max_steps: int, num_act_stepped: int, states: np.ndarray,
                 seq_lengths: list[int], log_planner_diagnostics: bool = False,
                 min_block_pos_diff: float = 40.0,
                 max_agent_block_dist: float = DEFAULT_MAX_AGENT_BLOCK_DIST,
                 world_model=None, log_umf: bool = False, settle_steps: int = 0,
                 fix_steps_left: bool = False, settle_mode: str = "zero",
                 settle_log_velocity: bool = False) -> dict:
    device = agent.device

    rs = np.random.RandomState(seed)
    init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs,
                                                        min_block_pos_diff=min_block_pos_diff,
                                                        max_agent_block_dist=max_agent_block_dist)
    # Logged so episode difficulty is auditable straight from episodes.jsonl,
    # without recomputing from states.pth -- this is exactly the number the
    # min_block_pos_diff filter above controls.
    init_block_pos_diff = float(np.linalg.norm(goal_state[2:4] - init_state[2:4]))
    init_block_angle_diff = float(np.abs((goal_state[4] - init_state[4] + np.pi) % (2 * np.pi) - np.pi))
    # P2d: logged so a dead (unreachable) episode is auditable straight from
    # episodes.jsonl without recomputing from states.pth.
    init_agent_block_dist = float(np.linalg.norm(init_state[0:2] - init_state[2:4]))

    goal_obs, _ = prepare_with_visual(base_env, regime, seed, goal_state)
    agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))

    obs, _ = prepare_with_visual(base_env, regime, seed, init_state)

    # n_replans_target is a LOOSE upper bound on the replan loop -- the real
    # termination is `elapsed >= max_steps` inside the loop below, checked
    # every iteration. steps_left passed to agent.act() must be MODEL-CHUNK
    # units, matching CEMPlanner.horizon's units (plan_length = min(horizon,
    # steps_left)), NOT raw environment steps -- see
    # atlas/harness.py::run_e1_episode for the same convention. Under this
    # file's default config (horizon=6, num_act_stepped=6 -- T6's substrate
    # config), one replan executes num_act_stepped*frameskip = 30 raw actions,
    # i.e. the WHOLE 30-step episode in a single replan, matching the plan's
    # "1 replan" summary regardless of this loop bound's exact value.
    #
    # FIXLOG V3-24 (2026-09-02, pre-draft audit Pass 3, additive/opt-in):
    # `max_steps` is a RAW step count but `num_act_stepped` is a MODEL-CHUNK
    # count -- dividing one by the other (default path below) omits the
    # FRAMESKIP=5 conversion used everywhere else in this file, so at
    # num_act_stepped < horizon (e.g. nas=2) n_replans_target is ~FRAMESKIP-x
    # too large, which keeps steps_left_model far above `horizon` on every
    # replan including the last -- CEM's remaining-budget truncation
    # (`plan_length = min(horizon, steps_left)`, upstream planner.py) never
    # fires. Confirmed inert at nas>=horizon (one replan already covers the
    # episode). Default (flag unset) reproduces the exact original formula,
    # byte-identical -- every archived number is unaffected by this change.
    if fix_steps_left:
        n_replans_target = max(max_steps // (num_act_stepped * FRAMESKIP), 1)
    else:
        n_replans_target = max(max_steps // num_act_stepped, 1)

    elapsed = 0
    success = False
    success_at_step = None  # raw step at which the pass-through criterion first fired
    replans = 0
    total_contacts = 0
    final_check = {"block_pos_diff": None, "block_angle_diff": None}
    planner_diagnostics = []  # per-replan CEM elite-cost convergence, if requested
    umf_per_replan = []  # per-replan UMF of the EXECUTED chunk under the predictor
                          # state already in effect (chart, if any, applied once by
                          # main() before the episode loop -- NOT re-applied here)
    t_start = time.time()
    # Nested progress bar -- at low num_act_stepped (e.g. nas=2 -> 15 replans/episode,
    # each a FULL CEM search), a single episode can take several minutes with zero
    # visibility otherwise; leave=False so it clears once the episode's own line
    # (in main()'s outer episode-level bar) takes over.
    replan_pbar = tqdm(range(n_replans_target), desc="  replans", unit="replan", leave=False)
    for replan_idx in replan_pbar:
        if elapsed >= max_steps:
            break
        obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
        steps_left_model = (n_replans_target - replan_idx) * num_act_stepped
        action = agent.act(obs_td, steps_left=max(steps_left_model, 1))
        replans += 1

        if log_planner_diagnostics:
            # Free -- GC_Agent.plan() already computes and stores these; no extra
            # planner compute. Per-CEM-iteration mean/std of the elite candidates'
            # cost, e.g. does the search converge to a low-cost action, or stay
            # high/noisy throughout -- a cheap signal on whether an adapter is
            # confusing CEM's cost ranking, before capturing full per-candidate costs.
            planner_diagnostics.append({
                "elite_losses_mean": agent._prev_elite_losses_mean.squeeze(-1).tolist(),
                "elite_losses_std": agent._prev_elite_losses_std.squeeze(-1).tolist(),
            })

        action = rearrange(action.cpu(), "t (f d) -> (t f) d", d=2)
        action = agent.preprocessor.denormalize_actions(action).numpy()

        imgs = [obs["visual"]]
        proprios = [obs["proprio"]]
        step_actions = []
        for a in action:
            if elapsed >= max_steps:
                break
            obs, reward, done, info = base_env.step(a)
            imgs.append(obs["visual"])
            proprios.append(obs["proprio"])
            step_actions.append(a)
            elapsed += 1
            total_contacts += info["n_contacts"]
            final_check = block_success(goal_state, info["state"])
            if final_check["success"]:
                success = True
                success_at_step = elapsed
                break

        if log_umf and world_model is not None:
            # Mirrors atlas/harness.py::run_e1_episode's executed-chunk encoding
            # exactly (E0_IMPLEMENTATION_PLAN.md T4 pattern) -- world_model.encode()
            # (not preprocessor.transform_obs_visual+encode_obs) so real proprio is
            # captured, since this checkpoint's predictor requires it. If success
            # cut this replan short mid-frameskip-group, truncate to the largest
            # prefix divisible by FRAMESKIP; skip UMF entirely if that's 0 raw steps.
            n_raw = (len(step_actions) // FRAMESKIP) * FRAMESKIP
            if n_raw == 0:
                umf_per_replan.append(None)
            else:
                keep_idx = list(range(0, n_raw + 1, FRAMESKIP))
                imgs_sub = np.stack([imgs[i] for i in keep_idx], axis=0)
                proprios_sub = np.stack([proprios[i] for i in keep_idx], axis=0)
                visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
                proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
                # FIX_SPEC.md D10 (SUBMISSION_PLAN.md Part A-vi) -- see
                # atlas/harness.py's identical assertion for the full rationale.
                assert visual_t.shape[0] == 1 and proprio_t.shape[0] == 1, (
                    f"D10: expected a single-episode batch (dim0=1), got "
                    f"visual={visual_t.shape}, proprio={proprio_t.shape}."
                )
                with torch.no_grad():
                    enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
                    enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T_model+1, N, D]
                    proprio_enc = enc["proprio"]                                  # [1, T_model+1, P_tok, D_p]

                acts_np = np.stack(step_actions[:n_raw], axis=0)  # [n_raw, 2]
                act_norm = agent.preprocessor.normalize_actions(
                    torch.from_numpy(acts_np).float().unsqueeze(0)
                ).squeeze(0)  # [n_raw, 2]
                act_model_used = act_norm.reshape(n_raw // FRAMESKIP, FRAMESKIP * 2).to(device)  # [T_model, 10]

                umf_value = score.rollout_umf(
                    world_model, enc_out, act_model_used, proprio_ctxt=proprio_enc[:, 0:1],
                )
                umf_per_replan.append(umf_value)

        replan_pbar.set_postfix(steps=elapsed, success=success,
                                 elapsed_s=f"{time.time() - t_start:.0f}")
        if success:
            break
    replan_pbar.close()

    # FABLE5 Day 1.1 (criterion validity). The success criterion above fires per raw
    # step and breaks immediately (pass-through). Under R2 (space.damping=0.5) the
    # block glides, so a transient crossing can count even if the block sails on.
    # settle_steps > 0: from wherever the episode left the env (right after a
    # pass-through crossing, OR at max_steps for a non-success), hold position for
    # settle_steps raw steps (relative=True default => a zero action is "hold",
    # pusht_env.py:481-483; step() does not touch space.damping) and re-check
    # block_success. Applied to EVERY episode so the settled distance is one
    # measurement across the whole arm, not a substitution only on successes.
    # settled_trace records block_success at checkpoints so settle-count sensitivity
    # is answerable from a single run.
    # FIXLOG V3-26: settle_mode selects what drives the env during the hold.
    #   "zero"    -- the original and default: a zero (hold-position) action, i.e. the
    #                controller is switched off. Every archived number uses this.
    #   "planner" -- the planner keeps replanning through the hold, which is what a real
    #                deployment would do. Added because the zero-action hold is the most
    #                natural objection to the settled criterion; both modes share the same
    #                checkpoints so they are directly comparable on the same seeds.
    settle_info = None
    if settle_steps > 0:
        checkpoints = sorted({c for c in (1, 5, 15, 30, 45, settle_steps) if 1 <= c <= settle_steps})
        trace = []
        settled_state = info["state"]  # last state the episode loop reached
        settle_obs = obs             # last observation, for settle_mode="planner"
        stepped = 0
        # settle_log_velocity: the block's own velocity, straight from the simulator.
        # The distance-to-goal trace only ever gives the RADIAL component, so every
        # existing coast-model estimate of v0 is a lower bound; this is the real thing.
        vel_trace = []
        if settle_log_velocity:
            vel_trace.append({"step": 0,
                              "block_vel": [float(v) for v in base_env.block.velocity],
                              "block_ang_vel": float(base_env.block.angular_velocity)})
        pending = []  # unconsumed raw actions of the current settle plan
        for cp in checkpoints:
            while stepped < cp:
                if settle_mode == "planner":
                    if not pending:
                        obs_td_s = make_obs_td(settle_obs["visual"], settle_obs["proprio"], device)
                        act_s = agent.act(obs_td_s, steps_left=max(settle_steps - stepped, 1))
                        act_s = rearrange(act_s.cpu(), "t (f d) -> (t f) d", d=2)
                        pending = list(agent.preprocessor.denormalize_actions(act_s).numpy())
                    a_s = pending.pop(0)
                else:
                    a_s = np.zeros(2, dtype=np.float32)
                settle_obs, _, _, si = base_env.step(a_s)
                settled_state = si["state"]
                stepped += 1
            bc = block_success(goal_state, settled_state)
            trace.append({"step": cp, "block_pos_diff": bc["block_pos_diff"],
                          "block_angle_diff": bc["block_angle_diff"], "success": bool(bc["success"])})
            if settle_log_velocity:
                vel_trace.append({"step": cp,
                                  "block_vel": [float(v) for v in base_env.block.velocity],
                                  "block_ang_vel": float(base_env.block.angular_velocity)})
        settled_check = block_success(goal_state, settled_state)
        settle_info = {
            "success_at_step": success_at_step,
            "passthrough_success": bool(success),
            "settle_mode": settle_mode,
            "settled_success": bool(settled_check["success"]),
            "settled_block_pos_diff": settled_check["block_pos_diff"],
            "settled_block_angle_diff": settled_check["block_angle_diff"],
            "settled_trace": trace,
        }
        if settle_log_velocity:
            settle_info["settled_velocity_trace"] = vel_trace

    result = {"success": success, "steps": elapsed, "replans": replans, "wall_time": time.time() - t_start,
              "init_block_pos_diff": init_block_pos_diff, "init_block_angle_diff": init_block_angle_diff,
              "init_agent_block_dist": init_agent_block_dist,
              "total_contacts": total_contacts,
              "fix_steps_left": fix_steps_left,
              **{k: v for k, v in final_check.items() if k != "success"}}
    if settle_info is not None:
        result.update(settle_info)
    if log_planner_diagnostics:
        result["planner_diagnostics"] = planner_diagnostics
    if log_umf:
        umf_valid = [v for v in umf_per_replan if v is not None]
        result["umf_per_replan"] = umf_per_replan
        result["umf_mean"] = (sum(umf_valid) / len(umf_valid)) if umf_valid else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E0: planning success (CEM-driven episodes).")
    parser.add_argument("--kind", required=True, choices=["baseline", "ln_act", "lora4", "full"],
                         help="'baseline' = frozen pretrained predictor, no chart applied.")
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--regime-config", type=str, default=None,
                         help="JSON string overriding this regime's physics params for "
                              "calibration sweeps (P1 Step 2), e.g. '{\"friction\": 2.0}' or "
                              "'{\"damping\": 0.5}'. Applied via atlas.regimes.set_regime_config "
                              "before PhysicsRegime is constructed; logged into the summary JSON "
                              "and every per-episode record so cells stay attributable. Omit to "
                              "use REGIME_CONFIGS' existing default for --regime.")
    parser.add_argument("--objective-alpha", type=float, default=0.1,
                         help="Weight on the PROPRIO term in the CEM planner cost "
                              "(objectives.py: diff_visual + alpha*diff_proprio). "
                              "Default 0.1 = the substrate's validated value; every "
                              "prior run used it, so the default is byte-identical to "
                              "before this flag existed. --objective-alpha 0 ablates "
                              "the proprio term -- the C2_FAILURE_DIAGNOSIS.md §3.2 test, "
                              "since chart fine-tuning supervises VISUAL latents only "
                              "and leaves the proprio head unconstrained.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=300,
                         help="Substrate's own validated Push-T config (T6): 300. "
                              "Use fewer (e.g. 16) for a local smoke test.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6,
                         help="MODEL-chunk units (1 chunk = FRAMESKIP=5 raw actions via "
                              "action chunking). Substrate default nas=6: one replan executes "
                              "6*5=30 raw actions -- the whole 30-step episode in one replan, "
                              "matching dino_wm_pusht's own validated eval config "
                              "(E0_IMPLEMENTATION_PLAN.md T6).")
    parser.add_argument("--charts-dir", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--out-dir", type=Path, default=atlas.OUT_DIR / "e0_planning",
                         help="Where to write per-episode JSONL + summary JSON.")
    parser.add_argument("--settle-mode", choices=["zero", "planner"], default="zero",
                         help="FIXLOG V3-26. What drives the env during the settle hold. "
                              "'zero' (default, and what every archived cell used) commands a "
                              "zero hold-position action, i.e. the controller is switched off. "
                              "'planner' keeps replanning through the hold -- the control for "
                              "the objection that a real deployment would not stop controlling.")
    parser.add_argument("--settle-log-velocity", action="store_true",
                         help="FIXLOG V3-26. Record the block's linear/angular velocity straight "
                              "from the simulator at settle step 0 and every settled_trace "
                              "checkpoint, as settled_velocity_trace. The distance-to-goal trace "
                              "only gives the radial component, so this is the only source of a "
                              "true v0 for the coast model.")
    parser.add_argument("--settle-steps", type=int, default=0,
                         help="FABLE5 six-day plan Day 1.1 (criterion-validity check). Default 0 = "
                              "byte-identical current behaviour. When >0: for EVERY episode, from "
                              "wherever the episode left the env (right after a pass-through crossing, "
                              "or at max_steps otherwise), step this many raw times with a zero "
                              "(hold-position) action and re-check block_success. Records "
                              "passthrough_success / success_at_step / settled_success / "
                              "settled_block_pos_diff / settled_block_angle_diff / settled_trace "
                              "(block_success at checkpoints 1/5/15/30/45/N). Isolates transient "
                              "goal-window crossings under R2's gliding physics. No planning-loop "
                              "decision changes; `success` / `block_pos_diff` keep pass-through semantics.")
    parser.add_argument("--fix-steps-left", action="store_true",
                         help="FIXLOG V3-24 (pre-draft audit Pass 3, opt-in). Default off = byte-"
                              "identical current behaviour: n_replans_target = max_steps // "
                              "num_act_stepped, which omits the FRAMESKIP conversion and (at "
                              "num_act_stepped < horizon) keeps steps_left_model far above `horizon` "
                              "on every replan, so CEM's remaining-budget truncation never fires. "
                              "When set: n_replans_target = max_steps // (num_act_stepped * "
                              "FRAMESKIP), matching the raw-step/model-chunk unit convention used "
                              "everywhere else in this file. Recorded per-episode as "
                              "'fix_steps_left' for provenance.")
    parser.add_argument("--log-planner-diagnostics", action="store_true",
                         help="Log CEM's per-iteration elite-cost mean/std (free -- already computed).")
    parser.add_argument("--no-log-umf", action="store_true",
                         help="Disable per-replan UMF logging (on by default). UMF is computed on "
                              "the ALREADY-EXECUTED chunk under whatever predictor state is in effect "
                              "(chart applied once at episode start, or frozen for --kind baseline) -- "
                              "reuses atlas/harness.py::run_e1_episode's encode pattern, no extra CEM "
                              "compute, adds one encode() + one predictor unroll per replan. Writes "
                              "umf_per_replan/umf_mean into each episode record, giving an "
                              "episode-level (UMF, success) pair for the dissociation figure instead "
                              "of only 5 arm-level ones.")
    parser.add_argument("--episode-start", type=int, default=0,
                         help="First episode/seed index this invocation runs (inclusive) -- lets two "
                              "Modal containers split one N-episode request into non-overlapping "
                              "shards (e.g. --episode-start 0 --episodes 50 and --episode-start 50 "
                              "--episodes 100) that run concurrently and write to separate files via "
                              "--out-suffix. seed == episode index throughout this script, so shards "
                              "never collide on RNG draws. Merge with scripts/merge_planning_shards.py.")
    parser.add_argument("--out-suffix", type=str, default="",
                         help="Appended to the output JSONL/summary filenames (e.g. '_shard0'), so "
                              "concurrent shards of the same kind/regime don't overwrite each other's "
                              "output file. Leave empty for a single-container run.")
    parser.add_argument("--min-block-pos-diff", type=float, default=40.0,
                         help="Minimum block displacement (px) required between a sampled real "
                              "init/goal pair -- rejects pairs where the block barely moved over "
                              "the real demo's 30-step window (real demos have idle/repositioning "
                              "stretches), which otherwise makes 'success' trivial without any "
                              "real pushing. Confirmed empirically: unfiltered sampling gave 5/10 "
                              "baseline R0 episodes finishing in <=8 raw steps. See "
                              "sample_dataset_init_goal()'s docstring.")
    parser.add_argument("--max-agent-block-dist", type=float, default=DEFAULT_MAX_AGENT_BLOCK_DIST,
                         help="Maximum agent-to-block distance (px) allowed in a sampled init "
                              "state -- rejects pairs the agent cannot plausibly reach within the "
                              "episode's step budget (E0_RECOVERY_PLAN.md P2d). Derived from P1 "
                              "Step-4 data: episode 2's agent_block_dist=184.37px made zero "
                              "contact in ALL THREE calibration cells, while the next-highest "
                              "reachable distance among the other 14 episodes was 150.48px -- the "
                              "default sits in that gap. See sample_dataset_init_goal()'s docstring.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(
        HUB_PATH, "dino_wm_pusht", source="local", force_reload=False, trust_repo=True,
    )
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = wm.to(device)
    model = model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    # T7 throughput fixes (bit-exact-ish; measured payoff order 1 > 2):
    # 1. SDPA is absent from the eval YAML so it defaults False, falling back
    #    to manual attention that materialises [num_samples,16,512,512] fp32
    #    three times per layer x 6 layers -- also the memory fix keeping
    #    num_samples=300 inside a 24GB GPU. Enabled post-load, not via YAML.
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    # 2. The checkpoint was trained in bf16 -- minimum matmul-precision fix.
    torch.set_float32_matmul_precision("high")

    if args.kind == "baseline":
        print("kind=baseline -- no chart applied, using frozen pretrained predictor as-is.")
    else:
        chart_path = args.charts_dir / f"chart_{args.kind}_{args.regime}.pt"
        print(f"Applying chart: {chart_path}")
        chart = Chart.load(str(chart_path), wm.predictor)
        chart.apply_(wm.predictor)

    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped,
                     objective_alpha=args.objective_alpha)
    agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
    agent.device = device

    base_env = PushTEnv(render_size=224, with_velocity=True)
    # regime wraps base_env DIRECTLY (not PushTWrapper) -- PushTWrapper.reset()
    # discards obs["visual"] (returns state instead, expecting a PixelWrapper
    # to re-render it), so PhysicsRegime(wrapper, ...) would silently break
    # prepare_with_visual()'s regime.reset() call. Matches
    # atlas/harness.py::run_e1_episode's construction exactly
    # (E0_IMPLEMENTATION_PLAN.md T6). `wrapper` itself is otherwise unused in
    # this file (goal sampling/success use the local sample_dataset_init_goal/
    # block_success functions, not PushTWrapper's methods) -- kept only for
    # run_episode()'s existing signature.
    if args.regime_config is not None:
        set_regime_config(args.regime, json.loads(args.regime_config))
    resolved_regime_cfg = dict(REGIME_CONFIGS.get(args.regime, {}))

    wrapper = PushTWrapper(base_env)
    regime = PhysicsRegime(base_env, args.regime)
    states, seq_lengths = load_dataset_states()

    print(f"Running {args.episodes} episode(s): kind={args.kind} regime={args.regime} "
          f"num_samples={args.num_samples} iterations={args.iterations} horizon={args.horizon} "
          f"num_act_stepped={args.num_act_stepped}")
    if args.num_samples < 300 or args.horizon < 6:
        print("NOTE: reduced CEM settings -- not the substrate's validated config (T6). See module docstring.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Resume/append support: don't silently overwrite + recompute episodes an
    # earlier invocation already paid real GPU time for (e.g. a small sanity
    # run before committing to a bigger --episodes count). jsonl_path's write
    # loop below always writes exactly one record per ep in range(args.episodes)
    # (seed=ep, no skipping) -- so on-disk episode indices are contiguous from
    # 0 by construction, unless the file was hand-edited or came from a
    # different (kind, regime) run merged in by mistake.
    # --out-suffix keeps concurrent shards (--episode-start-split runs) in
    # separate files; --episode-start shifts the starting seed/episode index
    # so two containers can cover non-overlapping ranges of the same
    # (kind, regime). seed == episode index everywhere in this script, so
    # shards never collide on RNG draws regardless of run order.
    jsonl_path = args.out_dir / f"{args.kind}_{args.regime}{args.out_suffix}.jsonl"
    existing_records = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))
        existing_eps = sorted(r["episode"] for r in existing_records)
        expected = list(range(args.episode_start, args.episode_start + len(existing_eps)))
        if existing_eps != expected:
            print(f"WARNING: {jsonl_path} episode indices are not contiguous from "
                  f"--episode-start={args.episode_start} ({existing_eps}) -- resume logic "
                  f"assumes contiguity within a shard; treating already_done as "
                  f"max(episode)+1, which may re-run or skip unexpectedly if the file was "
                  f"hand-edited or mixes runs.")
        already_done = (max(existing_eps) + 1) if existing_eps else args.episode_start
    else:
        already_done = args.episode_start

    if existing_records and args.episodes <= already_done:
        print(f"Requested --episodes {args.episodes} already satisfied by {already_done} "
              f"existing episode(s) in {jsonl_path} -- skipping episode loop, "
              f"recomputing summary from existing records only.")
    else:
        results = []
        new_eps = range(already_done, args.episodes)
        if already_done > args.episode_start:
            print(f"Resuming: {already_done - args.episode_start} episode(s) already in "
                  f"{jsonl_path}, running {len(new_eps)} new episode(s) "
                  f"(seeds {already_done}..{args.episodes - 1}).")
        pbar = tqdm(new_eps, desc=f"{args.kind}_{args.regime}{args.out_suffix}", unit="ep")
        with open(jsonl_path, "a" if already_done > args.episode_start else "w") as f:
            for ep in pbar:
                result = run_episode(agent, base_env, wrapper, regime, seed=ep, max_steps=args.max_steps,
                                      num_act_stepped=args.num_act_stepped, states=states,
                                      seq_lengths=seq_lengths,
                                      log_planner_diagnostics=args.log_planner_diagnostics,
                                      min_block_pos_diff=args.min_block_pos_diff,
                                      max_agent_block_dist=args.max_agent_block_dist,
                                      world_model=model, log_umf=not args.no_log_umf,
                                      settle_steps=args.settle_steps,
                                      fix_steps_left=args.fix_steps_left,
                                      settle_mode=args.settle_mode,
                                      settle_log_velocity=args.settle_log_velocity)
                results.append(result)
                f.write(json.dumps({"episode": ep, "kind": args.kind, "regime": args.regime,
                                     "regime_config": resolved_regime_cfg, **result}) + "\n")
                f.flush()
                pbar.set_postfix(success=result["success"], steps=result["steps"])

    # Summary stats are computed over ALL episodes on disk (old + newly run),
    # read back from the JSONL rather than relying on this invocation's
    # in-memory `results` list -- so a resumed run's summary reflects the full
    # accumulated episode count, not just what this process itself ran.
    with open(jsonl_path) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    success_rate = sum(r["success"] for r in all_records) / len(all_records)
    mean_time = sum(r["wall_time"] for r in all_records) / len(all_records)
    settled_records = [r for r in all_records if "settled_success" in r]
    settled_success_rate = (
        sum(r["settled_success"] for r in settled_records) / len(settled_records)
        if settled_records else None
    )
    # Peak GPU memory below reflects ONLY this process's own run (this
    # invocation's episodes, if any were run) -- torch's peak-memory-allocated
    # stat is process-local and can't retroactively recover a peak from an
    # earlier invocation's now-exited process. If this call resumed 0 new
    # episodes, this number is meaningless (no allocation happened here) and
    # should not be read as "peak over all episodes."
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    print(f"\nSuccess rate: {success_rate:.2f} ({sum(r['success'] for r in all_records)}/{len(all_records)})")
    print(f"Mean wall time per episode: {mean_time:.1f}s")
    if peak_mem_gb is not None:
        print(f"Peak GPU memory (THIS PROCESS ONLY, not necessarily over all "
              f"{len(all_records)} accumulated episodes if this was a resumed run): {peak_mem_gb:.2f} GB")

    umf_means = [r["umf_mean"] for r in all_records if r.get("umf_mean") is not None]

    summary_path = args.out_dir / f"{args.kind}_{args.regime}{args.out_suffix}_summary.json"
    summary_path.write_text(json.dumps({
        "kind": args.kind, "regime": args.regime, "regime_config": resolved_regime_cfg,
        "episodes": len(all_records),
        "episode_start": args.episode_start, "out_suffix": args.out_suffix,
        "num_samples": args.num_samples, "iterations": args.iterations, "horizon": args.horizon,
        "num_act_stepped": args.num_act_stepped,
        "min_block_pos_diff": args.min_block_pos_diff, "max_agent_block_dist": args.max_agent_block_dist,
        "success_rate": success_rate, "mean_wall_time_s": mean_time, "peak_gpu_memory_gb": peak_mem_gb,
        "settle_steps": args.settle_steps,
        "settle_mode": args.settle_mode,
        "settle_log_velocity": bool(args.settle_log_velocity),
        "fix_steps_left": args.fix_steps_left,
        "settled_success_rate": settled_success_rate,
        "settled_episodes_with_value": len(settled_records),
        "log_umf": not args.no_log_umf,
        "umf_mean_of_means": (sum(umf_means) / len(umf_means)) if umf_means else None,
        "umf_episodes_with_value": len(umf_means),
    }, indent=2))


if __name__ == "__main__":
    main()
