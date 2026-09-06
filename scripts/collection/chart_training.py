"""
scripts/run_e0.py — E0: Adapter capacity experiment.

Fine-tunes {ln_act, lora4, full} × {R1, R2} = 6 charts offline.
Evaluates UMF reduction and planning success in-regime.

Usage:
    python scripts/run_e0.py
    python scripts/run_e0.py --kinds ln_act lora4 --regimes R1 --steps 500  # quick test

Output:
    atlas_out/e0/chart_{kind}_{regime}.pt
    atlas_out/e0/loss_{kind}_{regime}.json
    atlas_out/e0/results.json   (UMF and success per kind × regime)
    atlas_out/e0/results.md     (T5 supplementary table)
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import determinism as _determinism  # noqa: E402  — sets CUBLAS_WORKSPACE_CONFIG before torch

import torch  # noqa: E402
from tqdm import tqdm
import atlas
from atlas.chart import Chart, ChartKind
from atlas.harness import run_e0_finetune, log_episode
from atlas.regimes import REGIME_CONFIGS, set_regime_config
from atlas.score import compute_motion_gate

DataSource = Literal["scripted", "dataset", "hybrid", "closed_loop"]


@functools.lru_cache(maxsize=None)
def _load_pusht_demo_dataset(split: str = "train"):
    """
    Loads (and caches, per-process) the real Push-T demonstration dataset used
    by T9's replay path. Cached because load_regime_trajectories() is called
    multiple times per regime (train + eval, and again by scripts/run_e1.py
    for its motion-gate sample) -- without caching, each call would re-read
    ~260MB (states.pth + rel_actions.pth) from disk.

    Only states.pth/rel_actions.pth/seq_lengths.pkl are loaded -- NOT
    tokens.pth (4.8GB; precomputed encodings we don't want, since we need
    live proprio/visual through world_model.encode(), see module docstring)
    or abs_actions.pth (env.step() with relative=True, the PushTEnv default,
    consumes rel_actions -- confirmed against pusht_dset.py:36-52's identical
    choice).

    Returns (states [N,246,5] float64, rel_actions [N,246,2] float64,
    seq_lengths: list[int]). states/actions are NOT dataset-normalized (no
    mean/std applied) -- these are the raw recorded values, matching what
    PushTEnv.reset_to_state / env.step() expect (env.step() itself does
    action*action_scale internally, action_scale=100 by default -- see
    _load_pusht_demo_dataset's caller for the /action_scale conversion,
    empirically confirmed exact under R0: replaying a real episode's actions
    from its own recorded initial state reproduces the recorded states to
    ~1e-5 (float32/64 rounding only), before any PhysicsRegime shift).
    """
    d = atlas.DATA_DIR / "pusht_noise" / split
    states = torch.load(d / "states.pth", weights_only=False).numpy()
    rel_actions = torch.load(d / "rel_actions.pth", weights_only=False).numpy()
    with open(d / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    return states, rel_actions, seq_lengths


def load_regime_trajectories(world_model, preprocessor, regime: str, num_trajs: int = 5, traj_len: int = 50, device: str = "cpu", max_tries: int = 8, seed_offset: int = 0, frameskip: int = 5, source: DataSource = "scripted", data_split: str = "train", agent=None, min_block_pos_diff: float = 40.0, max_agent_block_dist: float | None = None, corruption: str = "none", corruption_severity: float = 0.5, record_block_pose: bool = False, collect_nas: int = 1, traj_idx_offset: int = 0) -> list[dict]:
    """
    Collects real trajectories from PushTEnv under the specified regime and
    encodes them through the frozen vision backbone.

    Physics regime is applied via atlas.regimes.PhysicsRegime — R0=default,
    R1=high friction, R2=high restitution. R1/R2 are NOT mass/damping-based:
    see ATLAS_implementation_plan_v2.md §6.1a and REGIME_DESIGN_REVIEW.md —
    Push-T's kinematic pusher makes mass-based shifts physically impossible to
    detect at any scale, so R1/R2 were re-targeted onto shape.friction and
    shape.elasticity, which are not subject to the same cancellation.

    Actions are NOT sampled IID per step. A single random target near the
    block is chosen once per trajectory, and the agent aims at it every step
    (small Gaussian noise added for diversity) — pure per-step IID random
    actions (the previous scheme) were empirically found to produce
    agent-block contact in only ~13-17% of rollouts, since a memoryless random
    walk rarely displaces net position over a short horizon. This mirrors how
    every public Push-T lineage (IBC, Diffusion Policy, DINO-WM) generates
    data — none uses fresh per-step random actions either; see
    ACTION_SAMPLING_REVIEW.md. Each trajectory is retried with a new seed (up
    to max_tries) if it produces zero total contact, as a hard guarantee on
    top of the aimed sampling (which only raises the expected contact rate,
    it doesn't guarantee it for every seed/spawn configuration).

    Frame/action chunking (E0_IMPLEMENTATION_PLAN.md T3): raw env steps are
    still what gets simulated (frameskip has no effect on physics), but the
    RETURNED actions/frames are chunked/subsampled to the MODEL time base --
    `_open_loop_rollout` unrolls one model step per `frameskip` raw steps, so
    training/eval data must match that or every step after the first runs on
    a stale, wrong-time-base target (see E0_DIAGNOSIS_AND_PLAN.md's root-cause
    writeup). `world_model.encode()` (not the old
    preprocessor.transform_obs_visual + encode_obs path) is used so real
    proprio is captured and normalized/embedded the same way the planner does
    it -- this checkpoint's predictor concatenates proprio into the token
    channel width (VideoWM.concat_obs_act(), dim=3) and errors without it, so
    it is not an optional enrichment here.

    source="dataset" (E0_IMPLEMENTATION_PLAN.md T9) replaces the scripted
    aimed-walk actions above with REAL recorded action sequences from
    data/pusht_noise/{data_split}/ (18,685 real Push-T demonstrations, the
    same distribution dino_wm_pusht's checkpoint was trained on): a real
    episode + offset is chosen (per-trajectory, per-attempt, seeded exactly
    like the scripted path's target/noise draws), the env is reset to that
    episode's REAL recorded state via reset_to_state (not a fresh random
    reset), and the episode's REAL recorded raw actions are replayed step by
    step. The recording was made under R0 physics; replaying those same
    actions under PhysicsRegime's R1/R2 genuinely diverges from the original
    trajectory (confirmed: replaying under R0 itself reproduces the recorded
    states to ~1e-5, i.e. float rounding only -- so any larger divergence
    under R1/R2 is the regime shift, not a bug in the replay). The
    accept/retry-on-zero-contact structure (max_tries) is unchanged between
    both sources.

    Args:
        world_model: EncPredWM instance (the object torch.hub.load returns —
                     NOT .model).
        source:      'scripted' (default, unchanged) or 'dataset' (T9 replay).
        data_split:  Which data/pusht_noise/{split}/ to draw real episodes
                     from when source='dataset'. Unused for 'scripted'.

    Returns a list of dicts:
        {'encoder_output': [T_model+1, N, D],
         'actions':        [T_model, 10]  (model-chunk, normalized),
         'proprio':        [T_model+1, P_tok, D_p]  (encoded, full sequence),
         'seed':           int,   # accepted seed (both sources)
         'episode_idx':    int | None,  # source='dataset' only: which real episode
         'offset':         int | None}  # source='dataset' only: start offset within it
    """
    import sys
    import numpy as np
    from pathlib import Path

    # Ensure hub code is in sys.path
    hub_path = str(Path(__file__).parent.parent.parent / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    if hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    from atlas.regimes import PhysicsRegime

    if source == "closed_loop":
        # Reused rather than reimplemented: run_e0_planning.py already owns the
        # exact (init, goal) sampler, env-reset path and obs-TensorDict layout
        # the planner is validated against. Importing the sibling script keeps
        # collection and evaluation on literally the same code (a divergence
        # here would train on one task distribution and evaluate on another).
        from einops import rearrange
        from run_e0_planning import (DEFAULT_MAX_AGENT_BLOCK_DIST, GOAL_TRAJ_LEN,
                                      make_obs_td, prepare_with_visual,
                                      sample_dataset_init_goal)
        if agent is None:
            raise ValueError("source='closed_loop' requires a GC_Agent (pass agent=...)")
        if max_agent_block_dist is None:
            max_agent_block_dist = DEFAULT_MAX_AGENT_BLOCK_DIST

    if traj_len % frameskip != 0:
        raise ValueError(
            f"traj_len={traj_len} must be divisible by frameskip={frameskip} so "
            f"frame subsampling and action chunking land exactly on model steps."
        )

    # Disjoint seed ranges per regime (unchanged spirit from the original code,
    # extended to cover R0 too since PhysicsRegime now supports all three).
    # seed_offset additionally separates independent calls for the same regime
    # (e.g. train vs. eval trajectories) so they never draw the same seeds --
    # without it, two calls both starting traj_idx at 0 would silently overlap.
    seed_base = {"R0": 2000, "R1": 0, "R2": 1000}.get(regime, 0) + seed_offset

    if source in ("dataset", "hybrid", "closed_loop"):
        # hybrid (P2b) needs demo_states for real init sampling but not
        # demo_rel_actions (actions are generated live, not replayed) --
        # loaded anyway since _load_pusht_demo_dataset returns both from the
        # same file and is cached, so there is no extra cost.
        demo_states, demo_rel_actions, demo_seq_lengths = _load_pusht_demo_dataset(data_split)
        # Need traj_len real actions [offset, offset+traj_len) plus the state
        # AFTER the last one (offset+traj_len) as a valid (non-padding) row --
        # so seq_length must cover index offset+traj_len, i.e. >= traj_len+1.
        if source == "closed_loop" and min(demo_seq_lengths) < GOAL_TRAJ_LEN:
            # P5 / v3 §3.2: the (init, goal) pair is drawn GOAL_TRAJ_LEN-1 demo
            # steps apart, DECOUPLED from the collector's own rollout length
            # (traj_len). Assert the pool supports it rather than assuming.
            raise ValueError(
                f"closed_loop needs every demo episode to have seq_length >= "
                f"GOAL_TRAJ_LEN={GOAL_TRAJ_LEN}; min in data/pusht_noise/"
                f"{data_split} is {min(demo_seq_lengths)}."
            )
        valid_eps = [i for i, l in enumerate(demo_seq_lengths) if l >= traj_len + 1]
        if not valid_eps:
            raise ValueError(
                f"No episodes in data/pusht_noise/{data_split} have seq_length >= "
                f"traj_len+1={traj_len + 1} (max seq_length in this split is "
                f"{max(demo_seq_lengths)}) -- reduce traj_len."
            )

    trajectories = []
    traj_pbar = tqdm(range(num_trajs), desc=f"collect_{source}_{regime}", unit="traj")
    for traj_idx in traj_pbar:
        for attempt in range(max_tries):
            # traj_idx_offset (sharding): shifts which slice of the seed space
            # this call draws from, so N concurrent shard containers collecting
            # disjoint traj_idx ranges [0,k), [k,2k), ... never draw the same
            # seed. Default 0 -> byte-identical to the unsharded formula.
            seed = seed_base + (traj_idx + traj_idx_offset) * max_tries + attempt
            rs = np.random.RandomState(seed)
            # with_velocity=True: matches scripts/run_e1.py and the shipped
            # eval YAML (env.with_velocity: true) -- the checkpoint's
            # preprocessor.proprio_std is sized for the 4-dim (x,y,vx,vy)
            # proprio this produces, not the 2-dim default. Confirmed
            # empirically (RuntimeError on proprio_std shape mismatch without
            # it) -- see E0_IMPLEMENTATION_PLAN.md T3.
            base_env = PushTEnv(render_size=224, with_velocity=True)
            env = PhysicsRegime(base_env, regime)
            if corruption != "none":
                # E2 only: appearance shifts on top of (or instead of) a physics
                # shift. Wraps OUTSIDE PhysicsRegime so physics is untouched --
                # the corruption only rewrites obs["visual"] on its way out.
                # Seeded per trajectory so paired arms see identical noise.
                from atlas.regimes import VisualCorruption
                env = VisualCorruption(env, corruption, corruption_severity, seed=seed)

            episode_idx: int | None = None
            offset: int | None = None
            if source == "closed_loop":
                # Goal must be prepared BEFORE the init reset: prepare_with_visual
                # resets the env to whatever state it renders, so rendering the
                # goal last would leave the env sitting on the goal.
                # P5 / v3 §3.2: goal separation is GOAL_TRAJ_LEN (31 raw steps,
                # == run_e0_planning.py's eval value), NOT the collector's
                # rollout length. Passing `traj_len` here drew goals only
                # traj_len-1 steps apart — a shorter, easier task than eval.
                # P18: return_indices=True so episode_idx/offset are recorded.
                init_state7, goal_state, episode_idx, offset = sample_dataset_init_goal(
                    demo_states, demo_seq_lengths, rs, traj_len=GOAL_TRAJ_LEN,
                    min_block_pos_diff=min_block_pos_diff,
                    max_agent_block_dist=max_agent_block_dist, return_indices=True)
                goal_obs, _ = prepare_with_visual(base_env, env, seed, goal_state)
                agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
                base_env.seed(seed)
                base_env.reset_to_state = init_state7
            elif source in ("dataset", "hybrid"):
                episode_idx = int(valid_eps[rs.randint(len(valid_eps))])
                max_offset = demo_seq_lengths[episode_idx] - traj_len - 1
                offset = int(rs.randint(max_offset + 1)) if max_offset > 0 else 0
                init_state5 = demo_states[episode_idx, offset]  # [ax, ay, Tx, Ty, angle] -- no velocity
                # Pad to with_velocity=True's 7-dim reset state -- PushTEnv's
                # own random-reset branch hardcodes agent velocity to 0
                # regardless of the sampled state, so zero-padding here
                # matches the existing convention exactly (also done
                # identically in run_e0_planning.py::sample_dataset_init_goal).
                init_state7 = np.concatenate([init_state5, [0.0, 0.0]])
                # reset_to_state/seed are set on the INNER env, not the
                # PhysicsRegime wrapper -- gym.Wrapper proxies attribute READS
                # but not WRITES (see atlas/harness.py::_prepare_env's
                # docstring, and run_e0_planning.py::prepare_with_visual,
                # which this mirrors).
                base_env.seed(seed)
                base_env.reset_to_state = init_state7
            else:
                env.seed(seed)

            obs, state = env.reset()
            imgs = [obs["visual"]]  # RGB array [224, 224, 3]
            # Phase-0 measurement (P0-B motion gate): block (Tx, Ty, angle) per raw
            # step. Frame 0 comes from the reset state; every subsequent frame from
            # info["block_pose"] (PushTEnv._get_info). Off by default — pure add-on.
            block_poses = [np.asarray([state[2], state[3], state[4]], dtype=np.float64)]
            proprios = [obs["proprio"]]
            raw_actions = []
            total_contacts = 0

            if source == "dataset":
                action_scale = env.action_scale  # read-proxied through PhysicsRegime; env.action_scale==100 default
                for t in range(traj_len):
                    # Real recorded action, converted back to the raw
                    # [-1,1]-ish input env.step() expects -- env.step()
                    # itself does action*action_scale internally (relative=True
                    # default: += agent.position). Confirmed empirically:
                    # replaying an episode's own actions from its own recorded
                    # initial state under R0 reproduces the recorded states to
                    # ~1e-5 (see module docstring).
                    act = demo_rel_actions[episode_idx, offset + t] / action_scale
                    obs, reward, done, info = env.step(act)
                    total_contacts += info["n_contacts"]
                    imgs.append(obs["visual"])
                    proprios.append(obs["proprio"])
                    block_poses.append(np.asarray(info["block_pose"], dtype=np.float64))
                    raw_actions.append(act)
            elif source == "closed_loop":
                # ON-POLICY data (v3 §5.2): actions come from the CEM planner
                # replanning against the LIVE regime-shifted state, toward a real
                # sample_dataset_init_goal pair (agent.set_goal above), at the
                # eval planner config. The trajectory contains the model's own
                # overshoot AND the correction it then attempts.
                #
                # v3 §5.2 fix: `collect_nas` (== --collect-num-act-stepped) is now
                # FUNCTIONAL -- the planner commits `collect_nas` model-chunks
                # before replanning, matching the nas=2 EVAL protocol. Previously
                # this loop replanned every chunk and discarded all but the first
                # planned chunk, so collection did not match eval CEM behaviour
                # (the C-1 mismatch).
                n_chunks = traj_len // frameskip
                # P4 / v3 §3.1: match run_e0_planning.py::run_episode's
                # `steps_left` convention EXACTLY. That loop passes a LOOSE
                # upper bound of (n_replans_target - replan_idx) * num_act_stepped
                # in MODEL-step units, with n_replans_target = raw_steps // nas.
                # CEM does plan_length = min(horizon, steps_left), so this keeps
                # every collection search at plan_length = horizon (6) — the same
                # lookahead as eval. Previously `n_chunks - chunk_idx` gave
                # 5/3/1, truncating plan_length below 6 for 100% of collected
                # steps (EVIDENCE_LEDGER §4 N5). The 5x unit inflation is
                # reproduced deliberately; it is the reference eval protocol and
                # must NOT be "corrected" here.
                n_replans_target = max((n_chunks * frameskip) // collect_nas, 1)
                for chunk_idx in range(0, n_chunks, collect_nas):
                    # Each agent.act() is a full CEM search (collect_num_samples
                    # x collect_iterations) -- the slowest step in this branch.
                    traj_pbar.set_postfix(chunk=f"{chunk_idx + 1}/{n_chunks}",
                                          contacts=total_contacts, attempt=attempt + 1)
                    obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
                    steps_left = max(
                        (n_replans_target - chunk_idx // collect_nas) * collect_nas, 1)
                    act_chunk = agent.act(obs_td, steps_left=steps_left)
                    # [t, frameskip*2] -> [(t f), 2] raw env actions.
                    act_chunk = rearrange(act_chunk.cpu(), "t (f d) -> (t f) d", d=2)
                    act_chunk = agent.preprocessor.denormalize_actions(act_chunk).numpy()
                    # execute up to `collect_nas` model-chunks, but never past
                    # traj_len (keep_idx subsample below assumes exactly traj_len
                    # raw steps) or past what the planner actually returned.
                    n_exec = min(frameskip * collect_nas,
                                 (n_chunks - chunk_idx) * frameskip,
                                 len(act_chunk))
                    for act in act_chunk[:n_exec]:
                        obs, reward, done, info = env.step(act)
                        total_contacts += info["n_contacts"]
                        imgs.append(obs["visual"])
                        proprios.append(obs["proprio"])
                        block_poses.append(np.asarray(info["block_pose"], dtype=np.float64))
                        raw_actions.append(act)
            else:
                block_xy = state[2:4]
                # One persistent random target near the block, sampled once per
                # trajectory — NOT resampled every step, which is what keeps the
                # walk "aimed" instead of cancelling out like the old scheme.
                target = block_xy + rs.uniform(-40.0, 40.0, size=(2,))
                agent_xy = obs["proprio"][:2]

                for _ in range(traj_len):
                    # Undo the env's own act*action_scale (+= agent.position when
                    # relative=True, the default) to get a raw action that aims
                    # the agent at `target` this step.
                    direction = (target - agent_xy) / env.action_scale
                    noise = rs.normal(0.0, 0.15, size=(2,))
                    # ACTION_GAIN: without this, "aim hard at the target every
                    # step" produces raw actions with std ~[0.46, 0.43] (often
                    # saturating at the +-1 clip bound while the agent is still
                    # far away) -- but the checkpoint's own preprocessor.action_std
                    # is ~[0.20, 0.20] (real demonstration data takes gentler,
                    # smaller per-step actions), so un-scaled actions land ~2.2x
                    # outside the distribution the model was calibrated on, which
                    # after normalize_actions stretches to std ~2.1-2.3 and values
                    # up to +-5 std. Confirmed via direct audit (code-review.md
                    # Bug #6d). ACTION_GAIN=0.25 was tuned empirically (clipping
                    # is nonlinear, so std doesn't scale linearly with gain) to
                    # bring raw action std to ~[0.215, 0.204], matching the
                    # checkpoint's ~[0.202, 0.200] almost exactly. This alone
                    # drops single-attempt contact rate to ~43%, but the
                    # rejection-sampling retry above (max_tries=8) still reaches
                    # ~99% expected overall success (1-(1-0.43)^8).
                    ACTION_GAIN = 0.25
                    act = np.clip((direction + noise) * ACTION_GAIN, -1.0, 1.0)
                    obs, reward, done, info = env.step(act)
                    agent_xy = obs["proprio"][:2]
                    total_contacts += info["n_contacts"]
                    imgs.append(obs["visual"])
                    proprios.append(obs["proprio"])
                    block_poses.append(np.asarray(info["block_pose"], dtype=np.float64))
                    raw_actions.append(act)

            if source == "closed_loop" or total_contacts > 0 or attempt == max_tries - 1:
                # closed_loop (v3 §5.2): NO contact-rejection. The collector is
                # goal-directed by the planner's cost function; a low/no-contact
                # trajectory means the planner legitimately struggled under the
                # shift -- real signal for training, not something to filter out.
                # Filtering it re-conditions the training distribution on contact
                # (the residual proxy this collector exists to remove).
                break  # accept

        imgs_np = np.stack(imgs, axis=0)            # [T_raw+1, 224, 224, 3]
        proprios_np = np.stack(proprios, axis=0)    # [T_raw+1, proprio_dim]
        acts_np = np.stack(raw_actions, axis=0)     # [T_raw, 2]

        # Subsample frames to the model time base: keep every `frameskip`-th
        # raw frame (plus frame 0) -> T_raw/frameskip + 1 frames, matching the
        # chunked actions below one-to-one.
        keep_idx = list(range(0, traj_len + 1, frameskip))
        imgs_sub = imgs_np[keep_idx]                # [T_model+1, 224, 224, 3]
        proprios_sub = proprios_np[keep_idx]        # [T_model+1, proprio_dim]
        block_poses_sub = np.stack(block_poses, axis=0)[keep_idx]  # [T_model+1, 3]

        # Encode visual+proprio TOGETHER via the wrapper's own encode() -- the
        # same obs-dict layout GC_Agent.act() feeds it (raw uint8 visual, raw
        # proprio; encode() does its own /255 + preprocessor.transform +
        # normalize_proprios).
        visual_t = torch.from_numpy(imgs_sub.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        proprio_t = torch.from_numpy(proprios_sub.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
            enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T_model+1, 256, 384]
            proprio_enc = enc["proprio"].squeeze(0)                     # [T_model+1, P_tok, D_p]

        # Chunk raw actions [T_raw, 2] -> model actions [T_raw/frameskip, 10].
        act_tensor_raw = torch.from_numpy(acts_np).float()
        act_norm = preprocessor.normalize_actions(act_tensor_raw.unsqueeze(0)).squeeze(0)  # [T_raw, 2]
        act_model = act_norm.reshape(traj_len // frameskip, frameskip * 2).to(device)      # [T_model, 10]

        trajectories.append({
            "encoder_output": enc_out,
            "actions": act_model,
            "proprio": proprio_enc,
            "seed": seed,  # the ACCEPTED seed (whichever attempt broke the retry loop) --
                            # written to e0_seed_manifest.json so E1/downstream experiments
                            # can be audited for zero overlap with E0's train/eval seeds.
            "episode_idx": episode_idx,  # source='dataset' only; None for 'scripted'
            "offset": offset,            # source='dataset' only; None for 'scripted'
            "n_contacts": total_contacts,  # informs the contact-rate check in main()
            **({"block_pose": block_poses_sub} if record_block_pose else {}),
        })

    return trajectories


def _traj_guard(args, regime: str) -> dict:
    """The protocol fingerprint a persisted trajectory file must match before
    --load-trajs will train on it (P9). Anything that changes the collected
    distribution goes here."""
    return {
        "source": args.data_source,
        "regime": regime,
        "regime_config": dict(REGIME_CONFIGS.get(regime, {})),
        "train_traj_len": args.train_traj_len,
        "eval_traj_len": args.eval_traj_len,
        "num_train_trajs": args.num_train_trajs,
        "num_val_trajs": args.num_val_trajs,
        "num_test_trajs": args.num_test_trajs,
        "collect_cem": (f"{args.collect_num_samples}x{args.collect_iterations} "
                        f"nas={args.collect_num_act_stepped}"
                        if args.data_source == "closed_loop" else None),
    }


def dump_regime_chunks(out_dir: Path, regime: str, trajs: list[dict], world_model,
                       collect_nas: int) -> None:
    """Emit chunks_{regime}.jsonl: every T=collect_nas sliding window with its
    UMF under the frozen c₀ predictor, its latent Frobenius displacement, and
    its block pixel displacement. This is the on-policy chunk artifact §6.1/§6.6
    re-derive τ and the motion gate from (P9 / v3 §5 deviation-note 1).

    Assumes world_model.model.predictor is in its pristine c₀ state (no chart
    applied) — true at the call site (collection plans against frozen c₀).
    """
    import numpy as np
    from atlas.score import rollout_umf

    path = out_dir / f"chunks_{regime}.jsonl"
    n = 0
    with open(path, "w") as f:
        for ti, traj in enumerate(trajs):
            enc = traj["encoder_output"]          # [T+1, N, D]
            acts = traj["actions"]                # [T, 10]
            bp = traj.get("block_pose")           # [T+1, 3] or None
            T = acts.shape[0]
            for i in range(0, T - collect_nas + 1):
                j = i + collect_nas
                proprio_ctxt = traj["proprio"][i:i + 1].unsqueeze(0)
                u = rollout_umf(world_model, enc[i:j + 1], acts[i:j],
                                proprio_ctxt=proprio_ctxt)
                lat = (enc[j] - enc[i]).norm(p="fro").item()
                blk = (float(np.linalg.norm(np.asarray(bp[j][:2]) - np.asarray(bp[i][:2])))
                       if bp is not None else None)
                f.write(json.dumps({
                    "regime": regime, "traj": ti, "window": [i, j],
                    "umf_c0": u, "latent_disp": lat, "block_disp_px": blk,
                }) + "\n")
                n += 1
    print(f"  [P9] wrote {n} T={collect_nas} chunks -> {path}", flush=True)


BLOCK_STATIC_PX = 1.0  # matches phase0_measure.py's existing "block-static" convention


def report_block_static_fraction(out_dir: Path, regime: str,
                                 split_trajs: dict[str, list[dict]]) -> dict | None:
    """Reports, per split (train/val/eval/test) and combined, what fraction of
    COLLECTED trajectories and T=nas chunks have block pixel displacement below
    BLOCK_STATIC_PX -- i.e. the block barely or never moved.

    This is pure REPORTING, not filtering: nothing here changes which
    trajectories/chunks train a chart or feed umf_scores. It exists because
    nobody currently knows this number -- e.g. --num-val-trajs 8 is silently
    "8 minus however many are dead", and that gap has never been measured.
    Written for closed_loop trajectories (record_block_pose=True); returns
    None and prints a note if block_pose is absent (other sources).

    Chunk-level static fraction is computed straight from a trajectory's own
    block_pose here (not read back from chunks_{regime}.jsonl, which only
    covers train+val) so it can also cover the test split.
    """
    import numpy as np

    if not any(t.get("block_pose") is not None for trajs in split_trajs.values() for t in trajs):
        print(f"  [Debug] block-static report for {regime}: SKIPPED (no block_pose recorded "
              f"-- only closed_loop trajectories carry it)", flush=True)
        return None

    report: dict = {"regime": regime, "block_static_px_threshold": BLOCK_STATIC_PX, "splits": {}}
    all_traj_static, all_chunk_static, all_traj_n, all_chunk_n = 0, 0, 0, 0

    for split, trajs in split_trajs.items():
        trajs_with_bp = [t for t in trajs if t.get("block_pose") is not None]
        if not trajs_with_bp:
            continue
        traj_static = 0
        chunk_n, chunk_static = 0, 0
        for t in trajs_with_bp:
            bp = t["block_pose"]  # [T_model+1, 3]
            whole_disp = float(np.linalg.norm(np.asarray(bp[-1][:2]) - np.asarray(bp[0][:2])))
            if whole_disp < BLOCK_STATIC_PX:
                traj_static += 1
        report["splits"][split] = {"n_trajs": len(trajs_with_bp), "n_traj_static": traj_static,
                                   "frac_traj_static": traj_static / len(trajs_with_bp)}
        # Chunk-level static fraction (T=nas windows) is reported separately by
        # derive_and_report_motion_gate() from chunks_{regime}.jsonl, which
        # currently covers train+val only (dump_regime_chunks is not called on
        # test) -- noted there, not duplicated here.
        all_traj_static += traj_static
        all_traj_n += len(trajs_with_bp)

    report["combined"] = {
        "n_trajs": all_traj_n, "n_traj_static": all_traj_static,
        "frac_traj_static": (all_traj_static / all_traj_n) if all_traj_n else None,
    }

    path = out_dir / f"block_static_{regime}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  [Debug] block-static report for {regime}: "
          f"{all_traj_static}/{all_traj_n} trajectories "
          f"({100 * all_traj_static / all_traj_n:.0f}%) have whole-trajectory block "
          f"displacement < {BLOCK_STATIC_PX}px (never moved) -> {path}", flush=True)
    return report


def derive_and_report_motion_gate(out_dir: Path, regime: str, chunks_path: Path,
                                  current_gate_10pct: float) -> dict | None:
    """v3 §6.6: derive the P95-over-block-static-chunks motion gate from the
    full on-policy chunk dump, alongside the retired 10th-percentile value and
    the REALISED false-pass rate of each rule -- i.e. what fraction of
    block-static chunks (block_disp_px < BLOCK_STATIC_PX, no real dynamics
    signal) would still be scored as "informative" (latent_disp > gate) under
    each candidate. This does NOT adopt a new gate value anywhere in the
    pipeline -- CLAUDE.md §15-5 already requires explicit human sign-off for
    the motion gate, and this function's whole job is to produce real evidence
    to sign off against instead of leaving §6.6 unmeasured. Additive-only:
    reads chunks_{regime}.jsonl (written by dump_regime_chunks), changes
    nothing else.
    """
    import numpy as np

    if not chunks_path.exists():
        print(f"  [Debug] §6.6 gate derivation for {regime}: SKIPPED "
              f"({chunks_path} not found)", flush=True)
        return None
    rows = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]
    if not rows:
        return None

    latent_disp = np.array([r["latent_disp"] for r in rows], dtype=float)
    block_disp = np.array([r["block_disp_px"] for r in rows if r["block_disp_px"] is not None],
                          dtype=float)
    static_mask = np.array([r["block_disp_px"] is not None and r["block_disp_px"] < BLOCK_STATIC_PX
                            for r in rows])
    n_static = int(static_mask.sum())

    gate_10pct = float(current_gate_10pct)
    gate_p95_static = (float(np.percentile(latent_disp[static_mask], 95))
                       if n_static > 0 else None)

    def false_pass_rate(gate: float | None) -> float | None:
        # A block-static chunk that still clears the gate (latent_disp > gate)
        # is a FALSE PASS: scored as informative despite carrying no real
        # block-motion signal (agent-only motion can still move the latent).
        if gate is None or n_static == 0:
            return None
        return float((latent_disp[static_mask] > gate).sum()) / n_static

    report = {
        "regime": regime, "n_chunks": len(rows), "n_block_static_chunks": n_static,
        "frac_block_static_chunks": n_static / len(rows),
        "gate_10pct_RETIRED": gate_10pct,
        "gate_10pct_false_pass_rate": false_pass_rate(gate_10pct),
        "gate_p95_over_block_static_v3_6_6": gate_p95_static,
        "gate_p95_false_pass_rate": false_pass_rate(gate_p95_static),
        "note": ("NEITHER value is adopted by this run -- CLAUDE.md §15-5 requires explicit "
                "human sign-off on the motion gate. This report exists to give that sign-off "
                "real evidence instead of a percentile heuristic. false_pass_rate = fraction "
                "of block-static chunks (block_disp_px < %.1fpx) that still clear the gate "
                "(latent_disp > gate) -- i.e. treated as informative despite no real block "
                "motion; lower is better." % BLOCK_STATIC_PX),
    }
    path = out_dir / f"gate_calibration_{regime}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  [Debug] §6.6 gate derivation for {regime}: n_static={n_static}/{len(rows)} "
          f"({100*n_static/len(rows):.0f}%) | 10pct={gate_10pct:.1f} "
          f"(false-pass {report['gate_10pct_false_pass_rate']}) | "
          f"P95-static={gate_p95_static} (false-pass {report['gate_p95_false_pass_rate']}) "
          f"-> {path}. NOT adopted -- needs human sign-off (§15-5).", flush=True)
    return report


def evaluate_e0_chart(world_model, chart: Chart, val_trajectories: list[dict],
                       motion_gate: float | None = None,
                       chunk_nas: int = 2,
                       chunk_motion_gate: float | None = None) -> dict:
    """
    Evaluates a fine-tuned chart on held-out validation trajectories.

    Args:
        world_model: EncPredWM wrapper (the object torch.hub.load returns —
                     NOT .model). _open_loop_rollout/umf need the wrapper for
                     the canonical unroll(); chart apply/restore and
                     cfgs_loss-based loss reach the inner VideoWM via
                     world_model.model.
        motion_gate: Informative-chunk threshold (gate G6) — see
                     atlas.score.compute_motion_gate. None = skip gate.
        chunk_nas:   Window size (model steps) for the additional T=chunk_nas
                     UMF (P8): the deployed loop and every Phase-0 threshold
                     (τ, motion gate, σ_r) score T=2 chunks, but the trajectory
                     rollout below is 5–6 model steps, so its UMF is on a
                     different scale from τ. Report both.
        chunk_motion_gate: gate for the T=chunk_nas windowed calls ONLY (§7-B2).
                     `motion_gate` is trajectory-scale and would reject nearly
                     every 2-step window. None → windowed calls are ungated.

    Returns a dict:
        loss                 — mean open-loop prediction loss over all trajs
        umf                  — mean GATED trajectory-T UMF (the historical
                               `eval_umf`; unchanged definition)
        umf_ungated          — mean trajectory-T UMF with NO motion gate (P10:
                               so the gate's effect on the mean is visible, not
                               baked in. Direction is NOT assumed here — an
                               earlier claim that gating is always optimistic
                               ["low displacement -> small denominator -> large
                               UMF"] was checked against real R2 chunks
                               2026-08-29 and found backwards on that sample
                               [corr(latent_disp, umf_c0) = +0.398,
                               phase0_v3/p0g_smoke_v3/chunks_R2.jsonl] — report
                               both numbers and let the reader compare, don't
                               assert a sign)
        umf_chunkT{n}        — mean GATED UMF over T=chunk_nas sliding windows
                               (P8: comparable to τ ≈ 0.262)
        umf_chunkT{n}_ungated
        n_trajs              — trajectory count
        n_umf                — trajs that passed the gate for `umf` (P10b:
                               `umf` and `loss` are means over different
                               subsets and nothing recorded which)
        n_umf_chunkT{n}      — informative windows for umf_chunkT{n}
        n_windows            — total T=chunk_nas windows considered
    """
    import numpy as np
    from atlas.harness import compute_trajectory_loss
    from atlas.score import _open_loop_rollout, _make_z_ctxt, umf

    losses, umf_gated, umf_ungated = [], [], []
    cw_gated, cw_ungated, n_windows = [], [], 0

    with torch.no_grad():
        for traj in val_trajectories:
            enc_out = traj["encoder_output"]
            actions = traj["actions"]
            z_vis = enc_out[0]
            proprio = traj["proprio"]
            proprio_ctxt = proprio[0:1].unsqueeze(0)  # [1, 1, P_tok, D_p]
            z_ctxt = _make_z_ctxt(world_model, z_vis, proprio_ctxt)

            # Apply chart specifically for the open-loop rollout, then restore.
            # apply_() is INSIDE the try too (not just the rollout) -- if
            # apply_() itself raises partway through (e.g. a lora4 chart with
            # a corrupted _param_names list), restore_() must still run or the
            # predictor is left with a permanent, never-cleaned-up partial
            # parametrization that corrupts every subsequent chart/fine-tune
            # sharing this predictor object. See code-review.md Bug #6f.
            try:
                chart.apply_(world_model.model.predictor)
                z_preds = _open_loop_rollout(world_model, z_ctxt, actions)
            finally:
                chart.restore_(world_model.model.predictor)

            loss = compute_trajectory_loss(world_model.model, z_preds, enc_out[1:])
            losses.append(loss.item())

            # umf internally handles applying and restoring the chart,
            # expecting the predictor to start in baseline state.
            s_gated = umf(chart, world_model, enc_out, actions, motion_gate=motion_gate,
                          proprio_ctxt=proprio_ctxt)
            if s_gated is not None:
                umf_gated.append(s_gated)
            s_ung = umf(chart, world_model, enc_out, actions, motion_gate=None,
                        proprio_ctxt=proprio_ctxt)
            if s_ung is not None:
                umf_ungated.append(s_ung)

            # P8: T=chunk_nas sliding windows on the same trajectory.
            T = actions.shape[0]
            for i in range(0, T - chunk_nas + 1):
                n_windows += 1
                j = i + chunk_nas
                pc = proprio[i:i + 1].unsqueeze(0)
                w_g = umf(chart, world_model, enc_out[i:j + 1], actions[i:j],
                          motion_gate=chunk_motion_gate, proprio_ctxt=pc)  # §7-B2
                if w_g is not None:
                    cw_gated.append(w_g)
                w_u = umf(chart, world_model, enc_out[i:j + 1], actions[i:j],
                          motion_gate=None, proprio_ctxt=pc)
                if w_u is not None:
                    cw_ungated.append(w_u)

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "loss": mean(losses),
        "umf": mean(umf_gated),
        "umf_ungated": mean(umf_ungated),
        f"umf_chunkT{chunk_nas}": mean(cw_gated),
        f"umf_chunkT{chunk_nas}_ungated": mean(cw_ungated),
        "n_trajs": len(val_trajectories),
        "n_umf": len(umf_gated),
        f"n_umf_chunkT{chunk_nas}": len(cw_gated),
        "n_windows": n_windows,
    }


def _umf_detail_fields(m: dict, motion_gate: float | None, args,
                        chunk_motion_gate: float | None = None) -> dict:
    """P8/P10/P10b: the extra UMF-provenance fields that travel with eval_umf in
    results.json. `m` is an evaluate_e0_chart() return dict (or {} on error)."""
    nas = args.collect_num_act_stepped
    return {
        "eval_umf_ungated": m.get("umf_ungated"),                     # P10
        f"eval_umf_chunkT{nas}": m.get(f"umf_chunkT{nas}"),           # P8 (τ-scale)
        f"eval_umf_chunkT{nas}_ungated": m.get(f"umf_chunkT{nas}_ungated"),
        "eval_n_trajs": m.get("n_trajs"),                             # P10b
        "eval_n_umf": m.get("n_umf"),
        f"eval_n_umf_chunkT{nas}": m.get(f"n_umf_chunkT{nas}"),
        "eval_n_windows": m.get("n_windows"),
        "motion_gate_value": (float(motion_gate) if motion_gate is not None else None),
        "motion_gate_chunk_value": (float(chunk_motion_gate)          # §7-B2
                                    if chunk_motion_gate is not None else None),
        "motion_gate_rule": ("RETIRED (P10): 10th-pct of train latent displacement "
                             "(score.compute_motion_gate default) — motion_gate_value "
                             "at T=train_traj_len//frameskip, motion_gate_chunk_value at "
                             f"T={nas} for the windowed UMF (§7-B2). v3 §6.6 replaces "
                             "both with P95 over block-static chunks — NOT yet applied."),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E0: Adapter capacity fine-tune.")
    parser.add_argument("--kinds", nargs="+", default=["ln_act", "lora4", "full"],
                        choices=["ln_act", "lora4", "full"])
    parser.add_argument("--regimes", nargs="+", default=["R1", "R2"])
    parser.add_argument("--regime-config", type=str, default=None,
                         help="JSON string overriding physics params for EVERY regime in "
                              "--regimes (P1 Step 2/P3), e.g. '{\"friction\": 2.0}'. Applied via "
                              "atlas.regimes.set_regime_config before trajectories are loaded, so "
                              "training and (this script's own) eval use the same calibrated "
                              "physics. All real invocations pass a single --regimes value; "
                              "applying one config to multiple regimes at once is not a supported "
                              "use case and is not validated. Logged into e0_seed_manifest.json "
                              "per regime so a chart's training physics stays attributable -- "
                              "run_e0_planning.py's own --regime-config must match this at eval "
                              "time or the comparison is a silent, invalidating mismatch.")
    parser.add_argument("--steps", type=int, default=2000,
                         help="MAXIMUM gradient steps -- early stopping (see --patience) can "
                              "stop sooner once validation loss stops improving.")
    parser.add_argument("--num-train-trajs", type=int, default=20,
                         help="Number of TRAINING trajectories. run_e0_finetune() loops over "
                              "every trajectory on every step, so compute (and, since all "
                              "trajectories' graphs are held simultaneously before one "
                              "accumulated backward, GPU memory) scales ~linearly with this. "
                              "Bumped from the original 3 now that --data-source=dataset (T9) "
                              "gives real, diverse trajectories instead of one scripted policy "
                              "-- intended for Modal (24GB), not the 6GB local card.")
    parser.add_argument("--train-traj-len", type=int, default=25,
                         help="Steps per TRAINING trajectory (must be a multiple of frameskip=5). "
                              "run_e0_finetune() backprops through the full open-loop unroll, so "
                              "GPU memory scales ~linearly with this too (measured ~0.27GB/step "
                              "for kind=full on this predictor) -- combined with the "
                              "--num-train-trajs bump above, this is sized for Modal's 24GB L4, "
                              "not local. See code-review.md Bug #6e.")
    parser.add_argument("--num-val-trajs", type=int, default=8,
                         help="Number of held-out validation trajectories, used for early "
                              "stopping during training (consulted up to ~80x at --eval-every 25 "
                              "--patience 5) AND, historically, for the final reported UMF -- a "
                              "checkpoint-selection bias FIX_SPEC.md A4 fixes by adding a "
                              "separate --num-test-trajs set. Bumped from the original hardcoded 2.")
    parser.add_argument("--num-test-trajs", type=int, default=8,
                         help="Number of held-out TEST trajectories (FIX_SPEC.md A4), drawn at "
                              "seed_offset=20_000 -- disjoint from both train (offset 0) and val "
                              "(offset 10_000). eval_umf is reported from THIS set; the val-set "
                              "number is also reported as val_umf so the checkpoint-selection "
                              "bias is measurable (test_umf - val_umf) rather than merely "
                              "asserted. Set to 0 to restore the old val-only behaviour.")
    parser.add_argument("--eval-traj-len", type=int, default=50,
                         help="Steps per EVAL (held-out) trajectory. Runs under torch.no_grad "
                              "so length is cheap here -- needs to be long because 10 was too "
                              "short for properly-calibrated (gentle) actions to accumulate "
                              "enough displacement for UMF to be well-behaved. See "
                              "code-review.md Bug #6d. DINO-WM's own training trajectories "
                              "are ~100 steps.")
    parser.add_argument("--eval-every", type=int, default=25,
                         help="Gradient steps between early-stopping validation checks "
                              "(E0_IMPLEMENTATION_PLAN.md T9).")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop after this many consecutive validation checks with no "
                              "improvement (E0_IMPLEMENTATION_PLAN.md T9) -- the overfitting fix: "
                              "`full` previously reached train loss 0.0015 over 2000 steps on 30 "
                              "transitions with no validation signal at all.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--data-source", choices=["scripted", "dataset", "hybrid", "closed_loop"], default="dataset",
                         help="'dataset' (default, T9): replay real Push-T demo action "
                              "sequences from data/pusht_noise/{--data-split}/ under the "
                              "shifted regime -- OPEN-LOOP: actions never react to the "
                              "shifted physics. 'hybrid' (P2b): real dataset init state, but "
                              "actions are the 'scripted' aimed-walk policy driven by the LIVE "
                              "post-shift agent position each step -- CLOSED-LOOP, real/diverse "
                              "start states. 'scripted': the original synthetic aimed-walk "
                              "sampler with a random reset (documented fallback per "
                              "E0_IMPLEMENTATION_PLAN.md T9 -- use if the replay path proves "
                              "too slow). 'closed_loop': ON-POLICY -- real (init, goal) pair "
                              "from run_e0_planning.py's own sampler, actions produced by the "
                              "CEM planner replanning every --collect-num-act-stepped model "
                              "chunks against the live shifted state, at eval-matched "
                              "lookahead + goal separation (v3 §3.1/§3.2). The only source "
                              "whose trajectories contain the model's own overshoot AND its "
                              "attempted correction; 'hybrid' is reactive but its corrections "
                              "come from a scripted policy, not from the model being adapted. "
                              "Costs one CEM search per replan -- see --collect-num-samples.")
    parser.add_argument("--debug-predictor-fingerprint", action="store_true",
                         help="Print a sha256 of predictor params before each regime's "
                              "collection (P1 falsification test): with --regimes R0,R2 the "
                              "fingerprints must be IDENTICAL after the fix, DIFFERENT before.")
    parser.add_argument("--collect-traj-offset", type=int, default=0,
                         help="Sharding (mirrors run_e0_planning.py's --episode-start pattern, "
                              "modal_e0_planning.py's --num-shards): shifts the TRAIN seed space "
                              "so N concurrent shard containers each collecting num_train_trajs "
                              "trajectories with offsets 0, k, 2k, ... draw disjoint seeds instead "
                              "of duplicating each other's work. Not applied to val/test -- see "
                              "--collect-skip-val-test. Merge shard outputs with "
                              "scripts/merge_p0g_shards.py. Default 0 = unsharded, unchanged.")
    parser.add_argument("--collect-skip-val-test", action="store_true",
                         help="Sharding: skip collecting val/test trajectories entirely (they're "
                              "cheap and NOT split across shards -- exactly one shard, typically "
                              "--collect-traj-offset 0, should omit this flag so it collects them; "
                              "every other shard passes it).")
    parser.add_argument("--collect-only", action="store_true",
                         help="Collect + persist trajectories (and the chunks_{regime}.jsonl "
                              "dump) then EXIT before any fine-tuning (P2 — splits the Modal "
                              "collection job from the fine-tune job so a fine-tune failure "
                              "cannot destroy the collection).")
    parser.add_argument("--load-trajs", type=Path, default=None,
                         help="Directory containing trajs_{regime}.pt files from a prior run "
                              "(P9). When set, trajectories are LOADED instead of collected — "
                              "the expensive CEM collection is paid once and every fine-tune "
                              "re-run is then nearly free. Refuses to run if the stored "
                              "protocol fingerprint (_traj_guard) does not match the current "
                              "args. Also auto-detected in --out on resume.")
    parser.add_argument("--data-split", type=str, default="train",
                         help="data/pusht_noise/{split}/ to draw real episodes from "
                              "(--data-source=dataset, hybrid or closed_loop only).")
    parser.add_argument("--collect-num-samples", type=int, default=300,
                         help="CEM population for --data-source=closed_loop COLLECTION only. "
                              "SUBMISSION_PLAN.md E-C: defaults to the eval-side budget (300, "
                              "matching run_e0_planning.py's own --num-samples default) so "
                              "collection and evaluation are budget-matched. Previously "
                              "hardcoded to 100 (a ~9x cheaper, deliberately mismatched budget "
                              "-- see FIXLOG.md E-C). Override to reproduce the old, cheaper, "
                              "mismatched collection run if needed.")
    parser.add_argument("--collect-iterations", type=int, default=30,
                         help="CEM iterations for closed_loop collection. SUBMISSION_PLAN.md "
                              "E-C: defaults to the eval-side budget (30, matching "
                              "run_e0_planning.py's own --iterations default). Previously "
                              "hardcoded to 10.")
    parser.add_argument("--collect-num-act-stepped", type=int, default=2,
                         help="num_act_stepped for the closed_loop collector. FUNCTIONAL "
                              "since v3 §5.2 (P21): the collector loop replans every "
                              "`collect_nas` model-chunks and executes all of them, and "
                              "`steps_left` now matches run_e0_planning.py's convention so "
                              "plan_length stays at horizon=6 — collection CEM cadence + "
                              "lookahead now match eval. Default 2 per IMPLEMENTATION_PLAN_V3 "
                              "§3.6 (the eval protocol). The old 'this flag is a no-op' "
                              "caveat is obsolete.")
    parser.add_argument("--collect-max-tries", type=int, default=2,
                         help="Contact-rejection retries per trajectory for closed_loop. Much "
                              "lower than the default 8: each retry is a full planned episode, "
                              "and a goal-directed planner should make contact far more often "
                              "than the scripted sampler it was tuned for.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="atlas-e0", help="WandB project name")
    return parser


def _reported_train_loss(out_dir: Path, kind: str, regime: str, loss_log: list) -> float:
    """The train loss that should be reported alongside eval_loss/eval_umf.
    run_e0_finetune() returns the BEST-VAL checkpoint's weights, not the final
    step's -- so the final entry of loss_{kind}_{regime}.json describes a
    DIFFERENT (more overfit) snapshot than the one actually evaluated. Prefer
    val_loss_{kind}_{regime}.json's best_train_loss (the train loss recorded
    at the exact step best_params was captured); fall back to the final loss
    only if that file doesn't exist (e.g. no val_trajectories were used)."""
    val_loss_file = out_dir / f"val_loss_{kind}_{regime}.json"
    if val_loss_file.exists():
        d = json.loads(val_loss_file.read_text())
        if d.get("best_train_loss") is not None:
            return d["best_train_loss"]
    return loss_log[-1] if loss_log else float("nan")


def compute_motion_gates(train_trajectories: list[dict], nas: int,
                         verbose_label: str = "") -> tuple[float, float | None]:
    """Gate G6 (motion_gate) + §7-B2 (chunk_motion_gate), factored out of
    main() so scripts/merge_p0g_shards.py can recompute both on a MERGED
    train set (sharded collection must not report per-shard gate values —
    they're calibrated over whatever trajectories happen to land in that
    shard, not the full requested set)."""
    train_displacements = torch.tensor([
        (t["encoder_output"][-1] - t["encoder_output"][0]).norm(p="fro").item()
        for t in train_trajectories
    ])
    motion_gate = compute_motion_gate(train_displacements)
    print(f"  [Debug] motion_gate (10th pct of T={train_trajectories[0]['actions'].shape[0]}-step "
          f"train displacement){' for ' + verbose_label if verbose_label else ''} = "
          f"{motion_gate:.4f}", flush=True)

    # §7-B2: the T=nas windowed UMF (P8) must be gated at the SAME granularity
    # it is applied at. `motion_gate` above is the 10th pct of whole-trajectory
    # displacement; a 2-step window's displacement is far smaller, so that
    # threshold rejects almost every window (§6.6's named failure). Same
    # RETIRED 10th-pct rule, but at T=nas. (v3 §6.6's block-static P95
    # replacement is a separate Phase-0 task, fed by chunks_{regime}.jsonl.)
    chunk_displacements = torch.tensor([
        (t["encoder_output"][i + nas] - t["encoder_output"][i]).norm(p="fro").item()
        for t in train_trajectories
        for i in range(0, t["actions"].shape[0] - nas + 1)
    ])
    if chunk_displacements.numel() == 0:
        print(f"  [Debug] chunk_motion_gate: SKIPPED (no T={nas} windows)", flush=True)
        return motion_gate, None
    chunk_motion_gate = compute_motion_gate(chunk_displacements)
    print(f"  [Debug] chunk_motion_gate (10th pct, T={nas}){' for ' + verbose_label if verbose_label else ''} "
          f"= {chunk_motion_gate:.4f} (vs traj-scale {motion_gate:.1f})", flush=True)
    return motion_gate, chunk_motion_gate


def main() -> None:
    args = _build_parser().parse_args()

    _determinism.make_deterministic(0)  # cuBLAS/cuDNN determinism (v3 P0-G)

    wandb_group = f"e0_experiment_{int(time.time())}"

    print("Loading dino_wm_pusht from local hub...")
    # Raw ATLAS_HOME env var, not Path(__file__)-relative: on Modal the code
    # (/src) and the volume-mounted hub cache (/atlas_root/hub) live at
    # different paths -- see run_e0_planning.py's HUB_PATH for the same fix.
    import os
    _atlas_home = os.environ.get("ATLAS_HOME", str(atlas.ATLAS_HOME))
    hub_path = str(Path(_atlas_home) / "hub" / "hub" / "facebookresearch_jepa-wms_main")
    model, prep = torch.hub.load(
        hub_path, "dino_wm_pusht",
        source="local",
        force_reload=False, trust_repo=True,
    )
    # `model` is the EncPredWM wrapper (owns ctxt_window/proprio_mode and the
    # canonical unroll() this checkpoint was validated with -- see
    # E0_IMPLEMENTATION_PLAN.md T1/T2). `wm` is the inner VideoWM, needed for
    # predictor state-dict ops, Chart construction, and encoder freezing --
    # wrapper.to(device) moves it too, since it's a registered submodule.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = model.to(device)
    wm = wrapper.model if hasattr(wrapper, "model") else wrapper
    for p in wm.encoder.parameters():
        p.requires_grad_(False)

    # T7 throughput fix (E0_IMPLEMENTATION_PLAN.md), never ported here from
    # run_e0_planning.py: SDPA is absent from the eval YAML so it defaults
    # False, falling back to manual attention that materialises the full
    # attention matrix in fp32 every layer -- measured to cost 7.4-7.5s/step
    # on T4 for a 10.7k-param ln_act fine-tune, an order of magnitude slower
    # than this workload should need. Enabled post-load, not via YAML.
    for m in wm.predictor.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    # Pristine snapshot of the predictor, taken once before any fine-tuning.
    # run_e0_finetune() mutates wm.predictor in-place for ln_act/full charts and
    # never restores it (Chart.restore_ just re-applies the already-updated chart
    # params, not the original checkpoint weights) — without reloading this
    # snapshot before every fine-tune, later charts would train on top of earlier
    # charts' weights instead of the pretrained baseline. See code-review.md #1.
    pristine_predictor_state = copy.deepcopy(wm.predictor.state_dict())

    if args.regime_config is not None:
        resolved_cfg = json.loads(args.regime_config)
        for regime in args.regimes:
            set_regime_config(regime, resolved_cfg)

    # Built from the PRISTINE predictor, before any chart is applied: the point
    # of on-policy collection is data showing what the FROZEN model gets wrong,
    # which is what the chart is then fit to correct. Collecting under an
    # already-adapted model would be a second DAgger round, not this experiment.
    collector_agent = None
    if args.data_source == "closed_loop":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from run_e0_planning import build_cfg
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        # v3 §5.2: num_act_stepped is threaded into the agent's cfg AND is now
        # FUNCTIONAL in the collection loop (load_regime_trajectories' closed_loop
        # branch replans every `collect_nas` model-chunks, executing all of them).
        collector_agent = GC_Agent(
            build_cfg(args.collect_num_samples, args.collect_iterations, horizon=6,
                      num_act_stepped=args.collect_num_act_stepped),
            wrapper, dset=None, preprocessor=prep)
        collector_agent.device = device
        print(f"closed_loop collection: CEM {args.collect_num_samples}x"
              f"{args.collect_iterations}, num_act_stepped={args.collect_num_act_stepped}, "
              f"contact filter OFF (v3 §5.2)", flush=True)

    # [Debug print statement] Print setup info
    print(f"\nE0: {len(args.kinds)} kinds × {len(args.regimes)} regimes "
          f"= {len(args.kinds) * len(args.regimes)} fine-tunes", flush=True)
    print(f"Output: {args.out}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    seen_seeds: dict = {}     # P19: seed -> "regime/split", across all regimes this run
    seed_manifest: dict = {}  # regime -> {"train": [...], "eval": [...]} accepted seeds --
                              # written to e0_seed_manifest.json so downstream experiments
                              # (E1) can be audited for zero seed overlap with E0.

    for regime in args.regimes:
        results[regime] = {}
        # [Debug print statement] Print regime start
        print(f"\n── Regime {regime} ─────────────────────────────────────────────", flush=True)

        # P1 / v3 §5.2: on-policy collection MUST plan against the frozen c₀
        # predictor. run_e0_finetune()'s Chart.restore_() re-applies the chart's
        # TRAINED weights for ln_act/full (atlas/chart.py:126-127), and the
        # per-kind pristine reload lives INSIDE the kind loop below — so with
        # --regimes R0,R2 in one process, R2 would be collected under an
        # R0-adapted predictor. Reload pristine c₀ here, before collection.
        # collector_agent holds `wrapper`; wrapper.model IS `wm`, so this
        # in-place load_state_dict is seen by the agent (same object).
        wm.predictor.load_state_dict(pristine_predictor_state)
        if args.debug_predictor_fingerprint:
            import hashlib
            fp = hashlib.sha256(b"".join(
                p.detach().cpu().numpy().tobytes() for p in wm.predictor.parameters()
            )).hexdigest()[:16]
            print(f"  [P1] predictor fingerprint before {regime} collection: {fp}", flush=True)

        print(f"  Loading trajectories for regime {regime}...", flush=True)
        # Separate lengths: training backprops through the full unroll (memory
        # scales with length, kept short); eval runs under no_grad (cheap, kept
        # long since UMF needs real accumulated displacement). See --train-traj-len
        # / --eval-traj-len help text and code-review.md Bug #6d/#6e.
        collect_kw = ({"agent": collector_agent, "max_tries": args.collect_max_tries,
                       "collect_nas": args.collect_num_act_stepped,
                       "record_block_pose": True}
                      if args.data_source == "closed_loop" else {})

        # P9 / P2c: load persisted trajectories if a prior run collected them,
        # so re-runs of the fine-tune do not re-pay the CEM collection cost. A
        # resumed run also lands here (trajs_{regime}.pt in --out) instead of
        # re-collecting ~2h of trajectories to skip a cached chart.
        guard = _traj_guard(args, regime)
        traj_file = args.out / f"trajs_{regime}.pt"
        src_file = None
        if args.load_trajs is not None and (args.load_trajs / f"trajs_{regime}.pt").exists():
            src_file = args.load_trajs / f"trajs_{regime}.pt"
        elif traj_file.exists():
            src_file = traj_file

        if src_file is not None:
            # map_location="cpu": a persisted file's tensors may have been
            # saved from a GPU collection container OR re-saved by a CPU-only
            # merge (scripts/merge_p0g_shards.py, which itself loads with
            # map_location="cpu" -- see that script's comment) -- don't assume
            # either. Load onto CPU unconditionally, then move to `device`
            # explicitly below, so this path is correct regardless of the
            # file's provenance. Missing this caused a real crash: fine-tuning
            # from a merged (CPU-saved) trajectory file against a CUDA
            # predictor raised "Expected all tensors to be on the same
            # device, cpu and cuda:0" inside the first forward pass.
            blob = torch.load(src_file, map_location="cpu", weights_only=False)
            if blob["guard"] != guard:
                raise ValueError(
                    f"--load-trajs protocol mismatch for {regime}:\n"
                    f"  stored:  {blob['guard']}\n  current: {guard}\n"
                    f"Refusing to train on a different protocol's data.")

            def _to_device(trajs: list[dict]) -> list[dict]:
                for t in trajs:
                    for k in ("encoder_output", "actions", "proprio"):
                        if k in t and torch.is_tensor(t[k]):
                            t[k] = t[k].to(device)
                return trajs
            train_trajectories = _to_device(blob["train"])
            val_trajectories = _to_device(blob["val"])
            test_trajectories = _to_device(blob["test"])
            print(f"  [P9] loaded {len(train_trajectories)}/{len(val_trajectories)}/"
                  f"{len(test_trajectories)} train/val/test trajectories from {src_file} "
                  f"(NO collection, moved to {device})", flush=True)
        else:
            train_trajectories = load_regime_trajectories(
                wrapper, prep, regime, num_trajs=args.num_train_trajs, traj_len=args.train_traj_len,
                device=device, seed_offset=0, source=args.data_source, data_split=args.data_split,
                traj_idx_offset=args.collect_traj_offset, **collect_kw)
            # Sharding (--collect-skip-val-test): val/test are cheap (8 each vs
            # 100 train) and NOT split across shards -- one shard (offset=0)
            # collects them, the rest skip, and merge_p0g_shards.py takes them
            # from that shard. traj_idx_offset is NOT applied to val/test (they
            # always start their own seed space at 0; only relevant when this
            # shard is the one collecting them at all).
            if args.collect_skip_val_test:
                val_trajectories, test_trajectories = [], []
            else:
                val_trajectories = load_regime_trajectories(
                    wrapper, prep, regime, num_trajs=args.num_val_trajs, traj_len=args.eval_traj_len, device=device,
                    seed_offset=10_000, source=args.data_source, data_split=args.data_split,
                    **collect_kw)
                # FIX_SPEC.md A4: disjoint TEST set (seed_offset=20_000) -- the number
                # actually reported as eval_umf, so early-stopping's repeated use of the
                # val set does not bias it.
                test_trajectories = load_regime_trajectories(
                    wrapper, prep, regime, num_trajs=args.num_test_trajs, traj_len=args.eval_traj_len,
                    device=device, seed_offset=20_000, source=args.data_source,
                    data_split=args.data_split, **collect_kw) if args.num_test_trajs > 0 else []
            # Persist immediately (encoder_output kept fp32 — measure size on the
            # smoke before switching to .half()). Then re-runs load instead of
            # re-collecting.
            torch.save({"guard": guard, "train": train_trajectories,
                        "val": val_trajectories, "test": test_trajectories},
                       traj_file)
            print(f"  [P9] persisted trajectories -> {traj_file} "
                  f"({traj_file.stat().st_size / 1e6:.0f} MB)", flush=True)
            if args.data_source == "closed_loop":
                dump_regime_chunks(args.out, regime, train_trajectories + val_trajectories,
                                   wrapper, args.collect_num_act_stepped)
                # Distribution reporting, NOT filtering (per user direction 2026-08-29):
                # (1) what fraction of collected trajectories/chunks never moved the
                #     block at all -- answers Part 3 Q8 honestly ("8 val trajectories"
                #     is really "8 minus the dead fraction", previously unmeasured).
                report_block_static_fraction(args.out, regime, {
                    "train": train_trajectories, "val": val_trajectories,
                    "test": test_trajectories,
                })
        def _manifest_rows(trajs):  # P12: n_contacts was stdout-only before
            return [{"seed": t["seed"], "episode_idx": t["episode_idx"],
                     "offset": t["offset"], "n_contacts": t.get("n_contacts")}
                    for t in trajs]
        seed_manifest[regime] = {
            "source": args.data_source,
            "regime_config": dict(REGIME_CONFIGS.get(regime, {})),
            "train": _manifest_rows(train_trajectories),
            "eval": _manifest_rows(val_trajectories),
            "test": _manifest_rows(test_trajectories),
        }
        # P19: seeds must be disjoint within a regime AND across regimes (they
        # collide silently at num_trajs >= 501 for closed_loop). Assert it.
        _seed_sets = {k: {r["seed"] for r in seed_manifest[regime][k]}
                      for k in ("train", "eval", "test")}
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            dup = _seed_sets[a] & _seed_sets[b]
            assert not dup, f"P19: {regime} {a}/{b} seed overlap: {sorted(dup)[:5]}"
        for k, s in _seed_sets.items():
            clash = {sd: seen_seeds[sd] for sd in s if sd in seen_seeds}
            assert not clash, f"P19: {regime}/{k} seeds already used by {clash}"
            seen_seeds.update({sd: f"{regime}/{k}" for sd in s})
        # [Debug print statement] Print trajectories loaded
        print(f"  [Debug] Loaded {len(train_trajectories)} train & {len(val_trajectories)} eval trajectories for {regime}", flush=True)

        if args.data_source in ("dataset", "hybrid", "closed_loop"):
            # T9 acceptance check: real demo episodes should generally involve
            # agent-block contact (they're expert task completions), but verify
            # rather than assume -- especially post-regime-shift, where the
            # replayed actions no longer perfectly track the (now-different)
            # physics. n_contacts is per-trajectory total contact events across
            # traj_len steps, recorded by load_regime_trajectories() above.
            all_trajs = train_trajectories + val_trajectories
            n_with_contact = sum(1 for t in all_trajs if t["n_contacts"] > 0)
            _kind = ("on-policy planner" if args.data_source == "closed_loop"
                     else "real-demo replay")  # P12: label was always "Real-demo replay"
            print(f"  [Debug] {_kind} contact rate for {regime}: "
                  f"{n_with_contact}/{len(all_trajs)} trajectories had >=1 contact "
                  f"(n_contacts per traj: {[t['n_contacts'] for t in all_trajs]}). "
                  f"§15-2 pre-registered R2 check: if this collapses toward 0, "
                  f"damping=0.1 is the fallback.", flush=True)

        motion_gate, chunk_motion_gate = compute_motion_gates(
            train_trajectories, args.collect_num_act_stepped, verbose_label=regime)

        if args.data_source == "closed_loop":
            # v3 §6.6's gate derived from the real on-policy chunk dump, reported
            # alongside the retired 10th-pct value and each rule's realised
            # false-pass rate, for explicit human sign-off (§15-5). NOT adopted here.
            derive_and_report_motion_gate(args.out, regime,
                                          args.out / f"chunks_{regime}.jsonl", chunk_motion_gate)

        if args.collect_only:
            print(f"  [P2] --collect-only: trajectories for {regime} persisted, "
                  f"skipping fine-tune.", flush=True)
            continue

        for kind in args.kinds:
            # Reset the predictor to its pristine pretrained state before every
            # kind's fine-tune/eval. Without this, later charts train on top of
            # earlier charts' weights instead of the checkpoint baseline — see
            # code-review.md #1.
            wm.predictor.load_state_dict(pristine_predictor_state)

            loss_file = args.out / f"loss_{kind}_{regime}.json"
            chart_file = args.out / f"chart_{kind}_{regime}.pt"

            if loss_file.exists() and chart_file.exists():
                # [Resume support] Skip fine-tuning if results already exist on disk / volume
                print(f"  ⏩ [Resume] {kind}_{regime} already completed. Loading cached result...", flush=True)
                losses = json.loads(loss_file.read_text())
                final_loss = _reported_train_loss(args.out, kind, regime, losses)
                try:
                    chart = Chart.load(chart_file, wm.predictor)
                    n_params = chart.n_params()  # real parameter count, not a tensor count (T12 #9)
                    n_trainable = chart.n_trainable_params()  # FIX_SPEC.md A11
                    val_m = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate,
                                              args.collect_num_act_stepped, chunk_motion_gate)
                    if test_trajectories:
                        eval_m = evaluate_e0_chart(wrapper, chart, test_trajectories, motion_gate,
                                                   args.collect_num_act_stepped, chunk_motion_gate)
                        eval_umf_source = "test"
                    else:
                        eval_m = val_m
                        eval_umf_source = "val_ALIASED"  # P3: no disjoint test split
                    val_umf = val_m["umf"]
                    eval_loss, eval_umf = eval_m["loss"], eval_m["umf"]
                except Exception as e:
                    # Print the real error instead of silently returning NaN --
                    # a swallowed exception here previously masked a real bug
                    # (Chart.load()'s corrupted _param_names for lora4, see
                    # code-review.md Bug #6f) that also left the shared
                    # predictor in a corrupted state for every subsequent
                    # kind/regime, only surfacing as a confusing crash much
                    # later. Reset the predictor defensively so a failure here
                    # can't cascade the same way even if a new bug appears.
                    print(f"    ⚠️ [Resume] Failed to re-evaluate cached {kind}_{regime}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    wm.predictor.load_state_dict(pristine_predictor_state)
                    n_params = 0
                    n_trainable = 0
                    val_umf = float("nan")
                    eval_loss, eval_umf = float("nan"), float("nan")
                    eval_m = {}
                    eval_umf_source = "error"
                results[regime][kind] = {
                    "train_loss": final_loss,
                    "eval_loss": eval_loss,
                    "eval_umf": eval_umf,
                    "eval_umf_source": eval_umf_source,  # P3: "test" (disjoint, A4) or
                                                          # "val_ALIASED" (eval_umf==val_umf,
                                                          # selection bias NOT measurable)
                    "val_umf": val_umf,
                    "params": n_params,
                    "params_stored": n_params,
                    "params_trainable": n_trainable,
                    "status": "completed (cached)",
                    **_umf_detail_fields(eval_m, motion_gate, args, chunk_motion_gate),  # P8/P10/P10b/B2
                }
                print(f"    Done {kind}_{regime} (Cached): Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)
                continue

            # Initialize separate WandB run for this fine-tuning session under the shared group
            if args.wandb:
                try:
                    import wandb
                    wandb.init(
                        project=args.wandb_project,
                        group=wandb_group,
                        name=f"{kind}_{regime}",
                        reinit=True,
                        config={
                            "kind": kind,
                            "regime": regime,
                            "steps": args.steps,
                            "lr": args.lr,
                        },
                    )
                    wandb.define_metric("step")
                    wandb.define_metric("loss", step_metric="step")
                    wandb.define_metric(f"loss_{kind}_{regime}", step_metric="step")
                except Exception as e:
                    print(f"⚠️ WandB init failed for {kind}_{regime}: {e}", flush=True)

            # [Debug print statement] Print fine-tuning start
            print(f"  Fine-tuning {kind} on {regime} (up to {args.steps} steps, "
                  f"early stop after {args.patience} checks w/o improvement every "
                  f"{args.eval_every} steps)...", flush=True)
            chart = run_e0_finetune(
                world_model=wrapper,
                trajectories=train_trajectories,
                kind=kind,
                regime=regime,
                n_steps=args.steps,
                lr=args.lr,
                out_dir=args.out,
                val_trajectories=val_trajectories,
                eval_every=args.eval_every,
                patience=args.patience,
            )

            # Held-out Evaluation. val_* drives nothing here (early stopping
            # already used it); test_* (disjoint seed_offset=20_000) is the
            # reported eval_umf per FIX_SPEC.md A4.
            val_m = evaluate_e0_chart(wrapper, chart, val_trajectories, motion_gate,
                                      args.collect_num_act_stepped, chunk_motion_gate)
            if test_trajectories:
                eval_m = evaluate_e0_chart(wrapper, chart, test_trajectories, motion_gate,
                                           args.collect_num_act_stepped, chunk_motion_gate)
                eval_umf_source = "test"
            else:
                # P3: NO disjoint test split -> eval_umf is just the early-stopping
                # selection number. The A4 bias (eval_umf - val_umf) is then
                # identically 0 by construction, NOT a measured +0.0. p0g_collect
                # now passes --num-test-trajs 8 so this branch should not be hit
                # in P0-G; kept for --num-test-trajs 0 back-compat.
                eval_m = val_m
                eval_umf_source = "val_ALIASED"
            val_loss, val_umf = val_m["loss"], val_m["umf"]
            eval_loss, eval_umf = eval_m["loss"], eval_m["umf"]

            if args.wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"eval_loss": eval_loss, "eval_umf": eval_umf})
                        wandb.finish()
                except Exception:
                    pass
            
            # Compute evaluation metrics
            losses = json.loads(loss_file.read_text())
            final_loss = _reported_train_loss(args.out, kind, regime, losses)
            results[regime][kind] = {
                "train_loss": final_loss,
                "eval_loss": eval_loss,
                "eval_umf": eval_umf,          # test set (A4) IFF eval_umf_source=="test"
                "eval_umf_source": eval_umf_source,  # "test" | "val_ALIASED" (P3)
                "val_umf": val_umf,            # early-stopping set; bias = eval_umf - val_umf
                                               # is meaningful ONLY when source=="test"
                "params": chart.n_params(),
                "params_stored": chart.n_params(),           # FIX_SPEC.md A11
                "params_trainable": chart.n_trainable_params(),
                "status": "completed",
                **_umf_detail_fields(eval_m, motion_gate, args, chunk_motion_gate),  # P8/P10/P10b/B2
            }
            # [Debug print statement] Print fine-tuning & eval completed
            print(f"    Done {kind}_{regime}: Train Loss = {final_loss:.6f} | Eval Loss = {eval_loss:.6f} | Eval UMF = {eval_umf:.4f}", flush=True)

    # Save summary results
    results_json = args.out / "results.json"
    results_json.write_text(json.dumps(results, indent=2))

    # Seed manifest: audit trail proving E1 (or any downstream experiment) uses
    # seeds disjoint from every trajectory that went into fine-tuning/evaluating
    # these charts. E1's own seeds (atlas.streams.paired_seed, SHA256-derived)
    # live in a completely different space from these small deterministic
    # integers, but this manifest makes that an auditable fact, not a claim.
    seed_manifest["_provenance"] = _determinism.settings_dict(0)
    seed_manifest["_provenance"].update(
        data_source=args.data_source,
        collect_cem=(f"{args.collect_num_samples}x{args.collect_iterations} "
                     f"nas={args.collect_num_act_stepped}"
                     if args.data_source == "closed_loop" else None),
        contact_filter=("OFF" if args.data_source == "closed_loop" else "ON"),
    )
    seed_manifest_json = args.out / "e0_seed_manifest.json"
    seed_manifest_json.write_text(json.dumps(seed_manifest, indent=2))
    print(f"  - Seed manifest : {seed_manifest_json}")

    # Generate Markdown Table T5
    md_lines = [
        "# Table T5: E0 Adapter Capacity Benchmarks",
        "",
        "| Regime | Adapter Kind | Target Params | Train Loss | Eval Loss | Eval UMF | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime, kinds in results.items():
        for kind, metrics in kinds.items():
            md_lines.append(
                f"| {regime} | `{kind}` | {metrics['params']:,} | {metrics['train_loss']:.6f} | {metrics['eval_loss']:.6f} | {metrics['eval_umf']:.4f} | {metrics['status']} |"
            )
    
    results_md = args.out / "results.md"
    results_md.write_text("\n".join(md_lines))

    # Generate local loss curves plot from saved JSON files
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4), dpi=300)
        colors = {
            "ln_act": "#1f77b4",
            "lora4": "#ff7f0e",
            "full": "#2ca02c",
        }
        linestyles = {
            "R1": "-",
            "R2": "--",
        }

        for regime in args.regimes:
            for kind in args.kinds:
                loss_file = args.out / f"loss_{kind}_{regime}.json"
                if loss_file.exists():
                    losses = json.loads(loss_file.read_text())
                    steps = list(range(1, len(losses) + 1))
                    plt.plot(
                        steps,
                        losses,
                        label=f"{kind} ({regime})",
                        color=colors.get(kind, None),
                        linestyle=linestyles.get(regime, "-"),
                        alpha=0.85,
                        linewidth=1.5,
                    )

        plt.xlabel("Gradient Step (1..2000)")
        plt.ylabel("Prediction Loss")
        plt.title("E0 Adapter Capacity: Fine-Tuning Loss Curves")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        plot_png = args.out / "e0_loss_curves.png"
        plot_pdf = args.out / "e0_loss_curves.pdf"
        plt.savefig(plot_png)
        plt.savefig(plot_pdf)
        plt.close()
        print(f"  - Loss Plot PNG: {plot_png}")
        print(f"  - Loss Plot PDF: {plot_pdf}")
    except Exception as e:
        print(f"⚠️ Loss plot generation skipped: {e}")

    print(f"\n✅ E0 Experiment complete! Results saved to {args.out}")
    print(f"  - Summary JSON : {results_json}")
    print(f"  - Table T5 MD  : {results_md}")


if __name__ == "__main__":
    main()

