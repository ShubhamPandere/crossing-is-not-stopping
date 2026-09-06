"""
scripts/diagnose_cem_costs.py -- cost-ranking diagnostic (C3 validation):
does a chart's UMF improvement track a BETTER CEM cost ranking, or can UMF
improve while the planner's actual candidate ranking stays uninformative
(or gets worse)?

For ONE fixed (init_state, goal_state) pair: captures CEM's iteration-0
per-candidate costs under EACH requested kind (baseline + charts) -- every
kind sees the SAME K=num_samples candidate action sequences, since CEM's
iteration-0 draw is independent of the model (local_seed=0, mean=0/
std=var_scale, fixed in build_cfg()) -- then rolls out ALL K candidates for
REAL in the env (cheap CPU pymunk physics; the only GPU cost is the one CEM
search per kind already needed to capture costs) to get each candidate's
TRUE final block distance to goal -- an outcome no model, chart or baseline,
ever sees. Reports Spearman rho(planner_cost, true_final_dist) per kind:
both are "lower = better", so a well-calibrated cost ranking gives rho near
+1; near 0 or negative means the kind's cost function doesn't track real
outcomes even if its UMF (open-loop prediction error) looks good -- the
direct mechanism for why UMF and planning success can dissociate.

Rewritten for the post-T1/T6 pipeline (E0_IMPLEMENTATION_PLAN.md). The
original version of this file predated the rollout-bug fix and used
prepare_with_visual()'s old 3-arg signature and PhysicsRegime wrapping
PushTWrapper (both since changed in run_e0_planning.py, T6) -- replaced
outright rather than patched, since it would not run correctly as-is.

Usage:
    python scripts/diagnose_cem_costs.py --kinds baseline ln_act --regime R2 --seed 0
    # local smoke test (reduced CEM settings):
    python scripts/diagnose_cem_costs.py --kinds baseline --regime R2 --seed 0 \\
        --num-samples 16 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy.stats import spearmanr
from tqdm import tqdm

import atlas
from atlas.chart import Chart
from atlas.regimes import PhysicsRegime, set_regime_config
from atlas.score import rollout_umf  # noqa: E402 -- additive import only, score.py itself untouched

from cem_planning_eval import (  # noqa: E402
    FRAMESKIP, HUB_PATH, block_success, build_cfg, load_dataset_states,
    make_obs_td, prepare_with_visual, sample_dataset_init_goal,
)

if HUB_PATH not in sys.path:
    sys.path.insert(0, HUB_PATH)
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv  # noqa: E402
from evals.simu_env_planning.planning.gc_agent import GC_Agent  # noqa: E402


def instrument_cost_function(planner, capture: list, capture_iteration: str = "first") -> None:
    """Captures ONE CEM candidate batch (actions + costs).

    capture_iteration="first" (default, original behaviour): captures
    iteration-0 only -- the raw prior draw, same candidates for every kind by
    construction (CEM's first draw doesn't depend on the model), so later
    iterations are irrelevant and not captured.

    capture_iteration="last": keeps overwriting `capture` on every call, so
    once the CEM loop finishes it holds the FINAL iteration's population
    (the converged/elite-refined candidates CEM actually executes from) --
    answers the standing objection that iteration-0 is an untrained random
    draw, not what CEM actually uses (OPUS_REMAINING_TASKS.md C.25). Note
    this candidate batch is NOT shared across kinds the way iteration-0 is
    (each kind's own model shapes which candidates survive to the final
    iteration), so the cross-kind "same input" assumption/warning in main()
    only applies in "first" mode.
    """
    original = planner.cost_function
    call_count = [0]

    def wrapped(actions, z_init):
        cost = original(actions, z_init)
        call_count[0] += 1
        if capture_iteration == "first" and call_count[0] == 1:
            capture.append(actions.detach().cpu().clone())  # [horizon, num_samples, A]
            capture.append(cost.detach().cpu().clone().flatten())  # [num_samples]
        elif capture_iteration == "last":
            capture.clear()
            capture.append(actions.detach().cpu().clone())
            capture.append(cost.detach().cpu().clone().flatten())
        return cost

    planner.cost_function = wrapped


def rollout_true_outcomes(base_env, regime, seed: int, init_state: np.ndarray,
                           goal_state: np.ndarray, model_actions: torch.Tensor,
                           agent: GC_Agent) -> tuple[np.ndarray, np.ndarray]:
    """model_actions: [horizon, num_samples, action_dim] (model-chunk,
    normalized -- CEM's own units). Resets to the identical init_state before
    EVERY candidate so each rollout is independent, then executes that
    candidate's raw actions for real. Returns (true final block_pos_diff [px,
    lower=better], total_contacts) per candidate, both shape [num_samples] --
    contacts added for OPUS_REMAINING_TASKS.md A.4's degeneracy check (do
    most iteration-0 candidates ever touch the block at all?)."""
    num_samples = model_actions.shape[1]
    distances = np.empty(num_samples, dtype=np.float64)
    contacts = np.empty(num_samples, dtype=np.int64)
    for i in range(num_samples):
        cand = model_actions[:, i, :]  # [horizon, A]
        raw = rearrange(cand, "t (f d) -> (t f) d", d=2)
        raw = agent.preprocessor.denormalize_actions(raw).numpy()
        prepare_with_visual(base_env, regime, seed, init_state)  # reset -- same start every candidate
        final_state = None
        total_contacts = 0
        for a in raw:
            _, _, _, info = base_env.step(a)
            final_state = info["state"]
            total_contacts += info["n_contacts"]
        distances[i] = block_success(goal_state, final_state)["block_pos_diff"]
        contacts[i] = total_contacts
    return distances, contacts


def rollout_true_outcomes_and_umf(
    base_env, regime, seed: int, init_state: np.ndarray, goal_state: np.ndarray,
    model_actions: torch.Tensor, agent: GC_Agent, world_model, device,
    frameskip: int = FRAMESKIP, motion_gate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """SUBMISSION_PLAN.md E-A (`--save-latents`): same real rollout as
    `rollout_true_outcomes` (identical distances/contacts, computed the same
    way -- kept as a separate function rather than a flag threaded into the
    original so the default `--save-latents`-unset path is provably
    untouched), but ALSO builds the real encoder-output trajectory for each
    candidate (encode the actual visual/proprio observed at every
    model-chunk boundary the candidate's real rollout passes through) and
    scores it against the CURRENTLY-APPLIED chart/predictor for this `kind`
    via `atlas.score.rollout_umf` -- reusing score.py's own UMF definition
    and its own `_open_loop_rollout` rather than reimplementing prediction
    logic (score.py itself is not modified; only imported).

    Uses `rollout_umf`, NOT `umf()`: the caller (main()) already applies the
    kind's chart to `world_model.model.predictor` once, for the whole
    per-kind block (matching `rollout_umf`'s contract of "predictor already
    in the state to be scored" -- see score.py's own docstring comparing the
    two functions) -- `umf()`'s own apply_/restore_ would be redundant here
    and would leave the predictor state inconsistent with the rest of the
    per-kind block (goal encoding, agent.act, etc., which all run under the
    SAME applied chart).

    This is the "UMF on held-out candidates" readout E-A calls for. It does
    NOT save the raw per-candidate encoder-output latents themselves (300
    candidates x 7 frames x 256 x 384 floats per seed per kind is ~800MB and
    does not fit this script's existing per-seed JSON/incremental-jsonl
    format) -- per the dispatch spec's explicit "or equivalently the raw
    prediction error" alternative, only the resulting scalar UMF (score.py's
    own normalized prediction-error ratio) is persisted per candidate. A
    downstream fitting step (E-A's "fit ln_act with the existing
    run_e0_finetune loss") needs the raw transitions themselves and would
    reuse run_e0.py's own trajectory-collection path, not this function.

    Returns (distances, contacts, umfs) -- umfs[i] is a float, or None if
    score.py's own informative-chunk gate fired for that candidate (only
    possible when motion_gate is not None; default None here matches
    run_e0_planning.py's own post-hoc per-replan UMF logging convention,
    per run_e0.py's inline comment on the same choice).
    """
    num_samples = model_actions.shape[1]
    distances = np.empty(num_samples, dtype=np.float64)
    contacts = np.empty(num_samples, dtype=np.int64)
    umfs: list = [None] * num_samples
    for i in range(num_samples):
        cand = model_actions[:, i, :]  # [horizon, A] -- model-chunk, normalized (CEM's own units)
        raw = rearrange(cand, "t (f d) -> (t f) d", d=2)
        raw = agent.preprocessor.denormalize_actions(raw).numpy()
        obs0, _ = prepare_with_visual(base_env, regime, seed, init_state)  # reset -- same start every candidate
        imgs = [obs0["visual"]]
        proprios = [obs0["proprio"]]
        final_state = None
        total_contacts = 0
        for step_idx, a in enumerate(raw):
            obs, _, _, info = base_env.step(a)
            final_state = info["state"]
            total_contacts += info["n_contacts"]
            if (step_idx + 1) % frameskip == 0:
                # Model-chunk boundary -- capture the real observation here,
                # matching run_e0.py::load_regime_trajectories' subsampling
                # convention (keep every `frameskip`-th raw frame).
                imgs.append(obs["visual"])
                proprios.append(obs["proprio"])
        distances[i] = block_success(goal_state, final_state)["block_pos_diff"]
        contacts[i] = total_contacts

        imgs_np = np.stack(imgs, axis=0)          # [T_model+1, 224, 224, 3]
        proprios_np = np.stack(proprios, axis=0)  # [T_model+1, proprio_dim]
        visual_t = torch.from_numpy(imgs_np.copy()).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
        proprio_t = torch.from_numpy(proprios_np.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            # Same encode() call as run_e0.py::load_regime_trajectories --
            # raw uint8-range visual + raw proprio, encode() does its own
            # /255 + preprocessor.transform + normalize_proprios internally.
            enc = world_model.encode({"visual": visual_t, "proprio": proprio_t})
            enc_out = enc["visual"].squeeze(0).squeeze(1).flatten(1, 2)  # [T_model+1, N, D]
            proprio_enc = enc["proprio"].squeeze(0)                     # [T_model+1, P_tok, D_p]
            proprio_ctxt_i = proprio_enc[0:1].unsqueeze(0)               # [1, 1, P_tok, D_p]
            umfs[i] = rollout_umf(
                world_model, enc_out, cand.to(device),
                proprio_ctxt=proprio_ctxt_i, motion_gate=motion_gate,
            )
    return distances, contacts, umfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-ranking diagnostic: Spearman rho(planner cost, true outcome) per kind.")
    parser.add_argument("--kinds", nargs="+", required=True,
                         choices=["baseline", "ln_act", "lora4", "full"])
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--regime-config", type=str, default=None,
                         help="JSON dict overriding this regime's default physics params, e.g. "
                              "'{\"damping\": 0.25}' for an intermediate severity between R0's "
                              "implicit 0 and R2's default 0.5. Applied via "
                              "atlas.regimes.set_regime_config, same mechanism as "
                              "run_e0_planning.py's identical flag. Omit to use REGIME_CONFIGS' "
                              "existing default for --regime.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                         help="One (state,goal) pair per seed -- loops all kinds x all seeds, "
                              "reusing the loaded model (only the predictor weights get swapped "
                              "between kinds, via a pristine state-dict snapshot).")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=6)
    parser.add_argument("--capture-iteration", choices=["first", "last"], default="first",
                         help="'first' (default): iteration-0's raw prior draw, same candidates "
                              "across kinds by construction (the original design). 'last': the "
                              "FINAL iteration's converged/elite-refined population -- what CEM "
                              "actually executes from (OPUS_REMAINING_TASKS.md C.25). NOT shared "
                              "across kinds in 'last' mode -- the cross-kind same-input warning "
                              "below only fires meaningfully in 'first' mode.")
    parser.add_argument("--charts-dir", type=Path, default=atlas.OUT_DIR / "e0")
    parser.add_argument("--out-dir", type=Path, default=atlas.OUT_DIR / "cost_ranking")
    parser.add_argument("--save-latents", action="store_true",
                         help="SUBMISSION_PLAN.md E-A. When set, additionally encodes the real "
                              "observation trajectory each candidate's rollout actually passes "
                              "through and scores it against the currently-applied chart/predictor "
                              "via atlas.score.rollout_umf, saving a per-candidate UMF value under "
                              "results[kind]['umf_per_candidate'] (a new key -- all EXISTING keys "
                              "and the default flag-unset output are byte-identical to before this "
                              "flag existed). Costs one extra encode() call per candidate on top of "
                              "the already-required real env rollout -- see "
                              "rollout_true_outcomes_and_umf's docstring for what is and is not "
                              "saved (scalar UMF, not raw latents).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dino_wm_pusht from local hub...")
    model, prep = torch.hub.load(HUB_PATH, "dino_wm_pusht", source="local",
                                  force_reload=False, trust_repo=True)
    wm = model.model if hasattr(model, "model") else model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, model = wm.to(device), model.to(device)
    for p in wm.encoder.parameters():
        p.requires_grad_(False)
    for m in wm.predictor.modules():  # T7 throughput fix -- see run_e0_planning.py
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = True
    torch.set_float32_matmul_precision("high")

    # Snapshot BEFORE any chart is applied -- the safe way back to pristine
    # weights between kinds (chart.restore_() does NOT do this for most
    # kinds; see HANDOFF.md's chart.restore_() gotcha).
    pristine_predictor_state = {k: v.clone() for k, v in wm.predictor.state_dict().items()}

    if args.regime_config is not None:
        set_regime_config(args.regime, json.loads(args.regime_config))
    base_env = PushTEnv(render_size=224, with_velocity=True)
    regime = PhysicsRegime(base_env, args.regime)  # wraps base_env directly (T6) -- NOT PushTWrapper
    states, seq_lengths = load_dataset_states()
    cfg = build_cfg(args.num_samples, args.iterations, args.horizon, args.num_act_stepped)

    per_seed: list[dict] = []
    pooled_costs: dict[str, list[float]] = {k: [] for k in args.kinds}
    pooled_dist: dict[str, list[float]] = {k: [] for k in args.kinds}

    # FIX_SPEC.md C7: this script used to write ONLY at the very end (the
    # documented cause of two lost dose-response runs -- a crash/timeout
    # anywhere in the seeds x kinds loop below lost every already-computed
    # seed's data). Compute the output path up front and append each
    # completed seed's record to a sibling .jsonl file immediately, so a
    # crash mid-sweep loses at most the seed in progress, not the whole run.
    seeds_str = "-".join(str(s) for s in args.seeds)
    suffix = "" if args.capture_iteration == "first" else f"_iter{args.capture_iteration}"
    out_path = args.out_dir / f"cost_ranking_{args.regime}_seeds{seeds_str}{suffix}.json"
    incremental_path = out_path.with_suffix(".jsonl")
    if incremental_path.exists():
        incremental_path.unlink()  # fresh run -- do not append to a stale file from a prior attempt

    seed_pbar = tqdm(args.seeds, desc=f"cost_ranking_{args.regime}", unit="seed")
    for seed in seed_pbar:
        rs = np.random.RandomState(seed)
        init_state, goal_state = sample_dataset_init_goal(states, seq_lengths, rs)
        print(f"\n== seed={seed} regime={args.regime} "
              f"init_block_pos_diff={np.linalg.norm(goal_state[2:4] - init_state[2:4]):.1f}px ==")

        results: dict = {}
        shared_actions: torch.Tensor | None = None
        for kind in tqdm(args.kinds, desc="  kinds", unit="kind", leave=False):
            wm.predictor.load_state_dict(pristine_predictor_state)
            if kind != "baseline":
                chart_path = args.charts_dir / f"chart_{kind}_{args.regime}.pt"
                chart = Chart.load(str(chart_path), wm.predictor)
                chart.apply_(wm.predictor)

            agent = GC_Agent(cfg, model, dset=None, preprocessor=prep)
            agent.device = device

            capture: list = []
            instrument_cost_function(agent.planner, capture, capture_iteration=args.capture_iteration)

            goal_obs, _ = prepare_with_visual(base_env, regime, seed, goal_state)
            agent.set_goal(make_obs_td(goal_obs["visual"], goal_obs["proprio"], device))
            obs, _ = prepare_with_visual(base_env, regime, seed, init_state)
            obs_td = make_obs_td(obs["visual"], obs["proprio"], device)
            # steps_left in MODEL-chunk units (run_e0_planning.py's convention);
            # a single replan's worth (n_replans_target=1) is just num_act_stepped.
            agent.act(obs_td, steps_left=args.num_act_stepped)

            actions, costs = capture  # [horizon, num_samples, A], [num_samples]
            if args.capture_iteration == "first":
                if shared_actions is None:
                    shared_actions = actions
                else:
                    max_diff = (actions - shared_actions).abs().max().item()
                    if max_diff > 1e-5:
                        print(f"  [WARN] seed={seed} kind={kind}'s iteration-0 candidates differ "
                              f"from the first kind's by {max_diff:.6g} -- the same-input guarantee "
                              f"this diagnostic relies on doesn't hold here for this seed.")

            umf_per_candidate = None
            if args.save_latents:
                # Chart for this kind (if any) is already applied to
                # wm.predictor above and stays applied for the rest of this
                # per-kind block -- rollout_umf scores against that
                # already-applied state (see the function's own docstring).
                true_dist, contacts, umf_per_candidate = rollout_true_outcomes_and_umf(
                    base_env, regime, seed, init_state, goal_state, actions, agent,
                    world_model=model, device=device)
            else:
                true_dist, contacts = rollout_true_outcomes(base_env, regime, seed, init_state,
                                                              goal_state, actions, agent)
            rho, pval = spearmanr(costs.numpy(), true_dist)
            contact_mask = contacts > 0
            contact_frac = float(contact_mask.mean())
            if contact_mask.sum() >= 3:
                rho_contact, pval_contact = spearmanr(costs.numpy()[contact_mask], true_dist[contact_mask])
            else:
                rho_contact, pval_contact = float("nan"), float("nan")
            print(f"  kind={kind}: rho={rho:.3f} (p={pval:.4g}), n={len(true_dist)}, "
                  f"contact_frac={contact_frac:.3f}, rho|contact={rho_contact:.3f}, "
                  f"mean_true_dist={true_dist.mean():.1f}px, "
                  f"best_by_cost_true_dist={true_dist[int(costs.argmin())]:.1f}px")
            results[kind] = {
                "spearman_rho": float(rho), "spearman_p": float(pval),
                "n_candidates": int(len(true_dist)),
                "mean_true_dist": float(true_dist.mean()),
                "best_by_cost_true_dist": float(true_dist[int(costs.argmin())]),
                # A.4/A.5 (OPUS_REMAINING_TASKS.md): degeneracy + regret/top-tail metrics
                # need per-candidate data, not just summary stats -- persisted below.
                "contact_fraction": contact_frac,
                "min_true_dist": float(true_dist.min()),
                "median_true_dist": float(np.median(true_dist)),
                "max_true_dist": float(true_dist.max()),
                "spearman_rho_contact_subset": float(rho_contact),
                "spearman_p_contact_subset": float(pval_contact),
                "n_contact_subset": int(contact_mask.sum()),
                "costs": costs.numpy().tolist(),
                "true_dist": true_dist.tolist(),
                "contacts": contacts.tolist(),
            }
            if umf_per_candidate is not None:
                # SUBMISSION_PLAN.md E-A (--save-latents): new, clearly-flagged
                # keys only -- nothing above this block changes when the flag
                # is unset, and existing parsers reading costs/true_dist/
                # contacts are unaffected.
                results[kind]["umf_per_candidate"] = [
                    (float(u) if u is not None else None) for u in umf_per_candidate
                ]
                valid_umf = [u for u in umf_per_candidate if u is not None]
                results[kind]["umf_mean"] = float(np.mean(valid_umf)) if valid_umf else None
                results[kind]["umf_n_gated"] = int(len(umf_per_candidate) - len(valid_umf))
            pooled_costs[kind].extend(costs.numpy().tolist())
            pooled_dist[kind].extend(true_dist.tolist())

        seed_record = {
            "seed": seed, "init_state": init_state.tolist(), "goal_state": goal_state.tolist(),
            "results": results,
        }
        per_seed.append(seed_record)
        # FIX_SPEC.md C7: append immediately, don't wait for the sweep to finish.
        with open(incremental_path, "a") as f:
            f.write(json.dumps(seed_record) + "\n")
        seed_pbar.set_postfix({k: f"{results[k]['spearman_rho']:.3f}" for k in args.kinds})

    print(f"\n== Pooled across {len(args.seeds)} seed(s), "
          f"{args.num_samples} candidates/seed ==")
    pooled_summary: dict = {}
    for kind in args.kinds:
        rho, pval = spearmanr(pooled_costs[kind], pooled_dist[kind])
        seed_rhos = [s["results"][kind]["spearman_rho"] for s in per_seed]
        mean_of_seed_rhos = float(np.mean(seed_rhos))
        # A.6 (OPUS_REMAINING_TASKS.md): report the per-seed mean as a proper
        # CI, not a bare number -- SD/sqrt(n) across seeds, 95% via normal
        # approx (n=10 is thin for a t-based CI to matter much either way).
        sd_of_seed_rhos = float(np.std(seed_rhos, ddof=1)) if len(seed_rhos) > 1 else float("nan")
        se_of_mean = sd_of_seed_rhos / np.sqrt(len(seed_rhos)) if len(seed_rhos) > 1 else float("nan")
        ci95 = (mean_of_seed_rhos - 1.96 * se_of_mean, mean_of_seed_rhos + 1.96 * se_of_mean) \
            if len(seed_rhos) > 1 else (float("nan"), float("nan"))
        print(f"  kind={kind}: pooled rho={rho:.3f} (p={pval:.4g}, n={len(pooled_dist[kind])}) | "
              f"mean of per-seed rhos={mean_of_seed_rhos:.3f}, "
              f"95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}] (n_seeds={len(seed_rhos)})")
        pooled_summary[kind] = {
            "pooled_spearman_rho": float(rho), "pooled_spearman_p": float(pval),
            "pooled_n": len(pooled_dist[kind]),
            "mean_of_per_seed_rhos": mean_of_seed_rhos,
            "sd_of_per_seed_rhos": sd_of_seed_rhos,
            "se_of_mean_seed_rho": se_of_mean,
            "ci95_of_mean_seed_rho": list(ci95),
        }

    out_path.write_text(json.dumps({
        "regime": args.regime, "seeds": args.seeds, "kinds": args.kinds,
        "num_samples": args.num_samples, "iterations": args.iterations,
        "capture_iteration": args.capture_iteration,
        "charts_dir": str(args.charts_dir),  # provenance -- see the atlas_out/e0 (pre-fix,
                                              # invalidated) vs atlas_out/e0_v3_dataset (canonical)
                                              # gap this metadata field is meant to prevent.
        "per_seed": per_seed, "pooled": pooled_summary,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
