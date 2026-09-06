"""
modal/modal_e0_planning.py — Modal GPU runner for scripts/run_e0_planning.py.

Separate from modal_app.py because that file's image installs atlas-wm from
a stale GitHub remote with none of this session's fixes (code-review.md Bugs
#1-#7) or run_e0_planning.py itself. This image builds from the local repo.

Usage:
    modal volume create atlas-data   # if not already created
    modal volume put atlas-data atlas_out/e0 /atlas_root/atlas_out/e0  # upload local charts
    modal run modal/modal_e0_planning.py --kind ln_act --regime R1 --episodes 10

Needs the checkpoint/dataset in the volume already (see modal_app.py::download_data),
plus the E0 charts uploaded as above (they were trained locally, not on Modal).
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

import modal

atlas_volume = modal.Volume.from_name("atlas-data", create_if_missing=True)
ATLAS_MOUNT_PATH = "/atlas_root"
REPO_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("uv")
    .add_local_dir(
        # /src, NOT under /atlas_root: that's the volume mount point, and Modal
        # refuses to mount a volume onto a path that already has image content.
        str(REPO_ROOT), remote_path="/src", copy=True,
        ignore=[".venv", ".git", "data", "hub", "atlas_out", "graphify-out", "__pycache__", "logs",
                "phase0_v3",   # accessed via the mounted volume at runtime, not the image build --
                               # this was missing here (present in modal_phase0.py's ignore list)
                               # and was uploading 661MB+ of local trajectory data on every build.
                "Paper_Draft", "research_audit"],
                               # 2026-09-03: not read at runtime; a concurrent session editing the
                               # paper caused "file modified during build" failures. Excluding the
                               # doc trees makes builds immune to doc edits.
    )
    .run_commands(
        # vendor/jepa-wms installed explicitly first: pyproject.toml deliberately
        # does not list it as a dependency (uv can't resolve a relative `file:`
        # path for a dependency of this project's own wheel, and a hardcoded
        # absolute path isn't portable across machines/containers).
        "cd /src && uv pip install --system -e vendor/jepa-wms && "
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
        "uv pip install --system -e ."
    )
    .env({
        "ATLAS_HOME": ATLAS_MOUNT_PATH,
        "JEPAWM_DSET": f"{ATLAS_MOUNT_PATH}/data",
        "JEPAWM_LOGS": f"{ATLAS_MOUNT_PATH}/logs",
        "JEPAWM_CKPT": f"{ATLAS_MOUNT_PATH}/ckpts",
        "ATLAS_OUT": f"{ATLAS_MOUNT_PATH}/atlas_out",
        "TORCH_HOME": f"{ATLAS_MOUNT_PATH}/hub",
    })
)

app = modal.App("atlas-e0-planning", image=image)


@app.function(
    gpu="L4",  # 24GB @ $0.80/h -- reverted from T4 (2026-08-26): T4 measured ~3x slower
               # for planning specifically (450-465s/ep vs L4's ~148s/ep) despite fitting
               # comfortably in memory either way -- CEM's compute-bound candidate batch
               # apparently hits T4's weaker raw FLOPs much harder than training's
               # backward-pass-bound workload did (T4 was only ~1.65x slower there).
               # L4 costs more per hour but finishes faster, and wall-clock is now the
               # binding constraint, not $/hour alone.
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 6,
)
def run_e0_planning(
    kind: str = "ln_act",
    regime: str = "R1",
    regime_config: str | None = None,
    episodes: int = 10,
    num_samples: int = 300,
    iterations: int = 30,
    horizon: int = 6,
    num_act_stepped: int = 6,
    charts_subdir: str = "e0",
    out_subdir: str = "e0_planning",
    log_planner_diagnostics: bool = False,
    min_block_pos_diff: float = 40.0,
    max_agent_block_dist: float | None = None,
    episode_start: int = 0,
    out_suffix: str = "",
    log_umf: bool = True,
    charts_root: str = "atlas_out",
    out_root: str = "atlas_out",
    # C2_FAILURE_DIAGNOSIS.md 3.2 test: proprio weight in the planner cost.
    # 0.1 = the substrate default, so every existing caller is unchanged.
    objective_alpha: float = 0.1,
    # FABLE5 six-day plan Day 1.1: post-success settle tail. 0 = unchanged.
    settle_steps: int = 0,
    # FIXLOG V3-24: opt-in fix for the steps_left unit-mismatch bug. False = unchanged.
    fix_steps_left: bool = False,
    # FIXLOG V3-26: what drives the env during the settle hold ("zero" = unchanged),
    # and whether to log the block's simulator-reported velocity through it.
    settle_mode: str = "zero",
    settle_log_velocity: bool = False,
) -> None:
    """Defaults = the SUBSTRATE's own validated Push-T config (CEM 300x30,
    horizon 6, num_act_stepped 6 -> 30 raw steps/episode, 1 replan), the
    config dino_wm_pusht reports ~90% SR under -- see
    run_e0_planning.py's module docstring (E0_IMPLEMENTATION_PLAN.md T6).
    charts_subdir/out_subdir let a corrected-config re-run write to e.g.
    atlas_out/e0_planning_v2/ without overwriting earlier (superseded-config,
    kept for the record) results. min_block_pos_diff: minimum required block
    displacement between a sampled real init/goal pair -- unfiltered sampling
    let some pairs draw an already-near-goal state, making success trivial
    without any real pushing (confirmed empirically: 5/10 baseline R0
    episodes finished in <=8 raw steps under the old unfiltered sampler).
    Runs with this filter active must use a NEW out_subdir (e.g.
    e0_planning_v2) -- old results under the same seeds were sampled
    differently and are not resume-compatible with filtered runs.
    max_agent_block_dist (P2d, E0_RECOVERY_PLAN.md): rejects init states the
    agent cannot plausibly reach within the step budget -- None (default)
    defers to run_e0_planning.py's own DEFAULT_MAX_AGENT_BLOCK_DIST rather
    than duplicating that data-derived constant here.
    episode_start/out_suffix: for splitting one N-episode request across
    multiple concurrent Modal calls (e.g. episode_start=0/episodes=50 and
    episode_start=50/episodes=100, out_suffix='_shard0'/'_shard1') -- see
    scripts/merge_planning_shards.py to combine them afterward. log_umf:
    on by default -- logs per-replan UMF of the executed chunk alongside
    success, giving an episode-level (UMF, success) pair for free.
    charts_root/out_root (P0G_FIX_PLAN §4.5 C-2): override the "atlas_out"
    prefix so charts_subdir/out_subdir can point under phase0_v3 -- e.g.
    charts_root="phase0_v3", charts_subdir="p0g_onpolicy" plans with the real
    on-policy P0-G chart at phase0_v3/p0g_onpolicy/chart_{kind}_{regime}.pt.
    Default unchanged; every existing caller unaffected."""
    import subprocess
    import sys
    # Volumes are eventually consistent -- reload() picks up commits made by
    # a separate `modal volume put` process before this container started.
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/collection/cem_planning_eval.py",
           "--kind", kind,
           "--regime", regime,
           "--episodes", str(episodes),
           "--num-samples", str(num_samples),
           "--iterations", str(iterations),
           "--horizon", str(horizon),
           "--num-act-stepped", str(num_act_stepped),
           "--charts-dir", f"{ATLAS_MOUNT_PATH}/{charts_root}/{charts_subdir}",
           "--out-dir", f"{ATLAS_MOUNT_PATH}/{out_root}/{out_subdir}",
           "--min-block-pos-diff", str(min_block_pos_diff),
           "--episode-start", str(episode_start),
           "--out-suffix", out_suffix,
           "--objective-alpha", str(objective_alpha)]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    if max_agent_block_dist is not None:
        cmd += ["--max-agent-block-dist", str(max_agent_block_dist)]
    if settle_steps:
        cmd += ["--settle-steps", str(settle_steps)]
    if settle_mode != "zero":
        cmd += ["--settle-mode", settle_mode]
    if settle_log_velocity:
        cmd.append("--settle-log-velocity")
    if fix_steps_left:
        cmd.append("--fix-steps-left")
    if log_planner_diagnostics:
        cmd.append("--log-planner-diagnostics")
    if not log_umf:
        cmd.append("--no-log-umf")
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.function(
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=600,
)
def merge_shards(kind: str, regime: str, out_subdir: str, shards: list[str],
                 out_root: str = "atlas_out") -> None:
    """Runs scripts/merge_planning_shards.py inside a container with the
    volume mounted, so combining shard outputs needs no local download."""
    import subprocess
    import sys
    atlas_volume.reload()
    subprocess.run(
        [sys.executable, "scripts/merge_planning_shards.py",
         "--kind", kind, "--regime", regime,
         "--out-dir", f"{ATLAS_MOUNT_PATH}/{out_root}/{out_subdir}",
         "--shards", *shards],
        check=True, cwd="/src",
    )
    atlas_volume.commit()


@app.local_entrypoint()
def main(kind: str = "ln_act", regime: str = "R1", regime_config: str | None = None,
          episodes: int = 10,
          num_samples: int = 300, iterations: int = 30, horizon: int = 6,
          num_act_stepped: int = 6, charts_subdir: str = "e0", out_subdir: str = "e0_planning",
          log_planner_diagnostics: bool = False, min_block_pos_diff: float = 40.0,
          max_agent_block_dist: float | None = None,
          episode_start: int = 0, out_suffix: str = "", log_umf: bool = True,
          num_shards: int = 1,
          charts_root: str = "atlas_out", out_root: str = "atlas_out",
          objective_alpha: float = 0.1, settle_steps: int = 0,
          fix_steps_left: bool = False, settle_mode: str = "zero",
          settle_log_velocity: bool = False) -> None:
    """num_shards > 1: splits [episode_start, episodes) into that many
    contiguous, near-equal ranges, launches each as its own CONCURRENT Modal
    container (via .spawn(), not sequential .remote() calls), waits for all
    to finish, then merges them into the canonical {kind}_{regime}.jsonl via
    merge_shards(). This is the actual wall-clock lever on a deadline -- the
    workload is a sequential per-episode CEM loop on a small model (not
    GPU-flop-bound at this batch size), so N containers in parallel beats a
    single faster GPU. E.g. --episodes 100 --num-shards 4 runs four L4s
    concurrently instead of one L4 for 4x as long, for the same total cost."""
    if num_shards <= 1:
        # .spawn() not .remote() -- a bare .remote() was found NOT to survive a
        # local process kill despite --detach (P0-G FIXLOG V3-14/15).
        call = run_e0_planning.spawn(
            kind=kind, regime=regime, regime_config=regime_config, episodes=episodes,
            num_samples=num_samples, iterations=iterations, horizon=horizon,
            min_block_pos_diff=min_block_pos_diff, max_agent_block_dist=max_agent_block_dist,
            num_act_stepped=num_act_stepped, charts_subdir=charts_subdir,
            out_subdir=out_subdir, log_planner_diagnostics=log_planner_diagnostics,
            episode_start=episode_start, out_suffix=out_suffix, log_umf=log_umf,
            charts_root=charts_root, out_root=out_root,
            objective_alpha=objective_alpha, settle_steps=settle_steps,
            fix_steps_left=fix_steps_left, settle_mode=settle_mode,
            settle_log_velocity=settle_log_velocity)
        print(f"Spawned run_e0_planning as function call {call.object_id}. "
              f"Not waiting locally -- check `modal app logs` for progress/completion.")
        return

    total = episodes - episode_start
    if total <= 0:
        raise ValueError(f"episodes ({episodes}) must exceed episode_start ({episode_start}).")
    base, rem = divmod(total, num_shards)
    bounds = []
    start = episode_start
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)  # first `rem` shards get one extra episode
        if size == 0:
            continue  # more shards requested than episodes to cover
        bounds.append((start, start + size))
        start += size

    print(f"Splitting episodes [{episode_start},{episodes}) into {len(bounds)} shard(s): {bounds}")
    shard_suffixes = [f"_shard{i}" for i in range(len(bounds))]
    calls = [
        run_e0_planning.spawn(
            kind=kind, regime=regime, regime_config=regime_config, episodes=e,
            num_samples=num_samples, iterations=iterations, horizon=horizon,
            min_block_pos_diff=min_block_pos_diff, max_agent_block_dist=max_agent_block_dist,
            num_act_stepped=num_act_stepped, charts_subdir=charts_subdir, out_subdir=out_subdir,
            log_planner_diagnostics=log_planner_diagnostics,
            episode_start=s, out_suffix=suffix, log_umf=log_umf,
            charts_root=charts_root, out_root=out_root,
            objective_alpha=objective_alpha, settle_steps=settle_steps,
            fix_steps_left=fix_steps_left, settle_mode=settle_mode,
            settle_log_velocity=settle_log_velocity,
        )
        for (s, e), suffix in zip(bounds, shard_suffixes)
    ]
    # Containers run concurrently (already launched via .spawn() above); this
    # loop only blocks LOCALLY waiting for results, one shard at a time, in
    # whatever order they were spawned -- not the order they actually finish
    # in. That's fine for a progress bar (all N will complete regardless),
    # just don't read bar order as "shard i finished i-th".
    for call in tqdm(calls, desc=f"{kind}_{regime} shards", unit="shard"):
        call.get()
    print("All shards complete -- merging into the canonical file...")
    merge_shards.remote(kind=kind, regime=regime, out_subdir=out_subdir, shards=shard_suffixes,
                        out_root=out_root)
    print(f"Merged: {out_root}/{out_subdir}/{kind}_{regime}.jsonl")


@app.function(
    gpu="L4",  # 24GB @ $0.80/h -- switched from T4 2026-08-27 per user request
               # (L4 measured faster for this checkpoint's CEM workload in
               # run_e0_planning.py, same reasoning applies here: same K=300
               # candidate batch per seed, plus K cheap CPU-only env rollouts
               # per kind for the true-outcome side of the diagnostic).
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 3,  # was 3600 -- at ~178s/seed, 20 seeds is ~3560s of pure
                       # compute, plus model-load overhead pushes it over a
                       # 1-hour timeout (confirmed the hard way 2026-08-27:
                       # two dose-response jobs died at 19/20 and ~15/20
                       # seeds respectively, and since this script only
                       # writes its output ONCE AT THE END -- unlike
                       # run_e0_planning.py's resumable per-episode writes --
                       # a timeout this close to the true runtime loses
                       # EVERY completed seed, not just the unfinished one).
)
def diagnose_cem_costs(
    kinds: str = "baseline,ln_act",
    regime: str = "R1",
    regime_config: str | None = None,
    seeds: str = "0",
    num_samples: int = 300,
    iterations: int = 30,
    horizon: int = 6,
    num_act_stepped: int = 6,
    capture_iteration: str = "first",
    charts_subdir: str = "e0",
    out_subdir: str = "cost_ranking",
    charts_root: str = "atlas_out",
    out_root: str = "atlas_out",
) -> None:
    """scripts/diagnose_cem_costs.py -- cost-ranking diagnostic: for each
    fixed (init, goal) pair (one per seed in `seeds`, comma-separated, e.g.
    '0,1,2,3,4,5,6,7,8,9'), captures a CEM candidate batch under each kind
    (same K candidates every kind, by construction, in capture_iteration=
    'first' mode only), rolls all K out for real in the env to get each
    candidate's TRUE final block distance, and reports Spearman
    rho(cost, true_dist) per kind -- the mechanism figure for why UMF and
    planning success can dissociate even when UMF looks good. kinds:
    comma-separated, e.g. 'baseline,ln_act'. capture_iteration: 'first'
    (default, iteration-0 raw draw) or 'last' (final/converged population --
    OPUS_REMAINING_TASKS.md C.25). regime_config (added 2026-08-27, for the
    dose-response sweep): JSON dict overriding this regime's default physics
    param, e.g. '{"damping": 0.25}' for an intermediate severity between
    R0's implicit 0 and R2's default 0.5 -- same mechanism as
    run_e0_planning.py's identical flag. charts_root/out_root (added for
    P0G_FIX_PLAN §4.5 C-1): override the "atlas_out" prefix so charts_subdir/
    out_subdir can point under phase0_v3 instead -- e.g. charts_root=
    "phase0_v3", charts_subdir="p0g_onpolicy" loads the real on-policy P0-G
    chart at phase0_v3/p0g_onpolicy/chart_ln_act_R2.pt. Default unchanged, so
    every existing caller is unaffected."""
    import subprocess
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/collection/cem_cost_ranking.py",
           "--kinds", *kinds.split(","),
           "--regime", regime,
           "--seeds", *seeds.split(","),
           "--num-samples", str(num_samples),
           "--iterations", str(iterations),
           "--horizon", str(horizon),
           "--num-act-stepped", str(num_act_stepped),
           "--capture-iteration", capture_iteration,
           "--charts-dir", f"{ATLAS_MOUNT_PATH}/{charts_root}/{charts_subdir}",
           "--out-dir", f"{ATLAS_MOUNT_PATH}/{out_root}/{out_subdir}"]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.local_entrypoint(name="diagnose_cem_costs")
def diagnose_cem_costs_entrypoint(kinds: str = "baseline,ln_act", regime: str = "R1",
                                    regime_config: str | None = None, seeds: str = "0",
                                    num_samples: int = 300, iterations: int = 30, horizon: int = 6,
                                    num_act_stepped: int = 6, capture_iteration: str = "first",
                                    charts_subdir: str = "e0",
                                    out_subdir: str = "cost_ranking",
                                    charts_root: str = "atlas_out",
                                    out_root: str = "atlas_out") -> None:
    # .spawn() not .remote() -- a bare .remote() was found NOT to survive a
    # local process kill despite --detach (P0-G FIXLOG V3-14/V3-15), even
    # though it usually does; .spawn() is the proven-robust pattern.
    call = diagnose_cem_costs.spawn(
        kinds=kinds, regime=regime, regime_config=regime_config, seeds=seeds,
        num_samples=num_samples, iterations=iterations, horizon=horizon,
        num_act_stepped=num_act_stepped, capture_iteration=capture_iteration,
        charts_subdir=charts_subdir, out_subdir=out_subdir,
        charts_root=charts_root, out_root=out_root)
    print(f"Spawned diagnose_cem_costs as function call {call.object_id}. "
          f"Not waiting locally -- check `modal app logs` for progress/completion.")


@app.function(
    gpu="T4",  # 16GB @ $0.59/h -- offline fine-tuning (backprop through the adapter's
               # <=20.8M params only, no CEM candidate batch) needs less memory than
               # planning's measured 6.45GB peak, so T4's headroom is even more
               # comfortable here than in run_e0_planning (see that function's comment).
    volumes={ATLAS_MOUNT_PATH: atlas_volume},
    timeout=3600 * 6,
)
def run_e0_train(
    kinds: str = "ln_act",
    regimes: str = "R1",
    regime_config: str | None = None,
    steps: int = 2000,
    num_train_trajs: int = 20,
    train_traj_len: int = 25,
    num_val_trajs: int = 8,
    eval_traj_len: int = 50,
    eval_every: int = 25,
    patience: int = 5,
    data_source: str = "dataset",
    data_split: str = "train",
    out_subdir: str = "e0_v2",
    collect_num_samples: int = 100,
    collect_iterations: int = 10,
) -> None:
    """scripts/run_e0.py -- offline chart fine-tuning, T9: real-data replay
    (data_source='dataset') + early stopping (eval_every/patience) on a held-
    out val split (num_val_trajs), sized for Modal's 24GB L4 (not the 6GB
    local card). out_subdir defaults to e0_v2 so this doesn't overwrite the
    original (pre-T1-fix, invalidated) atlas_out/e0/ charts. regime_config: JSON
    string applied to EVERY regime in `regimes` via set_regime_config -- pass
    the SAME calibrated config used at eval time (run_e0_planning's own
    regime_config), or a chart trains under one physics and gets evaluated
    under another with no error raised anywhere (E0_RECOVERY_PLAN.md P3)."""
    import subprocess
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/collection/chart_training.py",
           "--kinds", *kinds.split(","),
           "--regimes", *regimes.split(","),
           "--steps", str(steps),
           "--num-train-trajs", str(num_train_trajs),
           "--train-traj-len", str(train_traj_len),
           "--num-val-trajs", str(num_val_trajs),
           "--eval-traj-len", str(eval_traj_len),
           "--eval-every", str(eval_every),
           "--patience", str(patience),
           "--data-source", data_source,
           "--data-split", data_split,
           "--out", f"{ATLAS_MOUNT_PATH}/atlas_out/{out_subdir}"]
    if data_source == "closed_loop":
        cmd += ["--collect-num-samples", str(collect_num_samples),
                "--collect-iterations", str(collect_iterations)]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    subprocess.run(cmd, check=True, cwd="/src")
    atlas_volume.commit()


@app.local_entrypoint(name="run_e0_train")
def run_e0_train_entrypoint(kinds: str = "ln_act", regimes: str = "R1",
                              regime_config: str | None = None, steps: int = 2000,
                              num_train_trajs: int = 20, train_traj_len: int = 25,
                              num_val_trajs: int = 8, eval_traj_len: int = 50,
                              eval_every: int = 25, patience: int = 5,
                              data_source: str = "dataset", data_split: str = "train",
                              out_subdir: str = "e0_v2", collect_num_samples: int = 100,
                              collect_iterations: int = 10) -> None:
    run_e0_train.remote(kinds=kinds, regimes=regimes, regime_config=regime_config, steps=steps,
                         num_train_trajs=num_train_trajs, train_traj_len=train_traj_len,
                         num_val_trajs=num_val_trajs, eval_traj_len=eval_traj_len,
                         eval_every=eval_every, patience=patience,
                         data_source=data_source, data_split=data_split,
                         out_subdir=out_subdir, collect_num_samples=collect_num_samples,
                         collect_iterations=collect_iterations)


