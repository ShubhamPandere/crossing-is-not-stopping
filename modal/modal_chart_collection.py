"""
modal/modal_phase0.py — Modal GPU runner for scripts/phase0_measure.py
(IMPLEMENTATION_PLAN_V3 §11.1 Phase 0, gates P0-A/B/D/E).

Forward-only, no CEM planner -> T4 is plenty. Writes under the volume's
phase0_v3/ (NOT atlas_out/). Same local-repo image as modal_e0_planning.py.

    modal run --detach modal/modal_phase0.py::main --num-trajs 80
"""
from __future__ import annotations

from pathlib import Path

import modal
from tqdm import tqdm

atlas_volume = modal.Volume.from_name("atlas-data", create_if_missing=True)
MOUNT = "/atlas_root"
REPO_ROOT = Path(__file__).parent.parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1")
    .pip_install("uv")
    .add_local_dir(
        str(REPO_ROOT), remote_path="/src", copy=True,
        ignore=[".venv", ".git", "data", "hub", "atlas_out", "graphify-out",
                "__pycache__", "logs", "phase0_v3"],
    )
    # §7-B1: modal_phase0.py imports _p0g_spec at module load; in the remote
    # container the entrypoint is at /root/ (not /root/modal/), so put the spec
    # there too. (The sys.path /src/scripts fallback below covers it as well.)
    .add_local_file(str(REPO_ROOT / "scripts" / "collection" / "collection_spec.py"),
                    remote_path="/root/collection_spec.py", copy=True)
    .run_commands(
        "cd /src && uv pip install --system -e vendor/jepa-wms && "
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
        "uv pip install --system -e ."
    )
    .env({
        "ATLAS_HOME": MOUNT,
        "JEPAWM_DSET": f"{MOUNT}/data",
        "JEPAWM_LOGS": f"{MOUNT}/logs",
        "JEPAWM_CKPT": f"{MOUNT}/ckpts",
        "ATLAS_OUT": f"{MOUNT}/atlas_out",
        "TORCH_HOME": f"{MOUNT}/hub",
    })
)

app = modal.App("atlas-phase0", image=image)


def _local_git_sha() -> str:
    """P20: read the SHA on the CLIENT (the image drops .git, so the remote
    `git rev-parse` returns 'unknown')."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
    except Exception:
        return "unknown"


def _run(cmd: list[str], git_sha: str) -> None:
    import os
    import subprocess
    env = {**os.environ, "ATLAS_GIT_SHA": git_sha}
    subprocess.run(cmd, check=True, cwd="/src", env=env)


@app.function(gpu="T4", volumes={MOUNT: atlas_volume}, timeout=3600 * 3)
def phase0(num_trajs: int = 80, traj_len: int = 60, num_act_stepped: int = 2,
           regimes: str = "R0,R1,R2", git_sha: str = "unknown") -> None:
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/phase0_measure.py",
           "--out", f"{MOUNT}/phase0_v3",
           "--regimes", *regimes.split(","),
           "--num-trajs", str(num_trajs),
           "--traj-len", str(traj_len),
           "--num-act-stepped", str(num_act_stepped)]
    _run(cmd, git_sha)
    atlas_volume.commit()


# P0-G is SPLIT into collection and fine-tune (P2 / v3 §5): the realistic
# single-container total for collect+finetune of both regimes is ~13.6 h — over
# the old timeout=3600*8. Splitting also means a fine-tune failure no longer
# destroys the ~4 h of collection, and the fine-tune becomes independently
# re-runnable (needed for §4.1's determinism-asymmetry check). Run ONE regime
# per call (P1: a combined R0,R2 call collects R2 under an R0-adapted predictor;
# also isolates CEM generator state). Cost line: the old SMOKE_SUMMARY.md
# "$3.6 / 4.5 h" is SUPERSEDED (§7-C-2) — §3.1 roughly doubles CEM compute per
# trajectory (plan_length 9→18 summed), so ~135 s/traj is the first-order
# estimate (NOT measured): ~4.3 h collection + ~5.8 h fine-tune per regime vs the
# 6 h / 10 h timeouts here. EVIDENCE_LEDGER §5 budgets P0-G at ~$15. Re-measure
# on the next smoke and rewrite this + SMOKE_SUMMARY.md + plan §11.2.
# R0 is a SEPARATE `--regime R0` call (τ / σ_r are defined over R0 chunks) —
# nothing here guards against forgetting it (§7-C-3).

# §7-B1: the guard-relevant flag spec lives in scripts/collection/collection_spec.py
# (no `modal` import there) so tests/test_p0g_guard.py can use it without a Modal
# runtime. This module is imported both on the client (during `modal run`, where
# the repo is at REPO_ROOT) AND inside the remote container (where the entrypoint
# file is copied to /root/ and the repo image lives at /src — REPO_ROOT is then
# "/" and useless). Insert every candidate path.
import sys as _sys  # noqa: E402
for _p in (str(REPO_ROOT / "scripts" / "collection"), "/src/scripts/collection", "/root"):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from collection_spec import _P0G_COMMON, _P0G_DEFAULTS, _p0g_flags  # noqa: E402,F401


@app.function(gpu="L4", volumes={MOUNT: atlas_volume}, timeout=3600 * 6)
def p0g_collect(regime: str = "R2",
                num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                traj_len: int = _P0G_DEFAULTS["traj_len"],
                nas: int = _P0G_DEFAULTS["nas"],
                num_samples: int = _P0G_DEFAULTS["num_samples"],
                iterations: int = _P0G_DEFAULTS["iterations"],
                num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                out_subdir: str = "p0g_onpolicy", git_sha: str = "unknown",
                traj_offset: int = 0, skip_val_test: bool = False,
                regime_config: str | None = None) -> None:
    """P0-G on-policy chart-training-data COLLECTION only (v3 §5.2). ONE regime
    per call. closed_loop collector, all v3 fixes: contact filter OFF,
    --collect-num-act-stepped FUNCTIONAL, N=300/it=10/nas=2, eval-matched
    lookahead + goal separation (§3.1/§3.2), determinism on. Persists
    trajs_{regime}.pt + chunks_{regime}.jsonl, then exits.
    Smoke: --num-trajs 5 --num-val-trajs 2 --num-test-trajs 2.

    traj_offset/skip_val_test: sharding hooks (§ user request 2026-08-29,
    mirrors modal_e0_planning.py's --num-shards). Use p0g-collect-sharded
    rather than setting these by hand.

    regime_config: JSON string overriding this regime's default physics param
    (e.g. '{"damping": 0.2}'), same mechanism as modal_e0_planning.py's
    identical flag — scripts/run_e0.py already accepts --regime-config
    natively; this was just never threaded through here. A chart collected
    under a non-default regime_config MUST be fine-tuned (p0g_finetune) with
    the SAME regime_config, or it trains on one physics and evaluates on
    another with no error raised anywhere (E0_RECOVERY_PLAN.md P3)."""
    import sys
    atlas_volume.reload()
    cmd = [sys.executable, "scripts/collection/chart_training.py", *_P0G_COMMON, "--collect-only",
           "--regimes", regime,
           "--collect-traj-offset", str(traj_offset),
           *_p0g_flags(traj_len, eval_traj_len, num_trajs, num_val_trajs,
                       num_test_trajs, num_samples, iterations, nas),
           "--out", f"{MOUNT}/phase0_v3/{out_subdir}"]
    if skip_val_test:
        cmd.append("--collect-skip-val-test")
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    _run(cmd, git_sha)
    atlas_volume.commit()


@app.function(volumes={MOUNT: atlas_volume}, timeout=1200)
def merge_p0g_shards(regime: str, shard_subdirs: list[str], out_subdir: str) -> None:
    """Runs scripts/merge_p0g_shards.py inside a container with the volume
    mounted, so combining shards needs no local download (mirrors
    modal_e0_planning.py::merge_shards)."""
    import subprocess
    import sys
    atlas_volume.reload()
    shard_dirs = [f"{MOUNT}/phase0_v3/{s}" for s in shard_subdirs]
    subprocess.run(
        [sys.executable, "scripts/merge_p0g_shards.py", "--regime", regime,
         "--shard-dirs", *shard_dirs,
         "--out-dir", f"{MOUNT}/phase0_v3/{out_subdir}"],
        check=True, cwd="/src",
    )
    atlas_volume.commit()


@app.function(gpu="L4", volumes={MOUNT: atlas_volume}, timeout=3600 * 10)
def p0g_finetune(regime: str = "R2", steps: int = 2000,
                 out_subdir: str = "p0g_onpolicy", load_subdir: str | None = None,
                 git_sha: str = "unknown",
                 num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                 traj_len: int = _P0G_DEFAULTS["traj_len"],
                 nas: int = _P0G_DEFAULTS["nas"],
                 num_samples: int = _P0G_DEFAULTS["num_samples"],
                 iterations: int = _P0G_DEFAULTS["iterations"],
                 num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                 num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                 eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                 regime_config: str | None = None) -> None:
    """P0-G ln_act fine-tune from cached trajectories (P2). Requires a prior
    p0g_collect (load_subdir, defaults to out_subdir) for this regime — loads
    trajs_{regime}.pt via --load-trajs, so ZERO CEM searches happen here.
    Independently re-runnable.

    The eight collection-defining params default off _P0G_DEFAULTS and are
    emitted via _p0g_flags — identical to p0g_collect — so run_e0.py's
    --load-trajs guard matches (§7-B1). If a collection used non-default values,
    pass the SAME values here.

    regime_config: pass the SAME value used for the p0g_collect call being
    loaded — this only affects env construction for eval-time chart screening
    inside run_e0.py, but a mismatch here still trains a chart whose screening
    metrics were computed under the wrong physics with no error raised
    (E0_RECOVERY_PLAN.md P3), so keep it pinned to the collection's value.

    §4.1 (P16) determinism-asymmetry check: run this TWICE with the same
    load_subdir but different out_subdir (e.g. det_run1 / det_run2) per regime,
    then diff val_loss_ln_act_{regime}.json stopped_early_at_step and
    results.json eval_umf across the two. NEVER reuse an out_subdir (§1.7)."""
    import sys
    atlas_volume.reload()
    out = f"{MOUNT}/phase0_v3/{out_subdir}"
    src = f"{MOUNT}/phase0_v3/{load_subdir or out_subdir}"
    cmd = [sys.executable, "scripts/collection/chart_training.py", *_P0G_COMMON,
           "--regimes", regime,
           "--load-trajs", src,
           "--steps", str(steps),
           *_p0g_flags(traj_len, eval_traj_len, num_trajs, num_val_trajs,
                       num_test_trajs, num_samples, iterations, nas),
           "--out", out]
    if regime_config is not None:
        cmd += ["--regime-config", regime_config]
    _run(cmd, git_sha)
    atlas_volume.commit()


@app.local_entrypoint()
def main(num_trajs: int = 80, traj_len: int = 60, num_act_stepped: int = 2,
         regimes: str = "R0,R1,R2") -> None:
    phase0.remote(num_trajs=num_trajs, traj_len=traj_len,
                  num_act_stepped=num_act_stepped, regimes=regimes,
                  git_sha=_local_git_sha())


@app.local_entrypoint(name="p0g-collect")
def p0g_collect_entry(regime: str = "R2",
                      num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                      traj_len: int = _P0G_DEFAULTS["traj_len"],
                      nas: int = _P0G_DEFAULTS["nas"],
                      num_samples: int = _P0G_DEFAULTS["num_samples"],
                      iterations: int = _P0G_DEFAULTS["iterations"],
                      num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                      num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                      eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                      out_subdir: str = "p0g_onpolicy",
                      regime_config: str | None = None) -> None:
    """ONE regime per call (P1). **Run BOTH: --regime R2 (charts) AND --regime R0**
    — τ = P95 UMF(c₀) over R0 chunks and σ_r over the R0 informative set (§6.1,
    §6.3), so R0 collection is required even though no R0 chart is used downstream
    (§7-C-3). Smoke: --num-trajs 5 --num-val-trajs 2 --num-test-trajs 2 (§7-C-1)."""
    p0g_collect.remote(regime=regime, num_trajs=num_trajs, traj_len=traj_len,
                       nas=nas, num_samples=num_samples, iterations=iterations,
                       num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
                       eval_traj_len=eval_traj_len, out_subdir=out_subdir,
                       regime_config=regime_config,
                       git_sha=_local_git_sha())


@app.local_entrypoint(name="p0g-collect-sharded")
def p0g_collect_sharded_entry(regime: str = "R2",
                              num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                              traj_len: int = _P0G_DEFAULTS["traj_len"],
                              nas: int = _P0G_DEFAULTS["nas"],
                              num_samples: int = _P0G_DEFAULTS["num_samples"],
                              iterations: int = _P0G_DEFAULTS["iterations"],
                              num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                              num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                              eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                              out_subdir: str = "p0g_onpolicy",
                              num_shards: int = 4,
                              regime_config: str | None = None) -> None:
    """num_shards > 1: splits [0, num_trajs) train trajectories into that many
    contiguous, near-equal ranges (same divmod scheme as
    modal_e0_planning.py's --num-shards), launches each as its own CONCURRENT
    Modal container via .spawn() (not sequential .remote() calls), waits for
    all to finish, then merges via merge_p0g_shards.py. The actual wall-clock
    lever: collection is a sequential per-trajectory CEM loop (not
    GPU-flop-bound at this batch size), so N containers in parallel beats one
    container N times as long, for the same total GPU-time cost. E.g.
    --num-trajs 100 --num-shards 4 runs four L4s concurrently, ~25 trajectories
    each, instead of one L4 for 4x as long.

    Exactly ONE shard (shard 0) collects val/test; the others pass
    --collect-skip-val-test (val/test are cheap, 8 each, and splitting them
    buys nothing). Shard i gets num_trajs = base + (1 if i < rem else 0)
    trajectories at traj_offset = running total so far -- disjoint seeds,
    proven by scripts/merge_p0g_shards.py's overlap check on merge.

    Smoke first at a tiny --num-trajs before trusting this at N=100 (§1.1) --
    e.g. --num-trajs 8 --num-shards 2 --num-val-trajs 2 --num-test-trajs 2."""
    git_sha = _local_git_sha()
    if num_shards <= 1:
        p0g_collect.remote(regime=regime, num_trajs=num_trajs, traj_len=traj_len,
                           nas=nas, num_samples=num_samples, iterations=iterations,
                           num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
                           eval_traj_len=eval_traj_len, out_subdir=out_subdir,
                           regime_config=regime_config, git_sha=git_sha)
        return

    base, rem = divmod(num_trajs, num_shards)
    bounds = []  # (offset, size)
    offset = 0
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue  # more shards requested than trajectories to cover
        bounds.append((offset, size))
        offset += size

    print(f"Splitting {num_trajs} train trajectories into {len(bounds)} shard(s): {bounds}")
    shard_subdirs = [f"{out_subdir}_shard{i}" for i in range(len(bounds))]
    calls = [
        p0g_collect.spawn(
            regime=regime, num_trajs=size, traj_len=traj_len, nas=nas,
            num_samples=num_samples, iterations=iterations,
            num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
            eval_traj_len=eval_traj_len, out_subdir=subdir, git_sha=git_sha,
            traj_offset=off, skip_val_test=(i != 0),  # only shard 0 collects val/test
            regime_config=regime_config,
        )
        for i, ((off, size), subdir) in enumerate(zip(bounds, shard_subdirs))
    ]
    # Containers run concurrently (already launched via .spawn() above); this
    # loop only blocks LOCALLY waiting for results, in spawn order not finish
    # order -- fine for a progress bar, all N complete regardless.
    for call in tqdm(calls, desc=f"p0g_collect_{regime} shards", unit="shard"):
        call.get()

    print("All shards complete -- merging into the canonical file...")
    merge_p0g_shards.remote(regime=regime, shard_subdirs=shard_subdirs, out_subdir=out_subdir)


@app.local_entrypoint(name="p0g-merge-shards")
def p0g_merge_shards_entry(regime: str, shard_subdirs: str, out_subdir: str) -> None:
    """Re-run just the merge step on already-collected shard directories --
    e.g. after fixing a merge-side bug, without re-paying for collection.
    shard_subdirs: comma-separated, e.g. p0g_R2_shard0,p0g_R2_shard1."""
    merge_p0g_shards.remote(regime=regime, shard_subdirs=shard_subdirs.split(","),
                            out_subdir=out_subdir)


@app.local_entrypoint(name="p0g-finetune")
def p0g_finetune_entry(regime: str = "R2", steps: int = 2000,
                       out_subdir: str = "p0g_onpolicy", load_subdir: str = "",
                       num_trajs: int = _P0G_DEFAULTS["num_trajs"],
                       traj_len: int = _P0G_DEFAULTS["traj_len"],
                       nas: int = _P0G_DEFAULTS["nas"],
                       num_samples: int = _P0G_DEFAULTS["num_samples"],
                       iterations: int = _P0G_DEFAULTS["iterations"],
                       num_val_trajs: int = _P0G_DEFAULTS["num_val_trajs"],
                       num_test_trajs: int = _P0G_DEFAULTS["num_test_trajs"],
                       eval_traj_len: int = _P0G_DEFAULTS["eval_traj_len"],
                       regime_config: str | None = None) -> None:
    """Collection-defining params must MATCH the p0g_collect run being loaded
    (they default identically off _P0G_DEFAULTS; override all of them together
    if the collection was non-default). §7-B1.

    Uses .spawn() (fire-and-forget), NOT .remote() -- found the hard way
    (2026-08-30): a bare .remote() call keeps an open RPC in flight for the
    ENTIRE fine-tune duration, tied to THIS LOCAL PROCESS staying alive. If
    the local process is killed (not just network-dropped -- confirmed
    reproducible twice, cancelling a real 100-trajectory fine-tune mid-run at
    steps 60 and 206), the client propagates a cancellation despite
    --detach. .spawn() submits and returns immediately -- the function call
    is independently trackable server-side from that point on, exactly like
    p0g_collect_sharded_entry's per-shard .spawn() calls, which DID survive a
    local kill. Check progress/completion via `modal app logs <app-id>`."""
    call = p0g_finetune.spawn(regime=regime, steps=steps, out_subdir=out_subdir,
                              load_subdir=load_subdir or None, git_sha=_local_git_sha(),
                              num_trajs=num_trajs, traj_len=traj_len, nas=nas,
                              num_samples=num_samples, iterations=iterations,
                              num_val_trajs=num_val_trajs, num_test_trajs=num_test_trajs,
                              eval_traj_len=eval_traj_len, regime_config=regime_config)
    print(f"Spawned p0g_finetune (regime={regime}) as function call {call.object_id}. "
          f"Not waiting locally -- check `modal app logs` for progress/completion.")
