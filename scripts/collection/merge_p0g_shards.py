"""
scripts/merge_p0g_shards.py — combine P0-G collection shards (produced by
concurrent Modal containers via run_e0.py's --collect-traj-offset /
--collect-skip-val-test) into one canonical trajs_{regime}.pt +
chunks_{regime}.jsonl + e0_seed_manifest.json, and recompute the gate/
block-static reports on the MERGED set (a per-shard motion_gate is calibrated
on that shard's own subset, not the full requested trajectory count — must
not be reported as if it were).

Mirrors scripts/merge_planning_shards.py's contract: reject overlap/mismatch
rather than silently drop or double-count.

Usage:
    python scripts/merge_p0g_shards.py --regime R2 \\
        --shard-dirs atlas_out/p0g_R2_shard0 atlas_out/p0g_R2_shard1 \\
        --out-dir atlas_out/p0g_R2_merged

Exactly one shard dir must contain val/test trajectories (the one launched
WITHOUT --collect-skip-val-test) -- pass it first or last, order doesn't
matter, it's found by content.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_training import (compute_motion_gates, derive_and_report_motion_gate,
                    dump_regime_chunks, report_block_static_fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge P0-G collection shards.")
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--shard-dirs", nargs="+", required=True, type=Path,
                         help="Each shard's --out directory (contains trajs_{regime}.pt).")
    parser.add_argument("--out-dir", type=Path, required=True,
                         help="Merged output directory (created if missing).")
    args = parser.parse_args()
    regime = args.regime

    blobs = []
    for d in args.shard_dirs:
        p = d / f"trajs_{regime}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Shard trajectory file not found: {p}")
        # map_location="cpu": trajectory tensors are saved from GPU collection
        # containers with a CUDA device tag; this merge step is pure
        # concatenation + light stats and correctly runs on a CPU-only Modal
        # container (no gpu= on merge_p0g_shards) -- torch.load without this
        # tries to restore onto the ORIGINAL cuda device and raises
        # "torch.cuda.is_available() is False" on a box with no GPU at all.
        blobs.append((d, torch.load(p, map_location="cpu", weights_only=False)))

    # Protocol agreement check: every shard's guard must match on everything
    # EXCEPT num_train_trajs (each shard collects a different-sized slice by
    # design). A mismatch on any other field means the shards were launched
    # under different CLI args and must NOT be merged.
    base_guard = dict(blobs[0][1]["guard"])
    base_guard.pop("num_train_trajs", None)
    for d, blob in blobs[1:]:
        g = dict(blob["guard"])
        g.pop("num_train_trajs", None)
        if g != base_guard:
            raise ValueError(
                f"Shard protocol mismatch: {blobs[0][0]} vs {d}\n"
                f"  {base_guard}\n  {g}\n"
                f"Shards were launched under different --collect-* args -- refusing to merge.")

    # Merge train: concatenate, then assert seed uniqueness (the real proof
    # that --collect-traj-offset was set correctly and no work overlapped).
    all_train = [t for _, blob in blobs for t in blob["train"]]
    seeds = [t["seed"] for t in all_train]
    dupes = {s for s in seeds if seeds.count(s) > 1}
    if dupes:
        raise ValueError(
            f"Overlapping seeds across shards: {sorted(dupes)[:10]} -- shards were run with "
            f"overlapping --collect-traj-offset ranges. Fix the offsets and re-run; refusing to "
            f"silently drop or double-count trajectories.")
    all_train.sort(key=lambda t: t["seed"])
    print(f"Merged train: {len(all_train)} trajectories from {len(blobs)} shard(s), "
          f"seeds {min(seeds)}..{max(seeds)}, all unique.")

    # val/test: taken from whichever shard(s) actually collected them (the one
    # launched WITHOUT --collect-skip-val-test). Exactly one is expected;
    # more than one means val/test was accidentally collected in >1 shard
    # (wasted compute, not a correctness bug) -- warn, then dedupe by seed the
    # same way train is protected.
    val_sources = [(d, blob["val"]) for d, blob in blobs if blob["val"]]
    test_sources = [(d, blob["test"]) for d, blob in blobs if blob["test"]]
    if len(val_sources) > 1:
        print(f"  [WARNING] val trajectories present in {len(val_sources)} shards "
              f"(expected 1) -- wasted compute, using {val_sources[0][0]}", flush=True)
    if len(test_sources) > 1:
        print(f"  [WARNING] test trajectories present in {len(test_sources)} shards "
              f"(expected 1) -- wasted compute, using {test_sources[0][0]}", flush=True)
    val_trajectories = val_sources[0][1] if val_sources else []
    test_trajectories = test_sources[0][1] if test_sources else []
    print(f"Merged val: {len(val_trajectories)} (from {val_sources[0][0] if val_sources else 'none'})")
    print(f"Merged test: {len(test_trajectories)} (from {test_sources[0][0] if test_sources else 'none'})")

    merged_guard = dict(base_guard)
    merged_guard["num_train_trajs"] = len(all_train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"guard": merged_guard, "train": all_train,
                "val": val_trajectories, "test": test_trajectories},
               args.out_dir / f"trajs_{regime}.pt")
    print(f"Wrote {args.out_dir / f'trajs_{regime}.pt'}")

    # chunks_{regime}.jsonl: concatenate every shard's dump (each shard already
    # emitted UMF(c0)/latent_disp/block_disp for its own train+val windows via
    # dump_regime_chunks -- no need to recompute, just concatenate the rows).
    # "traj" field values collide across shards (each restarts at 0); harmless
    # -- nothing downstream groups by that field, only aggregates per-row.
    n_chunks = 0
    with open(args.out_dir / f"chunks_{regime}.jsonl", "w") as out_f:
        for d, _ in blobs:
            src = d / f"chunks_{regime}.jsonl"
            if not src.exists():
                continue
            for line in src.read_text().splitlines():
                if line.strip():
                    out_f.write(line + "\n")
                    n_chunks += 1
    print(f"Merged {n_chunks} chunk rows -> {args.out_dir / f'chunks_{regime}.jsonl'}")

    # Seed manifest: reconstructed DIRECTLY from the trajectory objects already
    # loaded above (each carries seed/episode_idx/offset/n_contacts), NOT from
    # shard dirs' own e0_seed_manifest.json files. Those are NOT regime-
    # namespaced (unlike trajs_{regime}.pt / chunks_{regime}.jsonl) -- two
    # regimes launched concurrently into the same --out-subdir (as happened
    # here: R0 and R2 both used "p0g_onpolicy") race to the SAME path in a
    # shared shard dir, and whichever finishes last silently overwrites the
    # other's manifest. The trajectory .pt files are safe (regime-namespaced);
    # only this reconstruction-from-source approach is trustworthy under that
    # collision. Real bug, found via this exact merge failing with
    # `KeyError: 'R2'` on a manifest that had been overwritten with R0's.
    def _row(t: dict) -> dict:
        return {"seed": t["seed"], "episode_idx": t["episode_idx"],
                "offset": t["offset"], "n_contacts": t.get("n_contacts")}
    merged_manifest = {
        "source": "closed_loop", "regime_config": merged_guard["regime_config"],
        "train": sorted((_row(t) for t in all_train), key=lambda r: r["seed"]),
        "eval": sorted((_row(t) for t in val_trajectories), key=lambda r: r["seed"]),
        "test": sorted((_row(t) for t in test_trajectories), key=lambda r: r["seed"]),
    }
    (args.out_dir / "e0_seed_manifest.json").write_text(
        json.dumps({regime: merged_manifest}, indent=2))
    print(f"Wrote merged manifest (reconstructed from trajectory objects, not shard "
          f"e0_seed_manifest.json files) -> {args.out_dir / 'e0_seed_manifest.json'}")

    # Recompute gates + reports on the MERGED set -- per-shard values were
    # calibrated on a subset and must not be reported as final.
    nas = int(merged_guard["collect_cem"].split("nas=")[1]) if merged_guard.get("collect_cem") else 2
    motion_gate, chunk_motion_gate = compute_motion_gates(all_train, nas, verbose_label=f"{regime} (merged)")
    report_block_static_fraction(args.out_dir, regime, {
        "train": all_train, "val": val_trajectories, "test": test_trajectories,
    })
    derive_and_report_motion_gate(args.out_dir, regime,
                                  args.out_dir / f"chunks_{regime}.jsonl", chunk_motion_gate)

    print(f"\n[OK] Merge complete: {len(all_train)} train / {len(val_trajectories)} val / "
          f"{len(test_trajectories)} test trajectories -> {args.out_dir}")


if __name__ == "__main__":
    main()
