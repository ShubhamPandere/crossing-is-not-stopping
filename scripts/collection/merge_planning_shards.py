"""
scripts/merge_planning_shards.py — combine run_e0_planning.py shard outputs
(produced by concurrent Modal containers via --episode-start/--out-suffix)
into one canonical {kind}_{regime}.jsonl + summary, and report the UMF-vs-
success correlation (Kendall tau) across all episodes if --log-umf was on.

Usage:
    python scripts/merge_planning_shards.py --kind ln_act --regime R2 \\
        --out-dir atlas_out/e0_planning_n100 --shards _shard0 _shard1

Each shard file must be internally contiguous from its own --episode-start
(run_e0_planning.py enforces this); shards are concatenated and sorted by
episode index, then duplicate episode indices across shards (a sign two
shards were misconfigured with overlapping --episode-start/--episodes
ranges) are rejected rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge run_e0_planning.py shard JSONLs.")
    parser.add_argument("--kind", required=True, choices=["baseline", "ln_act", "lora4", "full"])
    parser.add_argument("--regime", required=True, choices=["R0", "R1", "R2"])
    parser.add_argument("--out-dir", type=Path, required=True,
                         help="Directory holding the shard files (same --out-dir every shard used).")
    parser.add_argument("--shards", nargs="+", required=True,
                         help="The --out-suffix values used for each shard, e.g. _shard0 _shard1.")
    parser.add_argument("--episode-start", type=int, default=0,
                         help="Expected first episode index of the merged output. FIX_SPEC.md "
                              "A13: the merged indices must be contiguous from here (no "
                              "upstream-missing episode passing silently).")
    parser.add_argument("--merged-suffix", type=str, default="",
                         help="--out-suffix of the merged output file (default: no suffix, i.e. "
                              "{kind}_{regime}.jsonl -- the canonical name downstream tools expect).")
    args = parser.parse_args()

    all_records: list[dict] = []
    for shard in args.shards:
        shard_path = args.out_dir / f"{args.kind}_{args.regime}{shard}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
        print(f"  {shard_path}: {sum(1 for _ in open(shard_path))} record(s)")

    eps = [r["episode"] for r in all_records]
    dupes = {e for e in eps if eps.count(e) > 1}
    if dupes:
        raise ValueError(
            f"Overlapping episode indices across shards: {sorted(dupes)} -- shards were run with "
            f"overlapping --episode-start/--episodes ranges. Fix the ranges and re-run; refusing to "
            f"silently drop or double-count episodes."
        )
    all_records.sort(key=lambda r: r["episode"])

    # FIX_SPEC.md A13: contiguity. Duplicates are already rejected above; this
    # catches the opposite failure -- an upstream-missing episode leaving a gap.
    sorted_eps = [r["episode"] for r in all_records]
    expected = list(range(args.episode_start, args.episode_start + len(sorted_eps)))
    if sorted_eps != expected:
        missing = sorted(set(expected) - set(sorted_eps))
        raise ValueError(
            f"Merged episode indices are not contiguous from --episode-start={args.episode_start}: "
            f"got {sorted_eps[:5]}...{sorted_eps[-5:]} (n={len(sorted_eps)}); "
            f"missing {missing[:20]}{' ...' if len(missing) > 20 else ''}. "
            f"A shard is incomplete -- re-run the missing episodes rather than merging a gapped set."
        )

    merged_path = args.out_dir / f"{args.kind}_{args.regime}{args.merged_suffix}.jsonl"
    with open(merged_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    n = len(all_records)
    success_rate = sum(r["success"] for r in all_records) / n
    mean_time = sum(r["wall_time"] for r in all_records) / n
    print(f"\nMerged {n} episode(s) -> {merged_path}")
    print(f"Success rate: {success_rate:.3f} ({sum(r['success'] for r in all_records)}/{n})")
    print(f"Mean wall time per episode: {mean_time:.1f}s")

    summary = {
        "kind": args.kind, "regime": args.regime, "episodes": n,
        "success_rate": success_rate, "mean_wall_time_s": mean_time,
        "shards": args.shards,
    }

    umf_pairs = [(r["umf_mean"], r["success"]) for r in all_records if r.get("umf_mean") is not None]
    if umf_pairs:
        umf_vals = [p[0] for p in umf_pairs]
        succ_vals = [int(p[1]) for p in umf_pairs]
        try:
            from scipy.stats import kendalltau
            tau, p_value = kendalltau(umf_vals, succ_vals)
            print(f"Kendall tau(UMF, success), n={len(umf_pairs)}: tau={tau:.3f}, p={p_value:.4f}")
            summary["umf_success_kendall_tau"] = tau
            summary["umf_success_kendall_p"] = p_value
            summary["umf_success_n"] = len(umf_pairs)
        except ImportError:
            print("scipy not available -- skipping Kendall tau (episode-level UMF/success pairs "
                  "are still in the merged JSONL for offline analysis).")

    summary_path = args.out_dir / f"{args.kind}_{args.regime}{args.merged_suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
