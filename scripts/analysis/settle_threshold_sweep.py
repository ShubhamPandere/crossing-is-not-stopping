"""
FABLE5 six-day plan Day 1.5 — free, local, no-GPU analyses of the archived C-2
cells. Read-only: consumes per-episode JSONLs already on disk, writes a JSON
summary + PNGs under phase0_v3/. Nothing here touches score.py/stats.py or any
planning loop.

Three deliverables:
  1. SR-vs-threshold-radius curve (position radius 10..60 px, angle fixed at the
     pi/9 success bound). The chart and frozen curves are expected to cross —
     the one-panel visualisation of "variance compression against a threshold".
  2. Final block-distance ECDFs per arm per cadence.
  3. Paired bootstrap CI on the sd-ratio (chart_sd / frozen_sd) of the final
     block-distance distribution (Levene alone at n=20 invites a reviewer quibble).

Cells (all damping=0.5, N=300, horizon=6, seeds 0-19):
  nas=2:  it=10 (headline), it=1, it=3, alpha=0   — differ in iterations/alpha,
          pooled ONLY for the descriptive ECDF/sd view, exactly as FIXLOG V3-19 §3 does
  nas=6:  the substrate's own validated cadence
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
P0 = ROOT / "phase0_v3"
ANGLE_BOUND = np.pi / 9

CELLS = {
    "it10_headline_nas2": ("p0c/p0c_it10_baseline_R2.jsonl", "c2_p0g_R2/ln_act_R2.jsonl"),
    "it1_nas2": ("c2_dose_it1_baseline/baseline_R2.jsonl", "c2_dose_it1_ln_act/ln_act_R2.jsonl"),
    "it3_nas2": ("c2_dose_it3_baseline/baseline_R2.jsonl", "c2_dose_it3_ln_act/ln_act_R2.jsonl"),
    "alpha0_nas2": ("c2_alpha0_baseline/baseline_R2.jsonl", "c2_alpha0_ln_act/ln_act_R2.jsonl"),
    "nas6": ("c2_nas6_baseline/baseline_R2.jsonl", "c2_nas6_ln_act/ln_act_R2.jsonl"),
}
POOLS = {
    "nas2_pool": ["it10_headline_nas2", "it1_nas2", "it3_nas2", "alpha0_nas2"],
    "nas6": ["nas6"],
    "headline_only": ["it10_headline_nas2"],
}


def load(rel: str) -> list[dict]:
    return [json.loads(l) for l in (P0 / rel).read_text().splitlines() if l.strip()]


def paired_arrays(cell: str):
    b_rel, c_rel = CELLS[cell]
    b, c = load(b_rel), load(c_rel)
    by = {r["episode"]: r for r in b}
    cy = {r["episode"]: r for r in c}
    eps = sorted(set(by) & set(cy))
    # pairing assertion — same init task in both arms
    for e in eps:
        assert abs(by[e]["init_block_pos_diff"] - cy[e]["init_block_pos_diff"]) < 1e-6, (cell, e)
    frozen_pos = np.array([by[e]["block_pos_diff"] for e in eps])
    chart_pos = np.array([cy[e]["block_pos_diff"] for e in eps])
    frozen_ang = np.array([by[e]["block_angle_diff"] for e in eps])
    chart_ang = np.array([cy[e]["block_angle_diff"] for e in eps])
    return eps, frozen_pos, chart_pos, frozen_ang, chart_ang


def sr_at_radius(pos, ang, r):
    return float(np.mean((pos < r) & (ang < ANGLE_BOUND)))


def crossing_radius(radii, cp, ca, fp, fa, min_frozen_sr=0.10):
    """Smallest radius at which chart SR >= frozen SR, restricted to radii where
    the frozen arm has already reached min_frozen_sr — so the degenerate r where
    both arms are still ~0 does not count as a crossing. None => the chart never
    catches up within the swept range."""
    for r in radii:
        f = sr_at_radius(fp, fa, r)
        if f < min_frozen_sr:
            continue
        if sr_at_radius(cp, ca, r) >= f:
            return float(r)
    return None


def bootstrap_sd_ratio(chart_pos, frozen_pos, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    n_ep = len(chart_pos)
    ratios = []
    for _ in range(n):
        idx = rng.integers(0, n_ep, n_ep)  # paired resample of episodes
        cs, fs = chart_pos[idx].std(ddof=1), frozen_pos[idx].std(ddof=1)
        if fs > 0:
            ratios.append(cs / fs)
    ratios = np.array(ratios)
    point = chart_pos.std(ddof=1) / frozen_pos.std(ddof=1)
    return point, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def main():
    radii = np.arange(10, 61, 2.0)
    out: dict = {"angle_bound_rad": ANGLE_BOUND, "cells": {}, "pools": {}}

    for cell in CELLS:
        eps, fp, cp, fa, ca = paired_arrays(cell)
        out["cells"][cell] = {
            "n": len(eps),
            "frozen_sr_20px": sr_at_radius(fp, fa, 20),
            "chart_sr_20px": sr_at_radius(cp, ca, 20),
            "frozen_mean_px": float(fp.mean()), "chart_mean_px": float(cp.mean()),
            "frozen_sd_px": float(fp.std(ddof=1)), "chart_sd_px": float(cp.std(ddof=1)),
            "sd_ratio_chart_over_frozen": float(cp.std(ddof=1) / fp.std(ddof=1)),
            "sr_curve_crossing_radius_px": crossing_radius(radii, cp, ca, fp, fa),
            "sr_curve": {float(r): {"frozen": sr_at_radius(fp, fa, r),
                                     "chart": sr_at_radius(cp, ca, r)} for r in radii},
        }

    for pool, cells in POOLS.items():
        fp = np.concatenate([paired_arrays(c)[1] for c in cells])
        cp = np.concatenate([paired_arrays(c)[2] for c in cells])
        fa = np.concatenate([paired_arrays(c)[3] for c in cells])
        ca = np.concatenate([paired_arrays(c)[4] for c in cells])
        sr_curve = {float(r): {"frozen": sr_at_radius(fp, fa, r), "chart": sr_at_radius(cp, ca, r)}
                    for r in radii}
        cross = crossing_radius(radii, cp, ca, fp, fa)
        pt, lo, hi = bootstrap_sd_ratio(cp, fp)
        out["pools"][pool] = {
            "n_paired": len(fp), "cells": cells,
            "sd_ratio_chart_over_frozen": pt, "sd_ratio_ci95": [lo, hi],
            "sd_ratio_ci_excludes_1": hi < 1.0 or lo > 1.0,
            "sr_curve_crossing_radius_px": cross,
            "sr_curve": sr_curve,
            "frozen_mean_px": float(fp.mean()), "chart_mean_px": float(cp.mean()),
            "frozen_sd_px": float(fp.std(ddof=1)), "chart_sd_px": float(cp.std(ddof=1)),
        }

    (P0 / "c2_threshold_sweep.json").write_text(json.dumps(out, indent=2))

    # ---- Figure 1: SR-vs-radius, nas2 pool + nas6 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, pool in zip(axes, ["nas2_pool", "nas6"]):
        d = out["pools"][pool]["sr_curve"]
        rs = sorted(d)
        ax.plot(rs, [d[r]["frozen"] for r in rs], "-o", ms=3, label="frozen c₀", color="#1b6ca8")
        ax.plot(rs, [d[r]["chart"] for r in rs], "-s", ms=3, label="ln_act chart", color="#c8102e")
        ax.axvline(20, ls=":", c="grey", lw=1, label="success bound (20 px)")
        cr = out["pools"][pool]["sr_curve_crossing_radius_px"]
        if cr is not None:
            ax.axvline(cr, ls="--", c="green", lw=1, label=f"chart catches frozen @ {cr:.0f} px")
        else:
            ax.text(0.5, 0.05, "chart never catches frozen in [10,60]",
                    transform=ax.transAxes, ha="center", fontsize=8, color="grey")
        ax.set_title(f"{pool}  (n={out['pools'][pool]['n_paired']} paired)")
        ax.set_xlabel("position success radius (px)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=8)
    fig.suptitle("C-2: SR vs position-success radius (angle bound fixed). "
                 "nas=6: curves cross ~30 px. nas=2: chart never catches up (inaction pathology).")
    fig.tight_layout()
    fig.savefig(P0 / "c2_fig_threshold_sweep.png", dpi=130)
    plt.close(fig)

    # ---- Figure 2: final-distance ECDFs ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, pool in zip(axes, ["nas2_pool", "nas6"]):
        cells = POOLS[pool]
        fp = np.concatenate([paired_arrays(c)[1] for c in cells])
        cp = np.concatenate([paired_arrays(c)[2] for c in cells])
        for arr, lab, col in [(fp, "frozen c₀", "#1b6ca8"), (cp, "ln_act chart", "#c8102e")]:
            xs = np.sort(arr)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", label=lab, color=col)
        ax.axvline(20, ls=":", c="grey", lw=1)
        ax.set_title(f"{pool}  (n={len(fp)})")
        ax.set_xlabel("final block distance (px)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECDF")
    axes[0].legend(fontsize=8)
    fig.suptitle("C-2: final block-distance ECDF — frozen is bimodal, chart is compressed")
    fig.tight_layout()
    fig.savefig(P0 / "c2_fig_final_dist_ecdf.png", dpi=130)
    plt.close(fig)

    print(json.dumps({k: out["pools"][k] for k in out["pools"]}, indent=2)[:2000])
    print("\nwrote phase0_v3/c2_threshold_sweep.json, c2_fig_threshold_sweep.png, c2_fig_final_dist_ecdf.png")


if __name__ == "__main__":
    main()
