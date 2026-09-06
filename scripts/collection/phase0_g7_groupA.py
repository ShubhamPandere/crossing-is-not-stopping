"""
scripts/phase0_g7_groupA.py — P0-F / G7 CALIBRATION-STAGE diagnostic ("Group A").

This is NOT the final G7 measurement. G7 proper (evaluate ONE frozen
τ + motion-gate config, pass/fail) runs only AFTER a human freezes those two
values. This script is the decision aid that informs that freeze. It asserts
nothing and picks nothing.

Runs with NO GPU and NO new experiment runs: pure analysis of the chunk records
in phase0_v3/phase0_chunks.jsonl (produced by phase0_measure.py).

  PART 1  MOTION-GATE CALIBRATION TABLE (decision aid).
          Percentile sweep. Per candidate threshold: % kept / % gated,
          block-displacement of kept vs gated, UMF distribution of kept vs
          gated, per-chunk strike rate on the kept set — shown at BOTH the
          measured P0-A τ and the §1.7-pinned τ, because the gate cannot be
          frozen independently of τ.

  PART 2  STRIKE-COUNTER — PRELIMINARY WITHIN-TRAJECTORY LIVENESS.
          Replays the REAL atlas.expand.Expander.record() over each trajectory's
          6 chunks IN TRUE TEMPORAL ORDER (grouped by (regime, traj), ordered by
          k — never globally sorted/filtered). Each trajectory is one regime and
          6 chunks, so at most ~2 arming opportunities per trajectory — these
          counts are a LOWER BOUND / first-pass diagnostic, not "the production
          expansion mechanism fires X%". The production liveness number comes
          from the full A/B/A/B stream (Group B, needs P0-G charts). No claim is
          made about what the per-regime rate should be.

VERIFIED against atlas/ source (not assumed):
  * motion gate — atlas/score.py:87-89: production gates iff
    (z[-1]-z[0]).norm(p="fro") <= motion_gate  ->  return None. `latent_disp`
    in phase0_chunks.jsonl is that exact Frobenius norm (phase0_measure.py), and
    this script uses the same `<=`.
  * post-arm reset — atlas/expand.py: all three exit paths of maybe_expand()
    when strikes>=q (rejected_full :125-127, committed :166-168, rejected_score
    :171-173) reset EXACTLY {_strikes=0, _deficit_chunks.clear(), _candidate=
    None}; the other fields they touch (_n_probes_*, _last_probe_debug, library)
    are never read by record()/the strike counter, so replaying those three
    field resets reproduces every production post-arm effect on subsequent
    counting. The candidate FIT is skipped (one-chart library, no world model) —
    which is why this is arming liveness, not commit liveness.

Group B (router switch rate, probe COMMIT rate, K_max) needs real charts from
P0-G and a full-loop stream sim on the local GPU — not this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from atlas.expand import Expander, ExpansionConfig

REPO = Path(__file__).resolve().parent.parent.parent
CHUNKS = REPO / "phase0_v3" / "phase0_chunks.jsonl"


def load_trajectories():
    """Return {(regime, traj): [chunk, ...]} with each list in true temporal
    (ascending-k) order. No global sort, no global filter."""
    rows = [json.loads(l) for l in CHUNKS.open()]
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["regime"], r["traj"]), []).append(r)
    for key in groups:
        groups[key].sort(key=lambda c: c["k"])  # within-trajectory order only
    return groups, rows


def pctl(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else float("nan")


# ── PART 1 ────────────────────────────────────────────────────────────────────
def calibration_table(rows, taus, percentiles=(25, 50, 75, 90, 95)):
    static = [r["latent_disp"] for r in rows if r["block_disp_px"] < 1.0]
    print(f"\n=== PART 1 — motion-gate calibration (decision aid, freezes nothing) ===")
    print(f"block-static (<1px) chunks: n={len(static)}   "
          f"their latent_disp: p50={pctl(static,50):.0f} p90={pctl(static,90):.0f} p95={pctl(static,95):.0f}")
    hdr = (f"{'gate':>6} {'thresh':>7} {'%kept':>6} {'%gated':>7} "
           f"{'blkdisp k/g':>16} {'UMFmed k/g':>14} {'UMFp90 k/g':>14}   " +
           "  ".join(f"strike%@τ={t:g}" for t in taus))
    print(hdr)
    for p in percentiles:
        thr = pctl(static, p)
        kept = [r for r in rows if r["latent_disp"] > thr]
        gated = [r for r in rows if r["latent_disp"] <= thr]
        ku = [r["umf_c0"] for r in kept if r["umf_c0"] is not None]
        gu = [r["umf_c0"] for r in gated if r["umf_c0"] is not None]
        kb = np.median([r["block_disp_px"] for r in kept]) if kept else float("nan")
        gb = np.median([r["block_disp_px"] for r in gated]) if gated else float("nan")
        sk = "   ".join(f"{100*np.mean([u > t for u in ku]):>9.0f}%" for t in taus)
        print(f"{'P'+str(p):>6} {thr:>7.0f} {100*len(kept)/len(rows):>5.0f}% "
              f"{100*len(gated)/len(rows):>6.0f}% "
              f"{kb:>7.1f}/{gb:<7.1f} "
              f"{np.median(ku):>6.2f}/{np.median(gu):<6.2f} "
              f"{np.percentile(ku,90):>6.2f}/{np.percentile(gu,90):<6.2f}   {sk}")
    print("\n(kept = chunk passes the gate and IS scored; gated = skipped, no score/strike/probe)")
    print("'strike%@τ' = per-chunk P(UMF(c0) > τ) on the KEPT set — NOT the strike-counter fire rate.")


# ── PART 2 ────────────────────────────────────────────────────────────────────
def strike_counter_liveness(groups, gate_grid, tau_grid, q=3):
    """Replay atlas.expand.Expander.record() over each trajectory in temporal
    order. Count arming events (counter reaches q). After each arm we replay
    maybe_expand()'s post-arm reset — {_strikes=0, _deficit_chunks.clear(),
    _candidate=None} — verified against all three exit paths of expand.py's
    maybe_expand (see module docstring). No claim about per-regime direction."""
    dummy = torch.zeros(1)
    print(f"\n=== PART 2 — strike-counter PRELIMINARY within-trajectory liveness "
          f"(q={q}, real Expander.record) ===")
    print("PRELIMINARY: 6 chunks/trajectory, one regime each => <=2 arm opportunities "
          "per trajectory. Lower bound, not the production fire rate. Full stream = Group B.")
    print("per (regime): [n trajectories, n informative chunks, counter-arm events, "
          "trajectories with >=1 arm]\n")
    print(f"{'gate':>6} {'tau':>5}   " +
          "  ".join(f"{rg:>22}" for rg in ("R0", "R1", "R2")))
    regimes = ("R0", "R1", "R2")
    for gname, gate in gate_grid:
        for tau in tau_grid:
            cells = []
            for rg in regimes:
                trajs = [v for (r, _), v in groups.items() if r == rg]
                n_inf = arms = trajs_with_arm = 0
                for chunks in trajs:
                    exp = Expander(ExpansionConfig(tau=tau, q=q))
                    this_traj_arms = 0
                    for c in chunks:
                        gated = c["latent_disp"] <= gate
                        best_umf = None if (gated or c["umf_c0"] is None) else c["umf_c0"]
                        if best_umf is not None:
                            n_inf += 1
                        exp.record(best_umf, dummy, dummy, None)
                        if exp._strikes >= q:            # counter armed -> probe fires
                            arms += 1
                            this_traj_arms += 1
                            exp._strikes = 0             # verified: every maybe_expand()
                            exp._deficit_chunks.clear()  # post-arm exit resets exactly
                            exp._candidate = None        # these three (expand.py 125/166/171)
                    trajs_with_arm += (this_traj_arms > 0)
                cells.append(f"[{len(trajs)}t {n_inf:3d}c {arms:2d}arm {trajs_with_arm:2d}tj]")
            print(f"{gname:>6} {tau:>5.2f}   " + "  ".join(f"{x:>22}" for x in cells))
    print("\nt=trajectories  c=informative chunks  arm=counter reached q  tj=trajectories with >=1 arm")
    print("Each trajectory is 6 chunks in one regime -> this is a WITHIN-trajectory liveness probe;")
    print("the full A/B/A/B cross-regime stream version is Group B (needs P0-G charts).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.262, 0.5],
                    help="τ values to show side by side (P0-A measured first, §1.7 pinned second)")
    args = ap.parse_args()

    groups, rows = load_trajectories()
    print(f"loaded {len(rows)} chunks in {len(groups)} trajectories "
          f"({len(set(k[0] for k in groups))} regimes, "
          f"{len(rows)//len(groups)} chunks/trajectory)")

    static = [r["latent_disp"] for r in rows if r["block_disp_px"] < 1.0]
    calibration_table(rows, taus=args.taus)

    gate_grid = [("P50", pctl(static, 50)), ("P75", pctl(static, 75)),
                 ("P90", pctl(static, 90)), ("P95", pctl(static, 95)),
                 ("none", -1.0)]
    strike_counter_liveness(groups, gate_grid, tau_grid=tuple(args.taus), q=args.q)


if __name__ == "__main__":
    main()
