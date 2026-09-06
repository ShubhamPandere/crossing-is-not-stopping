"""Coast-model window comparison (paper section 6 / EVIDENCE_LEDGER N14b).

WHY. The published coast-model validation (N14-coast-model) estimates the block's
post-control speed v0 from the distance-to-goal trace between hold steps 5 and 15, then
predicts the drift over 15->40. Both windows sit inside the same free-coast exponential,
so predicting the second from the first is close to extrapolating an exponential fit, and
the paper leans on that result while claiming "no fitted parameters".

This script runs the harder version that the archive can still support: fit v0 from the
0->1 read -- a single 0.1 s window immediately after control stops -- and predict the same
15->40 drift. The gap to the prediction target is much wider and the fit window is one
sample, so this is a strictly stronger test of the closed form.

PRE-COMMITTED REPORTING RULE, recorded before this script was ever run:
  * The 0->1 result is reported as PRIMARY whatever value it returns.
  * The 5->15 result is kept and reported as a SECONDARY robustness row.
  * Neither is dropped, and the window is not chosen after seeing the numbers.
A lower r on a much harder test is a more convincing result than a high r on an easy one,
and reporting both shows the harder version was attempted deliberately.

KNOWN LIMITATION, shared with the published method: settled_trace records DISTANCE TO
GOAL, so any finite difference of it gives the RADIAL component of velocity, not speed.
If the block coasts laterally, v0 is underestimated. This is not new to the 0->1 window --
the 5->15 fit has exactly the same defect -- but it is stated in the paper.

Model (paper Eq. 1), d = pymunk space.damping = fraction of velocity RETAINED per second:
    D(T) = v0 * (1 - d**T) / ln(1/d)          T in seconds
so the drift over a hold-step window [a, b] (10 Hz control, so t = step/10) is
    D(a->b) = v0 * (d**(a/10) - d**(b/10)) / ln(1/d)
and v0 is recovered from an observed window drift by inverting the same expression.
"""
import argparse, glob, json, math, os, statistics, sys

CONTROL_HZ = 10.0  # pusht_env.py: control_hz = 10, so one hold step is 0.1 s


def coast_coeff(d, a, b):
    """Multiplier k such that drift over hold steps [a,b] = v0 * k, under Eq. 1."""
    return (d ** (a / CONTROL_HZ) - d ** (b / CONTROL_HZ)) / math.log(1.0 / d)


def trace_dist(rec, step):
    """Distance to goal at a hold checkpoint; step 0 is the end of active control."""
    if step == 0:
        return rec["block_pos_diff"]
    for t in rec["settled_trace"]:
        if t["step"] == step:
            return t["block_pos_diff"]
    return None


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def cell_records(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if "settled_trace" in r:
                    out.append(r)
    return out


def analyse(cells, fit_a, fit_b, pred_a, pred_b):
    """Returns (per-cell rows, pooled predicted, pooled measured)."""
    rows, P, M = [], [], []
    for name, damping, path in cells:
        recs = cell_records(path)
        pp, mm = [], []
        k_fit = coast_coeff(damping, fit_a, fit_b)
        k_pred = coast_coeff(damping, pred_a, pred_b)
        for r in recs:
            d0, d1 = trace_dist(r, fit_a), trace_dist(r, fit_b)
            p0, p1 = trace_dist(r, pred_a), trace_dist(r, pred_b)
            if None in (d0, d1, p0, p1):
                continue
            # |radial displacement| over each window; sign is irrelevant to coast magnitude
            v0 = abs(d1 - d0) / k_fit
            pp.append(v0 * k_pred)
            mm.append(abs(p1 - p0))
        if len(pp) < 2:
            continue
        rows.append({
            "cell": name, "damping": damping, "n": len(pp),
            "r": pearson(pp, mm),
            "median_pred": statistics.median(pp),
            "median_meas": statistics.median(mm),
        })
        P += pp
        M += mm
    return rows, P, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="phase0_v3")
    args = ap.parse_args()
    R = args.root

    # The 10 dose-ladder cells of the paper's Table 1: 5 non-zero dampings x 2 cadences.
    # (damping 0 is excluded: zero coast, so v0 is identically 0 and the ratio undefined.)
    cells = [
        ("d0.05 closed", 0.05, f"{R}/ladder_dmp005_baseline_nas2_fixsteps"),
        ("d0.1  closed", 0.1,  f"{R}/ladder_dmp01_baseline_nas2_fixsteps"),
        ("d0.2  closed", 0.2,  f"{R}/ladder_dmp02_baseline_nas2_fixsteps"),
        ("d0.3  closed", 0.3,  f"{R}/ladder_dmp03_baseline_nas2_fixsteps"),
        ("d0.5  closed", 0.5,  f"{R}/c2_settle2_baseline_nas2_fixsteps"),
        ("d0.05 open",   0.05, f"{R}/ladder_dmp005_baseline_nas6"),
        ("d0.1  open",   0.1,  f"{R}/c2_settle2_dmp01_baseline_nas6"),
        ("d0.2  open",   0.2,  f"{R}/ladder_dmp02_baseline_nas6"),
        ("d0.3  open",   0.3,  f"{R}/ladder_dmp03_baseline_nas6"),
        ("d0.5  open",   0.5,  f"{R}/c2_settle2_baseline_nas6"),
    ]
    missing = [c for c in cells if not glob.glob(os.path.join(c[2], "*.jsonl"))]
    if missing:
        print("MISSING CELLS (fix the paths before trusting anything below):", file=sys.stderr)
        for m in missing:
            print("  ", m[2], file=sys.stderr)

    for label, (fa, fb) in (("PRIMARY  v0 from 0->1", (0, 1)),
                            ("SECONDARY v0 from 5->15", (5, 15))):
        rows, P, M = analyse(cells, fa, fb, 15, 40)
        print(f"\n=== {label}, predicting drift over hold steps 15->40 ===")
        print(f"{'cell':14s} {'n':>3s} {'r':>7s} {'med pred':>9s} {'med meas':>9s} {'ratio':>7s}")
        for row in rows:
            ratio = row["median_pred"] / row["median_meas"] if row["median_meas"] else float("nan")
            print(f"{row['cell']:14s} {row['n']:3d} {row['r']:7.3f} "
                  f"{row['median_pred']:9.2f} {row['median_meas']:9.2f} {ratio:7.3f}")
        ratios = [p / m for p, m in zip(P, M) if m > 0]
        rs = [row["r"] for row in rows if not math.isnan(row["r"])]
        print(f"POOLED n={len(P)}  r={pearson(P, M):.4f}  "
              f"median ratio={statistics.median(ratios):.3f}  "
              f"per-cell r in [{min(rs):.3f}, {max(rs):.3f}]")


if __name__ == "__main__":
    main()
