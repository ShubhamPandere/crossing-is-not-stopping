"""The keep-planning control (EVIDENCE_LEDGER N17, FIXLOG V3-26).

The paper's settled criterion commands a ZERO action for 40 steps after control stops. The
obvious reviewer objection is that a real deployment would keep planning. These cells are that
control: same config, same seeds, same tasks, but the planner keeps replanning through the hold.

READ THIS BEFORE USING ANY PAIRED NUMBER BELOW.
The two arms share TASKS (identical initial block pose and goal, 0 mismatches) but NOT realized
trajectories. The episode phase precedes the hold and therefore cannot be caused by it, yet the
end-of-control state differs in 19 of 20 episodes per cell (median 3-12 px, max 116 px), with
contact counts differing by up to 6 and crossing SR differing by up to 10 points. This is
cross-run CUDA nondeterminism amplified through contact-rich physics, the same class of effect
this project traced during the G7 sub-study. Consequence: this is a randomized task-level
comparison, not a bit-identical paired one, and the paired p-values below carry trajectory
noise as well as the hold-policy effect. They are reported, but the load-bearing claim is the
WITHIN-ARM survival statistic, which needs no pairing at all.
"""
import glob, json, math, os, statistics as st
from pathlib import Path

P = str(Path(__file__).resolve().parent.parent.parent / "phase0_v3")
CELLS = [
    ("d=0.1", f"{P}/ladder_dmp01_baseline_nas2_fixsteps", f"{P}/keepplan_dmp01_baseline_nas2"),
    ("d=0.3", f"{P}/ladder_dmp03_baseline_nas2_fixsteps", f"{P}/keepplan_dmp03_baseline_nas2"),
    ("d=0.5", f"{P}/c2_settle2_baseline_nas2_fixsteps",   f"{P}/keepplan_dmp05_baseline_nas2"),
]


def load(d):
    f = sorted(glob.glob(os.path.join(d, "*.jsonl")))[0]
    return {json.loads(l)["episode"]: json.loads(l) for l in open(f)}


def speed(rec, step):
    for v in rec.get("settled_velocity_trace", []):
        if v["step"] == step:
            return math.hypot(*v["block_vel"])
    return None


print("=" * 78)
print("WITHIN-ARM (no pairing needed): do threshold crossings become resting states?")
print("=" * 78)
print(f"{'cell':6s} {'arm':9s} {'crossings':>10s} {'of those, settled':>18s} {'settled SR':>11s}")
for name, zdir, pdir in CELLS:
    for arm, recs in (("zero", load(zdir)), ("planner", load(pdir))):
        s = sorted(recs)
        cross = [i for i in s if recs[i]["passthrough_success"]]
        surv = sum(recs[i]["settled_success"] for i in cross)
        sr = sum(recs[i]["settled_success"] for i in s) / len(s)
        print(f"{name:6s} {arm:9s} {len(cross):>4d}/{len(s):<5d} {surv:>13d}/{len(cross):<4d} {sr:>11.2f}")
    print()

print("=" * 78)
print("Is the block still moving at the END of a 40-step PLANNER-DRIVEN hold?")
print("(simulator-reported speed, px/s; only the planner arm logs velocity)")
print("=" * 78)
for name, _, pdir in CELLS:
    K = load(pdir)
    s = sorted(K)
    v0 = [speed(K[i], 0) for i in s]
    v40 = [speed(K[i], 40) for i in s]
    v0 = [v for v in v0 if v is not None]; v40 = [v for v in v40 if v is not None]
    print(f"{name:6s} speed at hold step 0: median {st.median(v0):7.2f}   "
          f"at step 40: median {st.median(v40):7.2f}   "
          f"episodes still moving (>1 px/s) at 40: {sum(v > 1 for v in v40)}/{len(v40)}")
print()

print("=" * 78)
print("TASK-LEVEL comparison (shares tasks, NOT trajectories -- see module docstring)")
print("=" * 78)


def wilcoxon(diffs):
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    W = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def boot_ci(diffs, iters=20000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(st.mean(rng.choice(diffs) for _ in range(n)) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


for name, zdir, pdir in CELLS:
    Z, K = load(zdir), load(pdir)
    s = sorted(set(Z) & set(K))
    zs = [Z[i]["settled_block_pos_diff"] for i in s]
    ks = [K[i]["settled_block_pos_diff"] for i in s]
    diffs = [k - z for z, k in zip(zs, ks)]
    lo, hi = boot_ci(diffs)
    nbad = sum(abs(Z[i]["block_pos_diff"] - K[i]["block_pos_diff"]) > 1e-9 for i in s)
    print(f"{name}: settled distance zero {st.mean(zs):.1f} px, planner {st.mean(ks):.1f} px; "
          f"delta {st.mean(diffs):+.1f} [{lo:+.1f}, {hi:+.1f}], p={wilcoxon(diffs):.3g}")
    print(f"       (task pairing valid: 0 init mismatches; trajectory pairing INVALID: "
          f"{nbad}/{len(s)} episodes diverge before the hold)")
