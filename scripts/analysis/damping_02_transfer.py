"""d=0.2 adapter-transfer eval + d=0.1 fixsteps-consistency rerun (2026-09-05).

Both cells use the SAME `analyse()` (imported unchanged from day2_n100_analysis.py,
which produced the paper's N15-n100 headline) so the CI/Wilcoxon/McNemar convention is
byte-identical to every other row in EVIDENCE_LEDGER.md. Read-only. Writes
phase0_v3/day3_dmp02_analysis.json.

d=0.2: chart phase0_v3/dmp02_transfer_ln_act_nas2 vs frozen
       phase0_v3/ladder_dmp02_baseline_nas2_fixsteps (both --fix-steps-left).
d=0.1: chart phase0_v3/dmp01_transfer_ln_act_nas2_fixsteps vs frozen
       phase0_v3/ladder_dmp01_baseline_nas2_fixsteps (both --fix-steps-left --
       supersedes the old dmp01_transfer_ln_act_nas2 / c2_settle2_dmp01_baseline_nas2
       cells, which predate the steps-left fix).
"""
import json
from pathlib import Path

from damping_dose_n100 import analyse, rows  # noqa: reuse the exact headline convention

P0 = Path("phase0_v3")

CELLS = {
    "d0.2": ("ladder_dmp02_baseline_nas2_fixsteps", "dmp02_transfer_ln_act_nas2"),
    "d0.1_fixsteps": ("ladder_dmp01_baseline_nas2_fixsteps", "dmp01_transfer_ln_act_nas2_fixsteps"),
    "d0.2_native": ("ladder_dmp02_baseline_nas2_fixsteps", "dmp02_native_ln_act_nas2"),
}

out = {}
for label, (fdir, cdir) in CELLS.items():
    fr, ch = rows(fdir), rows(cdir)
    out[label] = analyse(fr, ch, label)

(P0 / "day3_dmp02_analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
