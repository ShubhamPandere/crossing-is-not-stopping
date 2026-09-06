"""Regression guard for P0G_FIX_PLAN §7-B1.

`modal_chart_collection.py::p0g_collect` and `p0g_finetune` must emit the collection-
defining flags such that `chart_training.py::_traj_guard` compares EQUAL — otherwise
`--load-trajs` raises a spurious "protocol mismatch" that is really just
argparse defaults leaking into `p0g_finetune` (which does not collect and so
never sets `--eval-traj-len`, `--collect-iterations`, `--num-train-trajs`,
`--train-traj-len` itself). That failure is invisible until `p0g_finetune`
first runs — i.e. after a ~4 h collection has already been paid for.

Model-free, no GPU.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "collection"))

import chart_training as run_e0  # noqa: E402
import collection_spec as mp  # noqa: E402  — the shared flag spec modal_chart_collection.py also uses


def _args_from_cmd(cmd: list[str]):
    """cmd is the list modal_phase0 passes to subprocess: [python, run_e0.py, *flags]."""
    return run_e0._build_parser().parse_args(cmd[2:])


def _collect_cmd() -> list[str]:
    d = mp._P0G_DEFAULTS
    return ["python", "scripts/collection/chart_training.py", *mp._P0G_COMMON, "--collect-only",
            "--regimes", "R2",
            *mp._p0g_flags(d["traj_len"], d["eval_traj_len"], d["num_trajs"],
                           d["num_val_trajs"], d["num_test_trajs"],
                           d["num_samples"], d["iterations"], d["nas"]),
            "--out", "/vol/phase0_v3/p0g_onpolicy"]


def _finetune_cmd() -> list[str]:
    d = mp._P0G_DEFAULTS
    return ["python", "scripts/collection/chart_training.py", *mp._P0G_COMMON,
            "--regimes", "R2", "--load-trajs", "/vol/phase0_v3/p0g_onpolicy",
            "--steps", "2000",
            *mp._p0g_flags(d["traj_len"], d["eval_traj_len"], d["num_trajs"],
                           d["num_val_trajs"], d["num_test_trajs"],
                           d["num_samples"], d["iterations"], d["nas"]),
            "--out", "/vol/phase0_v3/p0g_onpolicy"]


def test_p0g_collect_and_finetune_guards_match():
    g_collect = run_e0._traj_guard(_args_from_cmd(_collect_cmd()), "R2")
    g_finetune = run_e0._traj_guard(_args_from_cmd(_finetune_cmd()), "R2")
    diff = {k: (g_collect[k], g_finetune[k]) for k in g_collect
            if g_collect[k] != g_finetune[k]}
    assert not diff, f"§7-B1: _traj_guard mismatch (stored, current): {diff}"


if __name__ == "__main__":
    test_p0g_collect_and_finetune_guards_match()
    print("PASS")
