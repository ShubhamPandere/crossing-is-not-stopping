"""P0-G collection-protocol spec — the single source of truth for the flags
that feed run_e0.py::_traj_guard (§7-B1).

Lives in its own module (no `modal` import) so both modal/modal_phase0.py and
tests/test_p0g_guard.py can use it. p0g_collect and p0g_finetune both default
their signatures off _P0G_DEFAULTS and emit _p0g_flags(...), so p0g_finetune
can never fall back to argparse defaults that don't match what p0g_collect
stored (which made --load-trajs raise a spurious "protocol mismatch").
"""
from __future__ import annotations

# Every value run_e0.py::_traj_guard compares, in ONE place.
_P0G_DEFAULTS = dict(traj_len=30, eval_traj_len=30, num_trajs=100,
                     num_val_trajs=8, num_test_trajs=8,
                     num_samples=300, iterations=10, nas=2)

# --num-test-trajs (P3) is in _p0g_flags, NOT here, so it is passed exactly once.
_P0G_COMMON = ["--kinds", "ln_act", "--data-source", "closed_loop"]


def _p0g_flags(traj_len: int, eval_traj_len: int, num_trajs: int, num_val_trajs: int,
               num_test_trajs: int, num_samples: int, iterations: int, nas: int) -> list[str]:
    """The guard-relevant flags — emitted IDENTICALLY by p0g_collect and
    p0g_finetune (§7-B1)."""
    return ["--num-train-trajs", str(num_trajs),
            "--train-traj-len", str(traj_len),
            "--num-val-trajs", str(num_val_trajs),
            "--num-test-trajs", str(num_test_trajs),
            "--eval-traj-len", str(eval_traj_len),
            "--collect-num-samples", str(num_samples),
            "--collect-iterations", str(iterations),
            "--collect-num-act-stepped", str(nas)]
