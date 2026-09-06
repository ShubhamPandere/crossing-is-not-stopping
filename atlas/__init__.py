"""
ATLAS: Routing persistent adapters for continual JEPA world models.

This package adds three rules over one normalised quantity (UMF) onto a
permanently frozen JEPA world model:
  SELECT  — pick the chart with lowest unexplained motion            (C1)
  REFINE  — one SGD step on the selected chart, after scoring        (AdaJEPA)
  EXPAND  — commit a new chart only after verifying on unseen data   (C2)

Import order: this __init__ sets up paths from env vars before anything else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows compatibility fallback for Linux-only imports in third-party substrates (e.g. clusterscope/fcntl)
if sys.platform == "win32" and "fcntl" not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules["fcntl"] = MagicMock()

# ── Resolve ATLAS_HOME ────────────────────────────────────────────────────────
# Load .env if present (not required — contributors can set vars themselves).
_repo_root = Path(__file__).parent.parent.resolve()
_dotenv = _repo_root / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        # Only set if not already overridden in the process environment.
        if _k.strip() not in os.environ:
            os.environ[_k.strip()] = _v.strip()

ATLAS_HOME: Path = Path(os.environ.get("ATLAS_HOME", str(_repo_root))).resolve()

# ── Derived directories ───────────────────────────────────────────────────────
DATA_DIR: Path   = Path(os.environ.get("JEPAWM_DSET",  str(ATLAS_HOME / "data"))).resolve()
LOGS_DIR: Path   = Path(os.environ.get("JEPAWM_LOGS",  str(ATLAS_HOME / "logs"))).resolve()
CKPT_DIR: Path   = Path(os.environ.get("JEPAWM_CKPT",  str(ATLAS_HOME / "ckpts"))).resolve()
OUT_DIR: Path    = Path(os.environ.get("ATLAS_OUT",    str(ATLAS_HOME / "atlas_out"))).resolve()

# Export environment variables into os.environ for jepa-wms YAML expansion
os.environ["JEPAWM_DSET"] = str(DATA_DIR)
os.environ["JEPAWM_LOGS"] = str(LOGS_DIR)
os.environ["JEPAWM_CKPT"] = str(CKPT_DIR)
os.environ.setdefault("TORCH_HOME", str(ATLAS_HOME / "hub"))

# Create directories on first import so contributors never hit a FileNotFoundError.
for _d in (DATA_DIR, LOGS_DIR, CKPT_DIR, OUT_DIR, ATLAS_HOME / "hub"):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = ["ATLAS_HOME", "DATA_DIR", "LOGS_DIR", "CKPT_DIR", "OUT_DIR"]
