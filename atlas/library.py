"""
library.py — Chart library: add / evict / clone / utilisation tracking.

The library always contains at least c₀ (the identity chart, never refined,
always competes in scoring). Charts are disjoint: updating one cannot alter
another's parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from atlas.chart import Chart, ChartKind


class Library:
    """
    A fixed-capacity library of Chart objects.

    c₀ is always index 0 and is never refined (callers must enforce this).
    """

    def __init__(self, c0: Chart, max_size: int = 10) -> None:
        """
        Args:
            c0:       The identity chart (index 0). Must be created from the
                      frozen predictor BEFORE any refinement occurs.
            max_size: Hard cap on library size (K_max = 10 in the plan).
        """
        if max_size < 1:
            raise ValueError(f"max_size must be ≥ 1, got {max_size}")
        self._charts: list[Chart] = [c0]
        self.max_size = max_size

    # ── Read ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._charts)

    def __iter__(self) -> Iterator[Chart]:
        return iter(self._charts)

    def __getitem__(self, idx: int) -> Chart:
        return self._charts[idx]

    @property
    def c0(self) -> Chart:
        return self._charts[0]

    def is_full(self) -> bool:
        return len(self._charts) >= self.max_size

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, chart: Chart) -> int:
        """
        Add a chart to the library.

        Raises:
            RuntimeError: if the library is already at max_size.

        Returns:
            Index of the newly added chart.
        """
        if self.is_full():
            raise RuntimeError(
                f"Library is full ({self.max_size} charts). "
                "Increase K_max or implement eviction before adding more charts."
            )
        self._charts.append(chart)
        return len(self._charts) - 1

    def clone_from(self, source_idx: int) -> Chart:
        """
        Create a deep copy of chart at *source_idx* (warm-start for expansion).

        The clone is NOT added to the library; callers decide whether to add it
        after verification.
        """
        if source_idx < 0 or source_idx >= len(self._charts):
            raise IndexError(f"source_idx {source_idx} out of range [0, {len(self._charts)})")
        return self._charts[source_idx].clone()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Save all charts to *directory*/chart_{i}.pt."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, chart in enumerate(self._charts):
            chart.save(directory / f"chart_{i:03d}.pt")
        # Save metadata
        torch.save({"n_charts": len(self._charts), "max_size": self.max_size},
                   directory / "library_meta.pt")

    @classmethod
    def load(cls, directory: str | Path, predictor) -> "Library":
        """Restore a library saved by Library.save()."""
        directory = Path(directory)
        meta_path = directory / "library_meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Library metadata not found at {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        charts = []
        for i in range(meta["n_charts"]):
            charts.append(Chart.load(directory / f"chart_{i:03d}.pt", predictor))
        lib = object.__new__(cls)
        lib._charts = charts
        lib.max_size = meta["max_size"]
        return lib

    def __repr__(self) -> str:
        return (
            f"Library(n={len(self._charts)}/{self.max_size}, "
            f"kinds={[c.kind for c in self._charts]})"
        )
