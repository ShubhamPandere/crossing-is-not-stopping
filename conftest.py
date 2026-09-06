"""
conftest.py — pytest configuration.

Skips tests that require torch if torch is not installed.
This allows test_stats.py and test_streams.py to run without any GPU or torch.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip torch-dependent tests if torch is not available."""
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_torch = pytest.mark.skip(reason="torch not installed — skipping GPU tests")
        for item in items:
            if "test_score" in item.nodeid:
                item.add_marker(skip_torch)
