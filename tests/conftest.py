"""Shared AiiDA pytest fixtures.

The previous file was named ``conftests.py`` and therefore was ignored by
pytest's fixture discovery.
"""

pytest_plugins = ['aiida.tools.pytest_fixtures']

# The pinned Yambopy/AiiDA dependency set still imports this NumPy 1.x alias.
# Keep the compatibility shim in the test harness rather than changing runtime
# package code or pinning NumPy for users.
import numpy as np

if not hasattr(np, 'cumproduct'):
    np.cumproduct = np.cumprod
