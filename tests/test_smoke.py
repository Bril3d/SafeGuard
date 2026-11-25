"""Foundation smoke tests for SafeGuard package layout."""

from __future__ import annotations

import importlib


def test_src_packages_importable() -> None:
    """Core source packages are importable without training dependencies."""
    for module_name in (
        "src",
        "src.data",
        "src.models",
        "src.compliance",
        "src.tracking",
        "src.utils",
    ):
        module = importlib.import_module(module_name)
        assert module is not None
