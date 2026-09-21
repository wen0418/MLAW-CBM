#!/usr/bin/env python3
"""Compatibility launcher for the existing New-V5 CPU contract tests."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    runpy.run_path(
        str(
            PROJECT_ROOT
            / "training"
            / "concept_supervision"
            / "test_new_v5_positive_result_contract.py"
        ),
        run_name="__main__",
    )
