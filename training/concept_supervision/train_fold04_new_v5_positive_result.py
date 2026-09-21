#!/usr/bin/env python3
"""Compatibility launcher for the audited New-V5 positive-result trainer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_isic2018_fold04_attribute_wavelet_last_new_v5_positive_result import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()
