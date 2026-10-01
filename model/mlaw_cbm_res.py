"""Preserved pre-refactor MLAW-CBM with direct-IDWT counterfactuals."""

from __future__ import annotations

from .mvpcbm_attribute_wavelet_last_v5_4 import mvpcbm as _LegacyMLAWCBM


class MLAWCBM_res(_LegacyMLAWCBM):
    """Original MLAW-CBM retained for checkpoint and result comparisons."""


MLAWCBMRes = MLAWCBM_res
mvpcbm = MLAWCBM_res


__all__ = ["MLAWCBM_res", "MLAWCBMRes", "mvpcbm"]
