"""Public MLAW-CBM model API.

MLAW-CBM combines multi-layer ViT patch representations, attribute-conditioned
counterfactual wavelet selection, and high-frequency-guided CLS aggregation.
The formal model uses a vectorized high-frequency residual bank and constructs
each counterfactual as ``X - R_band``.  The former direct-IDWT implementation
is preserved as :mod:`model.mlaw_cbm_res`.
"""

from __future__ import annotations

from .mvpcbm_attribute_wavelet_last_new_v5 import (
    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
)
from .mvpcbm_attribute_wavelet_last_v5_4 import mvpcbm as _AuditedMLAWCBM


class MLAWCBM(_AuditedMLAWCBM):
    """Canonical residual-bank MLAW-CBM implementation."""

    attribute_wavelet_aggregator_class = (
        ResidualSubtractionCounterfactualBandSelectedWaveletAggregator
    )


# Preserve the constructor name used throughout the original MVP-CBM code.
mvpcbm = MLAWCBM


__all__ = ["MLAWCBM", "mvpcbm"]
