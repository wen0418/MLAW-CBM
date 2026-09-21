"""Public MLAW-CBM model API.

MLAW-CBM combines multi-layer ViT patch representations, attribute-conditioned
counterfactual wavelet selection, and high-frequency-guided CLS aggregation.
The implementation is the audited model previously identified as V5-4.  This
module gives the paper model a stable, version-independent import path while
preserving checkpoint compatibility with the audited implementation.
"""

from __future__ import annotations

from .mvpcbm_attribute_wavelet_last_v5_4 import mvpcbm as _AuditedMLAWCBM


class MLAWCBM(_AuditedMLAWCBM):
    """Canonical MLAW-CBM implementation."""


# Preserve the constructor name used throughout the original MVP-CBM code.
mvpcbm = MLAWCBM


__all__ = ["MLAWCBM", "mvpcbm"]
