"""Public API for the MLAW-CBM wavelet-energy ablation.

This is the audited model previously identified as V5-7.  It keeps the same
multi-layer attribute/high-frequency CLS guidance as MLAW-CBM, while separating
the downstream concept representation into original-patch content and wavelet
RMS-energy evidence.
"""

from __future__ import annotations

from .mvpcbm_attribute_wavelet_last_v5_7 import mvpcbm as _AuditedMLAWCBMEnergy


class MLAWCBMEnergy(_AuditedMLAWCBMEnergy):
    """Canonical MLAW-CBM-Energy implementation."""


mvpcbm = MLAWCBMEnergy


__all__ = ["MLAWCBMEnergy", "mvpcbm"]
