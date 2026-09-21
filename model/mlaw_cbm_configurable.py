"""Dataset-configurable MLAW-CBM models for skin and scalp experiments."""

from __future__ import annotations

from .mvpcbm_attribute_wavelet_last_new_v5_configurable import (
    ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5,
)
from .mvpcbm_attribute_wavelet_last_v5_4 import (
    PreLayerNormWaveletAdaptivePoolingMixin,
)
from .mvpcbm_attribute_wavelet_last_v5_7 import OriginalAPWaveletEnergyMixin


class ConfigurableMLAWCBM(
    PreLayerNormWaveletAdaptivePoolingMixin,
    ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5,
):
    """MLAW-CBM with dataset-provided ordered Attribute prompts."""


class ConfigurableMLAWCBMEnergy(
    OriginalAPWaveletEnergyMixin,
    ConfigurableMLAWCBM,
):
    """MLAW-CBM-Energy with dataset-provided ordered Attribute prompts."""


__all__ = ["ConfigurableMLAWCBM", "ConfigurableMLAWCBMEnergy"]
