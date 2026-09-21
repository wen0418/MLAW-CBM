"""Dataset-configurable counterfactual band-selected new V5.

This wrapper keeps the audited new-V5 selector mathematics unchanged while
allowing the ordered Attribute-only prompts to come from the scalp experiment
configuration instead of the seven ISIC2018 prompts.
"""

from __future__ import annotations

from collections.abc import Mapping

from .mvpcbm_attribute_wavelet_last_new_v5 import (
    DEFAULT_ROUTE_TEMPERATURE,
    ROUTE_NAMES,
    CounterfactualBandSelectedWaveletAggregator,
)
from .mvpcbm_attribute_wavelet_last_v3_configurable import (
    ConfigurableAttributeWaveletLaSTV3,
)


class ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5(
    ConfigurableAttributeWaveletLaSTV3
):
    """new V5 with externally supplied, order-checked Attribute prompts."""

    def __init__(
        self,
        concept_list,
        *,
        attribute_prompts: Mapping[str, str],
        model_name: str = "openclip",
        config=None,
    ):
        super().__init__(
            concept_list,
            attribute_prompts=attribute_prompts,
            model_name=model_name,
            config=config,
        )

        num_layers = len(self.model.visual.trunk.blocks)
        hidden_dim = self.fc_confidence.in_features
        attribute_dim = self.fc_confidence.out_features
        route_temperature = float(
            getattr(
                config,
                "counterfactual_route_temperature",
                DEFAULT_ROUTE_TEMPERATURE,
            )
        )

        self.attribute_wavelet_aggregator = (
            CounterfactualBandSelectedWaveletAggregator(
                hidden_dim=hidden_dim,
                attribute_dim=attribute_dim,
                num_layers=num_layers,
                num_attributes=len(self.attribute_names),
                top_k=self.attribute_wavelet_top_k,
                attribute_temperature=self.attribute_temperature,
                route_temperature=route_temperature,
                initial_high_feature_scale=self.initial_high_feature_scale,
                initial_high_score_weight=self.initial_high_score_weight,
                eps=self.attribute_wavelet_eps,
            )
        )
        self.counterfactual_route_temperature = route_temperature
        self.counterfactual_route_names = ROUTE_NAMES
        self.counterfactual_ablation = "direct IDWT with one detail band zeroed"
        self.counterfactual_negative_effect = "exclude band from enhancement"
        self.counterfactual_raw_magnitude_used = True
        self.counterfactual_high_residual_injected = True
        self.counterfactual_ll_conv_gate_used = False
        self.attribute_selector_uses_concept_states = False
        self.attribute_wavelet_old_cls_used = False


mvpcbm = ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5
