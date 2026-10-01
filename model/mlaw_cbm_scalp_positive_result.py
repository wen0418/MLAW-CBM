"""Formal residual-bank scalp MLAW-CBM with per-image VLM supervision.

The residual-bank ACFS image path, concept features, and disease classifier are
inherited unchanged from :class:`ConfigurableMLAWCBM`.  The only model-side
change is a learnable ``nothing`` reference prepended to each concept head for
the joint concept loss.  These six additional scalar references are never fed
to the disease classifier.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .mlaw_cbm_configurable import ConfigurableMLAWCBM
from .mvpcbm_attribute_wavelet_last_new_v5 import (
    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
)
from .mvpcbm_attribute_wavelet_last_new_v5_positive_result import (
    NOTHING_LABEL,
    NOTHING_STATE_NAME,
    prepend_nothing_logits,
)


class ScalpPositiveResultMLAWCBM(ConfigurableMLAWCBM):
    """Joint scalp MLAW-CBM with shifted per-image concept targets."""

    def __init__(
        self,
        concept_list,
        model_name="openclip",
        config=None,
        attribute_prompts=None,
    ):
        super().__init__(
            concept_list,
            model_name=model_name,
            config=config,
            attribute_prompts=attribute_prompts,
        )
        if not isinstance(
            self.attribute_wavelet_aggregator,
            ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
        ):
            raise RuntimeError(
                "Scalp VLM supervision must wrap the formal residual-bank "
                "MLAW-CBM, not the preserved direct-IDWT model"
            )
        self.base_model_variant = "formal_mlaw_cbm_residual_bank_acfs"
        self.counterfactual_construction = "X_minus_R_band"
        self.positive_result_attribute_order = tuple(self.concept_token_dict)
        if self.positive_result_attribute_order != tuple(self.attribute_names):
            raise RuntimeError(
                "Positive-result Attribute order differs from MLAW selector "
                f"order: {self.positive_result_attribute_order} != "
                f"{tuple(self.attribute_names)}"
            )
        self.nothing_logits = nn.Parameter(
            torch.zeros(len(self.positive_result_attribute_order))
        )
        self.positive_result_head_sizes = tuple(
            len(self.concept_list[name]) + 1
            for name in self.positive_result_attribute_order
        )
        self.positive_result_nothing_label = NOTHING_LABEL
        self.positive_result_nothing_state_name = NOTHING_STATE_NAME
        self.positive_result_label_encoding = (
            "0=nothing; 1..S=original zero-based state plus one"
        )
        self.disease_classifier_input = (
            f"original_{sum(len(states) for states in concept_list.values())}_"
            "state_mcsaf_fused_activations"
        )
        self.nothing_logits_used_by_disease_classifier = False

    def build_positive_result_heads(
        self,
        state_logits: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return prepend_nothing_logits(
            state_logits=state_logits,
            attribute_order=self.positive_result_attribute_order,
            nothing_logits=self.nothing_logits,
        )

    def forward(self, imgs):
        disease_logits, original_state_logits, sparse_loss = super().forward(imgs)
        positive_result_logits = self.build_positive_result_heads(
            original_state_logits
        )
        return disease_logits, positive_result_logits, sparse_loss


__all__ = ["ScalpPositiveResultMLAWCBM"]
