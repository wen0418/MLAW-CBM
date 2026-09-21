"""V5-4 with per-image positive-result concept supervision.

This controlled variant keeps V5-4's complete image and disease-classification
path.  It only prepends one learnable ``nothing`` logit to each of the seven
Attribute heads used by the concept cross-entropy loss.  The disease classifier
continues to consume V5-4's original 34 MCSAF activations; the seven additional
``nothing`` logits never become disease-classifier features.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .mvpcbm_attribute_wavelet_last_new_v5_positive_result import (
    NOTHING_LABEL,
    NOTHING_STATE_NAME,
    prepend_nothing_logits,
)
from .mvpcbm_attribute_wavelet_last_v5_4 import mvpcbm as AttributeWaveletLaSTV5_4


class mvpcbm(AttributeWaveletLaSTV5_4):
    """Joint MVP-CBM/V5-4 with shifted per-image concept targets."""

    def __init__(self, concept_list, model_name="openclip", config=None):
        super().__init__(concept_list, model_name=model_name, config=config)
        self.positive_result_attribute_order = tuple(self.concept_token_dict)
        if self.positive_result_attribute_order != tuple(self.attribute_names):
            raise RuntimeError(
                "Positive-result Attribute order differs from V5-4 selector "
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
        self.disease_classifier_input = "original_34_mcsaf_fused_activations"
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
        # V5-4 computes both disease logits and the original 34 state logits.
        # Only the latter receive the additional per-Attribute nothing logit.
        cls_logits, original_state_logits, sparse_loss = super().forward(imgs)
        positive_result_logits = self.build_positive_result_heads(
            original_state_logits
        )
        return cls_logits, positive_result_logits, sparse_loss
