"""New-V5 with per-image positive-result concept supervision.

This controlled variant keeps the complete New-V5 selector and the original
MVP-CBM disease-classification path.  It changes only the concept-supervision
interface: every Attribute head receives a learnable ``nothing`` reference at
index 0, while its existing text-defined states are shifted to indices 1..S.

The disease classifier deliberately continues to consume the original 34
MCSAF activations.  The seven ``nothing`` references are used only by the
per-Attribute concept cross-entropy loss and cannot become extra disease
features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .mvpcbm_attribute_wavelet_last_new_v5 import (
    mvpcbm as AttributeWaveletLaSTNewV5,
)


NOTHING_LABEL = 0
NOTHING_STATE_NAME = "nothing visibly supported"


def prepend_nothing_logits(
    state_logits: Mapping[str, torch.Tensor],
    attribute_order: Sequence[str],
    nothing_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Prepend one scalar ``nothing`` reference to every Attribute head."""

    attribute_order = tuple(attribute_order)
    if nothing_logits.ndim != 1 or nothing_logits.numel() != len(attribute_order):
        raise ValueError(
            "nothing_logits must have shape [num_attributes], received "
            f"{tuple(nothing_logits.shape)} for {len(attribute_order)} Attributes"
        )
    if tuple(state_logits) != attribute_order:
        raise ValueError(
            "Concept head order differs from Attribute order: "
            f"heads={tuple(state_logits)}, attributes={attribute_order}"
        )

    result: dict[str, torch.Tensor] = {}
    batch_size = None
    for attribute_index, attribute_name in enumerate(attribute_order):
        logits = state_logits[attribute_name]
        if logits.ndim != 2:
            raise ValueError(
                f"{attribute_name} logits must be [B,S], received "
                f"{tuple(logits.shape)}"
            )
        if batch_size is None:
            batch_size = logits.size(0)
        elif logits.size(0) != batch_size:
            raise ValueError("All concept heads must have the same batch size")
        nothing = nothing_logits[attribute_index].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        result[attribute_name] = torch.cat(
            (nothing.expand(logits.size(0), 1), logits),
            dim=1,
        )
    return result


class mvpcbm(AttributeWaveletLaSTNewV5):
    """Joint MVP-CBM/New-V5 with shifted per-image concept targets."""

    def __init__(self, concept_list, model_name="openclip", config=None):
        super().__init__(concept_list, model_name=model_name, config=config)
        self.positive_result_attribute_order = tuple(self.concept_token_dict)
        if self.positive_result_attribute_order != tuple(self.attribute_names):
            raise RuntimeError(
                "Positive-result Attribute order differs from New-V5 selector "
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
        # Reuse New-V5's inherited V3 forward verbatim.  In particular, this
        # computes the disease logits from the original 34-state MCSAF output.
        cls_logits, original_state_logits, sparse_loss = super().forward(imgs)
        positive_result_logits = self.build_positive_result_heads(
            original_state_logits
        )
        return cls_logits, positive_result_logits, sparse_loss
