"""V5-7: baseline AP(X) plus a separate counterfactual-wavelet energy branch.

The complete V5-4 counterfactual selector and selective-CLS branch are kept
unchanged.  Only the downstream concept representation is changed.  It is
factorized into an original-patch semantic path and a pure-wavelet evidence
path:

    R = softplus(s_l) * H*                              [B, 196, 768]
    C = AP_7(X)                                         [B,   7, 768]
    E = sqrt(AP_7(R^2) + eps) - sqrt(eps)              [B,   7, 768]
    Z = MLP_content(C) + g[l,a] * MLP_energy(E)         [B,   7, 512]

The content MLP is the inherited baseline projector.  The energy projector is
an independent two-linear-layer 768-to-512-to-512 MLP with zero biases, so zero
wavelet energy produces exactly zero evidence.  Positive per-layer/per-
Attribute softplus gates initialize to one.  No LayerNorm, absolute-value patch
residual, selector attention P, or Top-K map enters the concept path.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .mvpcbm_attribute_wavelet_last_v5_4 import (
    mvpcbm as PreLayerNormWaveletAPV5_4,
)


INITIAL_WAVELET_ENERGY_GATE = 1.0


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("softplus initialization must be finite and positive")
    return math.log(math.expm1(value))


class OriginalAPWaveletEnergyMixin:
    """Use AP(X) as content and RMS-pooled alpha*H* as additive evidence."""

    concept_pooling = "baseline_ap_of_x_plus_separate_wavelet_rms_energy"
    concept_content_input = "original_patch_X"
    concept_wavelet_input = "R_equals_scaled_counterfactual_H_star"
    concept_pooling_uses_attribute_attention = False
    concept_pooling_uses_selector_topk = False
    concept_similarity = "inherited_logit_scaled_dot_product"

    def __init__(
        self,
        concept_list,
        model_name="openclip",
        config=None,
        **model_kwargs,
    ):
        super().__init__(
            concept_list,
            model_name=model_name,
            config=config,
            **model_kwargs,
        )
        num_layers = len(self.model.visual.trunk.blocks)
        num_attributes = len(self.concept_token_dict)

        # A compact separate projector lets non-negative wavelet magnitudes
        # learn a mapping without forcing the signed content MLP to serve two
        # input distributions.  Zero biases guarantee P_energy(0) == 0.  It is
        # deliberately two linear layers rather than a copy of this project's
        # much deeper inherited content projector, limiting added capacity.
        self.wavelet_energy_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        for module in self.wavelet_energy_projector.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        initial_raw_gate = _inverse_softplus(INITIAL_WAVELET_ENERGY_GATE)
        self.raw_wavelet_energy_gates = nn.Parameter(
            torch.full((num_layers, num_attributes), initial_raw_gate)
        )

    def wavelet_energy_gates(self) -> torch.Tensor:
        return F.softplus(self.raw_wavelet_energy_gates)

    def _original_ap_plus_wavelet_energy_concept_features(
        self,
        patch_tokens: torch.Tensor,
        filtered_high_tokens: torch.Tensor,
        high_feature_scale: torch.Tensor,
        layer_index: int,
        return_details: bool = False,
    ):
        """Return the clean AP(X) + separately projected RMS-energy feature."""

        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens must be [B,N,D], received "
                f"{tuple(patch_tokens.shape)}"
            )
        if tuple(filtered_high_tokens.shape) != tuple(patch_tokens.shape):
            raise ValueError(
                "H* and X must have identical shapes; received "
                f"H*={tuple(filtered_high_tokens.shape)} and "
                f"X={tuple(patch_tokens.shape)}"
            )
        if high_feature_scale.numel() != 1:
            raise ValueError(
                "high_feature_scale must be scalar, received "
                f"{tuple(high_feature_scale.shape)}"
            )
        if not 0 <= layer_index < self.raw_wavelet_energy_gates.shape[0]:
            raise IndexError(
                f"layer_index={layer_index} outside "
                f"[0, {self.raw_wavelet_energy_gates.shape[0]})"
            )

        residual = high_feature_scale * filtered_high_tokens

        # Both paths use exactly the same seven baseline horizontal bins.
        # Content receives original X only; signed R never enters this path.
        content_visual = self.Avg.sampler(
            patch_tokens.transpose(1, 2)
        ).transpose(1, 2)
        residual_mean_square = self.Avg.sampler(
            residual.square().transpose(1, 2)
        ).transpose(1, 2)
        epsilon = float(self.attribute_wavelet_eps)
        energy_visual = torch.sqrt(residual_mean_square + epsilon) - math.sqrt(
            epsilon
        )

        content_projected = self.Avg.mlp_projector(content_visual)
        energy_projected = self.wavelet_energy_projector(energy_visual)
        gate = self.wavelet_energy_gates()[layer_index].view(1, -1, 1)
        fused_projected = content_projected + gate * energy_projected

        if not return_details:
            return patch_tokens, fused_projected
        return patch_tokens, fused_projected, {
            "residual": residual,
            "content_visual": content_visual,
            "energy_visual": energy_visual,
            "content_projected": content_projected,
            "energy_projected": energy_projected,
            "energy_gate": gate,
            "fused_projected": fused_projected,
        }

    def forward(self, imgs):
        self.visual_features.clear()
        self.model(imgs, None)

        layer_features = []
        num_layers = len(self.model.visual.trunk.blocks)
        for layer_index in range(num_layers):
            block_tokens = self.visual_features[layer_index]
            patch_tokens = block_tokens[:, 1:, :]
            batch_size = patch_tokens.size(0)

            # The counterfactual selector and selective CLS are exactly V5-4.
            attribute_wavelet_cls, selector_details = (
                self.build_attribute_wavelet_cls(
                    patch_tokens,
                    layer_index=layer_index,
                    return_details=True,
                )
            )
            filtered_high_tokens = selector_details["filtered_high_tokens"]
            high_feature_scale = selector_details["high_feature_scale"]

            class_feature = self.fc_confidence(
                attribute_wavelet_cls
            ).unsqueeze(1)
            global_feature_importance = []
            for key in self.global_attr_concepts.keys():
                global_similarity = (
                    class_feature
                    @ self.global_attr_concepts[key]
                    .repeat(batch_size, 1, 1)
                    .permute(0, 2, 1)
                ).squeeze(1)
                global_feature_importance.append(global_similarity)
            global_feature_importance = torch.sigmoid(
                torch.cat(global_feature_importance, dim=1)
            )
            global_preference = self.ImportanceThresholding(
                global_feature_importance
            )

            _, projected_features = (
                self._original_ap_plus_wavelet_energy_concept_features(
                    patch_tokens,
                    filtered_high_tokens,
                    high_feature_scale,
                    layer_index=layer_index,
                )
            )
            del selector_details

            image_logits_per_attribute = []
            for attribute_index, key in enumerate(self.concept_token_dict.keys()):
                concept_scores = (
                    self.logit_scale
                    * projected_features[
                        :, attribute_index : attribute_index + 1, :
                    ]
                    @ self.concept_token_dict[key]
                    .repeat(batch_size, 1, 1)
                    .permute(0, 2, 1)
                ).squeeze(1)
                attribute_preference = global_preference[
                    :, attribute_index : attribute_index + 1
                ]
                image_logits_per_attribute.append(
                    attribute_preference * concept_scores
                )
            layer_features.append(torch.cat(image_logits_per_attribute, dim=-1))

        layer_score = torch.stack(layer_features, dim=1)
        image_logits, _, sparse_loss, _ = self.SparseConceptSpecificAttentionPreAgg(
            layer_score
        )

        image_logits_dict = {}
        concept_index = 0
        for key in self.concept_token_dict.keys():
            num_states = len(self.concept_list[key])
            image_logits_dict[key] = image_logits[
                :, concept_index : concept_index + num_states
            ]
            concept_index += num_states

        cls_logits = self.cls_head(image_logits)
        self.visual_features.clear()
        return cls_logits, image_logits_dict, sparse_loss


class mvpcbm(OriginalAPWaveletEnergyMixin, PreLayerNormWaveletAPV5_4):
    """ISIC2018 V5-7: original AP semantics plus wavelet RMS evidence."""

    pass
