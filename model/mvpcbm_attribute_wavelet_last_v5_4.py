"""new V5-4: new-V5 selector plus pre-LayerNorm wavelet residual AP.

The complete counterfactual new-V5 selector and selective-CLS branch are kept
unchanged.  In particular, selector semantics still use

    W* = LN(X + softplus(s_l) H*)

V5-4 changes only the downstream concept input.  The inherited baseline
AdaptiveAvgPool1d(A) and MLP projector receive the pre-LayerNorm residual sum:

    X_cf = X + softplus(s_l) H*                         [B, N, 768]
    AvgPoolProjector(X_cf)                              [B, A, 512]

No detach, absolute value, new gate, new projector, Attribute attention P,
selector Top-K index, or vote map is introduced into the concept path.
"""

from __future__ import annotations

import torch

from .mvpcbm_attribute_wavelet_last_new_v5 import (
    mvpcbm as CounterfactualBandSelectedWaveletLaSTNewV5,
)


class PreLayerNormWaveletAdaptivePoolingMixin:
    """Feed X + alpha*H* (before selector LayerNorm) into baseline AP."""

    concept_pooling = "baseline_adaptive_average_pooling_of_pre_norm_wavelet_patch"
    concept_pooling_input = "X_plus_scaled_counterfactual_H_star_before_layernorm"
    concept_pooling_uses_attribute_attention = False
    concept_pooling_uses_selector_topk = False
    concept_similarity = "inherited_logit_scaled_dot_product"

    def _pre_norm_wavelet_ap_concept_features(
        self,
        patch_tokens: torch.Tensor,
        filtered_high_tokens: torch.Tensor,
        high_feature_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact AP input X+alpha*H* and projected [B,A,512] features."""

        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens must be [B,N,D], received "
                f"{tuple(patch_tokens.shape)}"
            )
        if filtered_high_tokens.ndim != 3:
            raise ValueError(
                "filtered_high_tokens must be [B,N,D], received "
                f"{tuple(filtered_high_tokens.shape)}"
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

        concept_patch = patch_tokens + high_feature_scale * filtered_high_tokens
        projected_features = self.Avg(concept_patch)
        return concept_patch, projected_features

    def forward(self, imgs):
        self.visual_features.clear()
        self.model(imgs, None)

        layer_features = []
        num_layers = len(self.model.visual.trunk.blocks)
        for layer_index in range(num_layers):
            block_tokens = self.visual_features[layer_index]
            patch_tokens = block_tokens[:, 1:, :]
            batch_size = patch_tokens.size(0)

            # Selector/CLS is exactly new-V5: it still constructs and consumes
            # post-LayerNorm W*.  The details expose the same H* and alpha for
            # the separate, pre-LayerNorm concept ablation.
            attribute_wavelet_cls, selector_details = (
                self.build_attribute_wavelet_cls(
                    patch_tokens,
                    layer_index=layer_index,
                    return_details=True,
                )
            )
            filtered_high_tokens = selector_details["filtered_high_tokens"]
            high_feature_scale = selector_details["high_feature_scale"]
            del selector_details

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

            # V5-4's only architectural change relative to new-V5: retain the
            # complete baseline AP -> MLP path, while adding the selected H*
            # residual to X without applying selector-side LayerNorm.
            _concept_pooling_input, projected_features = (
                self._pre_norm_wavelet_ap_concept_features(
                    patch_tokens,
                    filtered_high_tokens,
                    high_feature_scale,
                )
            )
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


class mvpcbm(
    PreLayerNormWaveletAdaptivePoolingMixin,
    CounterfactualBandSelectedWaveletLaSTNewV5,
):
    """ISIC2018 new V5-4 (2x2 ablation C)."""

    pass
