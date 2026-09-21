"""Attribute-guided wavelet LaST aggregation for refined MVP-CBM (V3).

V3 is deliberately narrower than the prototype V2 experiment.  It introduces
no concept-state matching, prototypes, prototype diversity loss, or prototype
cross-attention.  For every ViT layer it:

1. extracts a spatially aligned high-frequency residual with a fixed Haar DWT;
2. lets seven attribute-only text queries score wavelet-enhanced patches;
3. uses the union of those attribute maps to gate channel-wise high-frequency
   energy;
4. selects K patches independently for every channel; and
5. aggregates the corresponding original ViT patch values into a new CLS.

The original ViT CLS remains in the transformer stream but is never read by
the V3 selector.  The generated CLS is used only in MVP-CBM's existing global
attribute-preference branch.  The original downstream patch-to-concept path is
kept unchanged so concept-state interaction remains a later-stage experiment.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mvpcbm_refined import mvpcbm as RefinedMVPCBM


DEFAULT_TOP_K = 98
DEFAULT_ATTRIBUTE_TEMPERATURE = 0.07
DEFAULT_INITIAL_HIGH_FEATURE_SCALE = 0.1
DEFAULT_INITIAL_HIGH_SCORE_WEIGHT = 1.0
DEFAULT_EPS = 1e-6


ATTRIBUTE_PROMPTS = {
    "color": "this is a dermoscopic image; focus on the color attribute of the lesion",
    "shape": "this is a dermoscopic image; focus on the shape attribute of the lesion",
    "border": "this is a dermoscopic image; focus on the border attribute of the lesion",
    "dermoscopic patterns": (
        "this is a dermoscopic image; focus on the dermoscopic pattern attribute "
        "of the lesion"
    ),
    "texture": "this is a dermoscopic image; focus on the texture attribute of the lesion",
    "symmetry": "this is a dermoscopic image; focus on the symmetry attribute of the lesion",
    "elevation": (
        "this is a dermoscopic image; focus on the elevation attribute of the lesion"
    ),
}


def _inverse_softplus(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("Softplus initialization must be finite and positive")
    return math.log(math.expm1(value))


class HaarDWT2D(nn.Module):
    """Fixed orthonormal one-level 2-D Haar transform."""

    def forward(self, feature_map: torch.Tensor):
        if feature_map.ndim != 4:
            raise ValueError(
                "feature_map must have shape [batch, channels, height, width], "
                f"received {tuple(feature_map.shape)}"
            )
        height, width = feature_map.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                "Haar DWT requires even spatial dimensions, received "
                f"{height}x{width}"
            )

        top_left = feature_map[:, :, 0::2, 0::2]
        top_right = feature_map[:, :, 0::2, 1::2]
        bottom_left = feature_map[:, :, 1::2, 0::2]
        bottom_right = feature_map[:, :, 1::2, 1::2]
        ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
        hl = (-top_left + top_right - bottom_left + bottom_right) * 0.5
        lh = (-top_left - top_right + bottom_left + bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """Exact inverse of :class:`HaarDWT2D`."""

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError("All inverse-Haar bands must have equal shapes")
        top_left = (ll - hl - lh + hh) * 0.5
        top_right = (ll + hl - lh - hh) * 0.5
        bottom_left = (ll - hl + lh - hh) * 0.5
        bottom_right = (ll + hl + lh + hh) * 0.5

        batch_size, channels, height, width = ll.shape
        output = ll.new_empty(batch_size, channels, height * 2, width * 2)
        output[:, :, 0::2, 0::2] = top_left
        output[:, :, 0::2, 1::2] = top_right
        output[:, :, 1::2, 0::2] = bottom_left
        output[:, :, 1::2, 1::2] = bottom_right
        return output


class AttributeWaveletLaSTAggregator(nn.Module):
    """Create one attribute-guided wavelet selective CLS per ViT layer."""

    def __init__(
        self,
        hidden_dim: int,
        attribute_dim: int,
        num_layers: int,
        num_attributes: int,
        top_k: int = DEFAULT_TOP_K,
        attribute_temperature: float = DEFAULT_ATTRIBUTE_TEMPERATURE,
        initial_high_feature_scale: float = DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
        initial_high_score_weight: float = DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
        eps: float = DEFAULT_EPS,
    ):
        super().__init__()
        if hidden_dim <= 0 or attribute_dim <= 0:
            raise ValueError("hidden_dim and attribute_dim must be positive")
        if num_layers <= 0 or num_attributes <= 0:
            raise ValueError("num_layers and num_attributes must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if attribute_temperature <= 0.0 or not math.isfinite(attribute_temperature):
            raise ValueError("attribute_temperature must be finite and positive")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive")

        self.hidden_dim = int(hidden_dim)
        self.attribute_dim = int(attribute_dim)
        self.num_layers = int(num_layers)
        self.num_attributes = int(num_attributes)
        self.top_k = int(top_k)
        self.attribute_temperature = float(attribute_temperature)
        self.eps = float(eps)

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.wavelet_patch_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        # Attribute-only, layer-specific residuals start at zero, preserving the
        # frozen text meaning at initialization while allowing depth adaptation.
        self.layer_attribute_residuals = nn.Parameter(
            torch.zeros(num_layers, num_attributes, attribute_dim)
        )
        self.raw_high_feature_scales = nn.Parameter(
            torch.full(
                (num_layers,),
                _inverse_softplus(float(initial_high_feature_scale)),
            )
        )
        self.raw_high_score_weights = nn.Parameter(
            torch.full(
                (num_layers,),
                _inverse_softplus(float(initial_high_score_weight)),
            )
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _square_grid_size(num_patches: int) -> int:
        grid_size = math.isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError(
                "Attribute-wavelet aggregation requires a square patch grid; "
                f"received {num_patches} patches"
            )
        if grid_size % 2:
            raise ValueError(
                "One-level Haar aggregation requires an even patch grid; "
                f"received {grid_size}x{grid_size}"
            )
        return grid_size

    def forward(
        self,
        patch_tokens: torch.Tensor,
        attribute_embeddings: torch.Tensor,
        semantic_projector: nn.Module,
        layer_index: int,
        return_details: bool = False,
    ):
        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens must have shape [batch, patches, hidden_dim], "
                f"received {tuple(patch_tokens.shape)}"
            )
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, received {hidden_dim}"
            )
        if attribute_embeddings.shape != (
            self.num_attributes,
            self.attribute_dim,
        ):
            raise ValueError(
                "Expected attribute_embeddings shape "
                f"{(self.num_attributes, self.attribute_dim)}, received "
                f"{tuple(attribute_embeddings.shape)}"
            )
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} outside [0, {self.num_layers})"
            )
        if self.top_k > num_patches:
            raise ValueError(
                f"top_k={self.top_k} exceeds patch count {num_patches}"
            )

        grid_size = self._square_grid_size(num_patches)
        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size, hidden_dim, grid_size, grid_size
        )
        ll, lh, hl, hh = self.dwt(patch_map)
        # High-only inverse DWT keeps the three directional bands separated
        # until exact spatial reconstruction; no signed LH+HL+HH sum is used.
        high_map = self.idwt(torch.zeros_like(ll), lh, hl, hh)
        high_tokens = high_map.flatten(2).transpose(1, 2)

        high_feature_scale = F.softplus(
            self.raw_high_feature_scales[layer_index]
        )
        wavelet_patch = patch_tokens + high_feature_scale * high_tokens
        wavelet_patch = self.wavelet_patch_norms[layer_index](wavelet_patch)

        semantic_patches = F.normalize(
            semantic_projector(wavelet_patch), p=2, dim=-1, eps=self.eps
        )
        attribute_queries = F.normalize(
            attribute_embeddings
            + self.layer_attribute_residuals[layer_index],
            p=2,
            dim=-1,
            eps=self.eps,
        )
        attribute_logits = torch.einsum(
            "ad,bnd->ban", attribute_queries, semantic_patches
        ) / self.attribute_temperature
        # Each attribute distributes its attention over all spatial positions.
        attribute_attention = torch.softmax(attribute_logits, dim=-1)
        # Attribute union: a patch remains eligible when any attribute cares.
        attribute_union = attribute_attention.amax(dim=1)  # [B, N]

        high_magnitude = high_tokens.abs()
        high_mean = high_magnitude.mean(dim=1, keepdim=True)
        high_std = high_magnitude.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        high_energy = torch.sigmoid(
            (high_magnitude - high_mean) / high_std
        )
        high_score_weight = F.softplus(
            self.raw_high_score_weights[layer_index]
        )
        selection_scores = attribute_union.unsqueeze(-1) * (
            1.0 + high_score_weight * high_energy
        )

        topk_scores, topk_indices = torch.topk(
            selection_scores, k=self.top_k, dim=1, largest=True, sorted=True
        )
        selected_values = torch.gather(
            patch_tokens, dim=1, index=topk_indices
        )
        # Positive score-normalized averaging preserves LaST's selection/value
        # separation while allowing gradients to reach the learnable selector.
        aggregation_weights = topk_scores / (
            topk_scores.sum(dim=1, keepdim=True) + self.eps
        )
        new_cls = self.output_norm(
            (aggregation_weights * selected_values).sum(dim=1)
        )

        if not return_details:
            return new_cls

        selection_mask = torch.zeros_like(selection_scores)
        selection_mask.scatter_(1, topk_indices, 1.0)
        topk_vote_map = selection_mask.mean(dim=-1)
        details = {
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "high_map": high_map,
            "high_tokens": high_tokens,
            "high_energy": high_energy,
            "high_feature_scale": high_feature_scale,
            "high_score_weight": high_score_weight,
            "wavelet_patch": wavelet_patch,
            "semantic_patches": semantic_patches,
            "attribute_queries": attribute_queries,
            "attribute_logits": attribute_logits,
            "attribute_attention": attribute_attention,
            "attribute_union": attribute_union,
            "selection_scores": selection_scores,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "aggregation_weights": aggregation_weights,
            "topk_vote_map": topk_vote_map,
        }
        return new_cls, details


class mvpcbm(RefinedMVPCBM):
    """Refined MVP-CBM with Attribute-Guided Wavelet LaST V3 CLS."""

    def __init__(self, concept_list, model_name="openclip", config=None):
        super().__init__(concept_list, model_name=model_name, config=config)

        attribute_names = tuple(self.global_attr_concepts.keys())
        if tuple(ATTRIBUTE_PROMPTS.keys()) != attribute_names:
            raise RuntimeError(
                "Attribute prompt order does not match MVP-CBM attributes: "
                f"prompts={tuple(ATTRIBUTE_PROMPTS)}, model={attribute_names}"
            )
        if model_name != "biomedclip":
            raise ValueError(
                "V3 attribute prompts are currently audited only for biomedclip"
            )
        with torch.no_grad():
            attribute_tokens = self.tokenizer(
                [ATTRIBUTE_PROMPTS[name] for name in attribute_names]
            ).cuda()
            _, attribute_embeddings, _ = self.model(None, attribute_tokens)
        self.register_buffer(
            "attribute_text_embeddings",
            attribute_embeddings.detach().clone(),
            persistent=True,
        )
        self.attribute_names = attribute_names

        num_layers = len(self.model.visual.trunk.blocks)
        hidden_dim = self.fc_confidence.in_features
        attribute_dim = self.fc_confidence.out_features
        top_k = int(getattr(config, "attribute_wavelet_top_k", DEFAULT_TOP_K))
        attribute_temperature = float(
            getattr(
                config,
                "attribute_temperature",
                DEFAULT_ATTRIBUTE_TEMPERATURE,
            )
        )
        initial_high_feature_scale = float(
            getattr(
                config,
                "initial_high_feature_scale",
                DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
            )
        )
        initial_high_score_weight = float(
            getattr(
                config,
                "initial_high_score_weight",
                DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
            )
        )
        eps = float(getattr(config, "attribute_wavelet_eps", DEFAULT_EPS))

        self.attribute_wavelet_aggregator = AttributeWaveletLaSTAggregator(
            hidden_dim=hidden_dim,
            attribute_dim=attribute_dim,
            num_layers=num_layers,
            num_attributes=len(attribute_names),
            top_k=top_k,
            attribute_temperature=attribute_temperature,
            initial_high_feature_scale=initial_high_feature_scale,
            initial_high_score_weight=initial_high_score_weight,
            eps=eps,
        )
        self.attribute_wavelet_top_k = top_k
        self.attribute_temperature = attribute_temperature
        self.initial_high_feature_scale = initial_high_feature_scale
        self.initial_high_score_weight = initial_high_score_weight
        self.attribute_wavelet_eps = eps
        self.attribute_wavelet_application = "all_vit_layers_used_by_icpm"
        self.attribute_selector_uses_concept_states = False
        self.attribute_wavelet_old_cls_used = False

    def build_attribute_wavelet_cls(
        self,
        patch_tokens: torch.Tensor,
        layer_index: int,
        return_details: bool = False,
    ):
        """Public entry point for training and later causal heat maps."""

        return self.attribute_wavelet_aggregator(
            patch_tokens=patch_tokens,
            attribute_embeddings=self.attribute_text_embeddings,
            semantic_projector=self.fc_confidence,
            layer_index=layer_index,
            return_details=return_details,
        )

    def forward(self, imgs):
        self.visual_features.clear()
        self.model(imgs, None)

        layer_features = []
        num_layers = len(self.model.visual.trunk.blocks)
        for layer_index in range(num_layers):
            block_tokens = self.visual_features[layer_index]
            patch_tokens = block_tokens[:, 1:, :]
            batch_size = patch_tokens.size(0)

            # The original block CLS is deliberately not used by V3.
            attribute_wavelet_cls = self.build_attribute_wavelet_cls(
                patch_tokens, layer_index=layer_index
            )
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

            # The original downstream patch-to-concept path remains untouched.
            projected_features = self.Avg(patch_tokens)
            image_logits_per_attribute = []
            for attribute_index, key in enumerate(self.concept_token_dict.keys()):
                concept_scores = (
                    self.logit_scale
                    * projected_features[:, attribute_index : attribute_index + 1, :]
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

