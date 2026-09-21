"""Counterfactual band-selected Attribute-Wavelet LaST (new V5).

This variant is a direct successor to V3.  Counterfactual reasoning is moved
upstream and used only to construct a cleaner high-frequency residual.  The
rest of the selector is V3's original semantic/high-frequency score:

    S[b, n, c] = U[b, n] * (1 + softplus(w[l]) * E[b, n, c]).

For each ViT layer, the module:

1. applies a fixed one-level Haar DWT to X [B, N, D];
2. reconstructs three counterfactual patches by zeroing LH, HL, or HH directly
   in IDWT (rather than subtracting a residual from X);
3. computes Delta = cosine(base) - cosine(ablated) for every attribute, band,
   and patch, then Sparsemax-routes over {None, LH, HL, HH};
4. lets only positive-Delta band routes form attribute-weighted spatial gates;
5. gates exact LH/HL/HH-only IDWT residuals and sums them into H* [B, N, D];
6. builds W* = LN(X + softplus(s[l]) H*); and
7. runs V3's U/E/score, channel-wise Top-K, and original-X aggregation.

None is therefore a real no-enhancement option.  A negative Delta excludes a
band from H*; unlike the old V5, it does not directly penalize the final score.
The original downstream MVP-CBM patch-to-concept path remains unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .mvpcbm_attribute_wavelet_last_v3 import (
    ATTRIBUTE_PROMPTS,
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_TOP_K,
    HaarDWT2D,
    HaarIDWT2D,
    _inverse_softplus,
    mvpcbm as AttributeWaveletLaSTV3,
)


DEFAULT_ROUTE_TEMPERATURE = 0.05
NUM_DETAIL_BANDS = 3
ROUTE_NAMES = ("none", "lh", "hl", "hh")


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits onto the probability simplex with sparse support."""

    if logits.numel() == 0:
        return logits
    dim = dim if dim >= 0 else logits.ndim + dim
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    sorted_logits, _ = torch.sort(shifted, dim=dim, descending=True)
    support_size = logits.size(dim)
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = support_size
    ranks = torch.arange(
        1,
        support_size + 1,
        device=logits.device,
        dtype=logits.dtype,
    ).view(ranks_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1 + ranks * sorted_logits > cumulative
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(dim, k - 1) - 1) / k.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class CounterfactualBandSelectedWaveletAggregator(nn.Module):
    """Build a V3 CLS from counterfactually selected wavelet detail bands."""

    def __init__(
        self,
        hidden_dim: int,
        attribute_dim: int,
        num_layers: int,
        num_attributes: int,
        top_k: int = DEFAULT_TOP_K,
        attribute_temperature: float = DEFAULT_ATTRIBUTE_TEMPERATURE,
        route_temperature: float = DEFAULT_ROUTE_TEMPERATURE,
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
        for name, value in (
            ("attribute_temperature", attribute_temperature),
            ("route_temperature", route_temperature),
            ("initial_high_feature_scale", initial_high_feature_scale),
            ("initial_high_score_weight", initial_high_score_weight),
            ("eps", eps),
        ):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")

        self.hidden_dim = int(hidden_dim)
        self.attribute_dim = int(attribute_dim)
        self.num_layers = int(num_layers)
        self.num_attributes = int(num_attributes)
        self.top_k = int(top_k)
        self.attribute_temperature = float(attribute_temperature)
        self.route_temperature = float(route_temperature)
        self.eps = float(eps)

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.wavelet_patch_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
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
                "Counterfactual wavelet aggregation requires a square patch "
                f"grid; received {num_patches} patches"
            )
        if grid_size % 2:
            raise ValueError(
                "One-level Haar aggregation requires an even patch grid; "
                f"received {grid_size}x{grid_size}"
            )
        return grid_size

    @staticmethod
    def _tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    def _direct_ablation_tokens(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Return direct-IDWT remove-LH/HL/HH patches as [B, 3, N, D]."""

        zero = torch.zeros_like(ll)
        ablated_maps = (
            self.idwt(ll, zero, hl, hh),
            self.idwt(ll, lh, zero, hh),
            self.idwt(ll, lh, hl, zero),
        )
        return torch.stack([self._tokens(item) for item in ablated_maps], dim=1)

    def _detail_residual_tokens(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact LH/HL/HH-only residuals as [B, 3, N, D]."""

        zero = torch.zeros_like(ll)
        detail_maps = (
            self.idwt(zero, lh, zero, zero),
            self.idwt(zero, zero, hl, zero),
            self.idwt(zero, zero, zero, hh),
        )
        return torch.stack([self._tokens(item) for item in detail_maps], dim=1)

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
        ablated_tokens = self._direct_ablation_tokens(ll, lh, hl, hh)
        detail_tokens = self._detail_residual_tokens(ll, lh, hl, hh)

        attribute_queries = F.normalize(
            attribute_embeddings
            + self.layer_attribute_residuals[layer_index],
            p=2,
            dim=-1,
            eps=self.eps,
        )

        # Raw-X attention P0 is used only to combine attribute routing opinions.
        base_semantic = F.normalize(
            semantic_projector(patch_tokens), p=2, dim=-1, eps=self.eps
        )
        base_similarity = torch.einsum(
            "ad,bnd->ban", attribute_queries, base_semantic
        )
        base_attribute_attention = torch.softmax(
            base_similarity / self.attribute_temperature, dim=-1
        )

        ablated_semantic = F.normalize(
            semantic_projector(ablated_tokens), p=2, dim=-1, eps=self.eps
        )
        ablated_similarity = torch.einsum(
            "ad,brnd->barn", attribute_queries, ablated_semantic
        )
        counterfactual_delta = (
            base_similarity.unsqueeze(2) - ablated_similarity
        )  # [B, A, 3, N]

        none_logits = torch.zeros_like(counterfactual_delta[:, :, :1, :])
        route_logits = torch.cat(
            (none_logits, counterfactual_delta / self.route_temperature),
            dim=2,
        )
        route_weights = sparsemax(route_logits, dim=2)  # [B, A, 4, N]
        band_route_weights = route_weights[:, :, 1:, :]

        # None consumes routing mass but produces no residual.  A band with
        # Delta <= 0 is explicitly excluded, even if Sparsemax gives it mass.
        positive_band_routes = band_route_weights * (
            counterfactual_delta > 0
        ).to(band_route_weights.dtype)
        attribute_mass = base_attribute_attention.sum(dim=1).clamp_min(self.eps)
        band_gates = (
            base_attribute_attention.unsqueeze(2) * positive_band_routes
        ).sum(dim=1) / attribute_mass.unsqueeze(1)  # [B, 3, N]

        filtered_high_tokens = (
            detail_tokens * band_gates.unsqueeze(-1)
        ).sum(dim=1)  # [B, N, D]
        high_feature_scale = F.softplus(
            self.raw_high_feature_scales[layer_index]
        )
        wavelet_patch = self.wavelet_patch_norms[layer_index](
            patch_tokens + high_feature_scale * filtered_high_tokens
        )

        # From this point onward, the computation is deliberately V3.
        semantic_patches = F.normalize(
            semantic_projector(wavelet_patch), p=2, dim=-1, eps=self.eps
        )
        attribute_logits = torch.einsum(
            "ad,bnd->ban", attribute_queries, semantic_patches
        ) / self.attribute_temperature
        attribute_attention = torch.softmax(attribute_logits, dim=-1)
        attribute_union = attribute_attention.amax(dim=1)  # U: [B, N]

        high_magnitude = filtered_high_tokens.abs()
        high_mean = high_magnitude.mean(dim=1, keepdim=True)
        high_std = high_magnitude.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        high_energy = torch.sigmoid(
            (high_magnitude - high_mean) / high_std
        )  # E: [B, N, D]
        high_score_weight = F.softplus(
            self.raw_high_score_weights[layer_index]
        )
        selection_scores = attribute_union.unsqueeze(-1) * (
            1.0 + high_score_weight * high_energy
        )  # S: [B, N, D]

        topk_scores, topk_indices = torch.topk(
            selection_scores,
            k=self.top_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        selected_values = torch.gather(patch_tokens, dim=1, index=topk_indices)
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
        details = {
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "ablated_tokens": ablated_tokens,
            "detail_tokens": detail_tokens,
            "attribute_queries": attribute_queries,
            "base_semantic": base_semantic,
            "base_similarity": base_similarity,
            "base_attribute_attention": base_attribute_attention,
            "ablated_similarity": ablated_similarity,
            "counterfactual_delta": counterfactual_delta,
            "route_logits": route_logits,
            "route_weights": route_weights,
            "route_none_weight": route_weights[:, :, 0, :],
            "band_route_weights": band_route_weights,
            "positive_band_routes": positive_band_routes,
            "band_gates": band_gates,
            "filtered_high_tokens": filtered_high_tokens,
            "high_feature_scale": high_feature_scale,
            "wavelet_patch": wavelet_patch,
            "semantic_patches": semantic_patches,
            "attribute_logits": attribute_logits,
            "attribute_attention": attribute_attention,
            "attribute_union": attribute_union,
            "high_energy": high_energy,
            "high_score_weight": high_score_weight,
            "selection_scores": selection_scores,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "aggregation_weights": aggregation_weights,
            "topk_vote_map": selection_mask.mean(dim=-1),
        }
        return new_cls, details


class mvpcbm(AttributeWaveletLaSTV3):
    """Refined MVP-CBM with counterfactually selected V3 wavelet enhancement."""

    def __init__(self, concept_list, model_name="openclip", config=None):
        super().__init__(concept_list, model_name=model_name, config=config)

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

