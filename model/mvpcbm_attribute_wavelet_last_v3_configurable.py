"""Dataset-configurable Attribute-Wavelet LaST V3.

This module intentionally leaves the audited ISIC2018 V3 source untouched.
It reuses that implementation's aggregator and forward method, but accepts an
ordered attribute-prompt mapping from the experiment configuration instead of
depending on a dataset-specific global constant.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .mvpcbm_attribute_wavelet_last_v3 import (
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_TOP_K,
    AttributeWaveletLaSTAggregator,
    mvpcbm as AuditedAttributeWaveletLaSTV3,
)
from .mvpcbm_refined import mvpcbm as RefinedMVPCBM


def validate_attribute_prompts(
    concept_list: Mapping[str, object],
    attribute_prompts: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate exact prompt/concept key order and return ordered prompts."""

    if not isinstance(attribute_prompts, Mapping):
        raise TypeError("attribute_prompts must be an ordered mapping")
    attribute_names = tuple(concept_list.keys())
    prompt_names = tuple(attribute_prompts.keys())
    if prompt_names != attribute_names:
        raise RuntimeError(
            "Attribute prompt order must exactly match MVP-CBM concept order: "
            f"prompts={prompt_names}, concepts={attribute_names}"
        )
    prompts = tuple(attribute_prompts[name] for name in attribute_names)
    invalid = [
        name
        for name, prompt in zip(attribute_names, prompts)
        if not isinstance(prompt, str) or not prompt.strip()
    ]
    if invalid:
        raise ValueError(f"Attribute prompts must be nonempty strings: {invalid}")
    return attribute_names, prompts


class ConfigurableAttributeWaveletLaSTV3(AuditedAttributeWaveletLaSTV3):
    """V3 with externally supplied, order-checked attribute-only prompts."""

    def __init__(
        self,
        concept_list,
        *,
        attribute_prompts: Mapping[str, str],
        model_name: str = "openclip",
        config=None,
    ):
        # Bypass the audited V3 constructor because it intentionally enforces
        # the seven ISIC2018 prompts.  Its aggregator, public selector method,
        # and forward method remain inherited without modification.
        RefinedMVPCBM.__init__(
            self,
            concept_list,
            model_name=model_name,
            config=config,
        )
        attribute_names, ordered_prompts = validate_attribute_prompts(
            concept_list, attribute_prompts
        )
        model_attribute_names = tuple(self.global_attr_concepts.keys())
        if model_attribute_names != attribute_names:
            raise RuntimeError(
                "Encoded MVP-CBM attribute order changed unexpectedly: "
                f"model={model_attribute_names}, requested={attribute_names}"
            )
        if model_name != "biomedclip":
            raise ValueError(
                "Configurable V3 attribute prompts are audited only for biomedclip"
            )

        device = next(self.model.parameters()).device
        with torch.no_grad():
            attribute_tokens = self.tokenizer(list(ordered_prompts)).to(device)
            _, attribute_embeddings, _ = self.model(None, attribute_tokens)
        self.register_buffer(
            "attribute_text_embeddings",
            attribute_embeddings.detach().clone(),
            persistent=True,
        )
        self.attribute_names = attribute_names
        self.attribute_prompts = dict(attribute_prompts)

        num_layers = len(self.model.visual.trunk.blocks)
        hidden_dim = self.fc_confidence.in_features
        attribute_dim = self.fc_confidence.out_features
        top_k = int(getattr(config, "attribute_wavelet_top_k", DEFAULT_TOP_K))
        attribute_temperature = float(
            getattr(config, "attribute_temperature", DEFAULT_ATTRIBUTE_TEMPERATURE)
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
        self.attribute_prompt_source = "experiment_configuration"


mvpcbm = ConfigurableAttributeWaveletLaSTV3

