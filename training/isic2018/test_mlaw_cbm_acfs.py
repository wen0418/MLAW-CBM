"""CPU contract tests for the formal residual-bank MLAW-CBM ACFS."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from model.mlaw_cbm import MLAWCBM
from model.mlaw_cbm_res import MLAWCBM_res
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    CounterfactualBandSelectedWaveletAggregator,
    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
)


class _CountingIDWT(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.calls = 0

    def forward(self, *bands):
        self.calls += 1
        return self.wrapped(*bands)


class ResidualBankACFSTest(unittest.TestCase):
    hidden_dim = 8
    attribute_dim = 6
    num_layers = 2
    num_patches = 16

    def _aggregator(self, cls, num_attributes: int):
        return cls(
            hidden_dim=self.hidden_dim,
            attribute_dim=self.attribute_dim,
            num_layers=self.num_layers,
            num_attributes=num_attributes,
            top_k=8,
            attribute_temperature=0.07,
            route_temperature=0.05,
            initial_high_feature_scale=0.1,
            initial_high_score_weight=1.0,
            eps=1.0e-6,
        )

    def test_public_model_routes_formal_and_preserved_implementations(self):
        self.assertIs(
            MLAWCBM.attribute_wavelet_aggregator_class,
            ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
        )
        self.assertIs(
            MLAWCBM_res.attribute_wavelet_aggregator_class,
            CounterfactualBandSelectedWaveletAggregator,
        )

    def test_equivalence_and_dynamic_attribute_count(self):
        for num_attributes in (3, 7):
            with self.subTest(num_attributes=num_attributes):
                torch.manual_seed(43)
                legacy = self._aggregator(
                    CounterfactualBandSelectedWaveletAggregator,
                    num_attributes,
                )
                formal = self._aggregator(
                    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
                    num_attributes,
                )
                formal.load_state_dict(legacy.state_dict())
                projector = nn.Linear(self.hidden_dim, self.attribute_dim)
                patch_tokens = torch.randn(2, self.num_patches, self.hidden_dim)
                attribute_embeddings = torch.randn(
                    num_attributes, self.attribute_dim
                )

                legacy_cls, legacy_details = legacy(
                    patch_tokens,
                    attribute_embeddings,
                    projector,
                    layer_index=1,
                    return_details=True,
                )
                formal_cls, formal_details = formal(
                    patch_tokens,
                    attribute_embeddings,
                    projector,
                    layer_index=1,
                    return_details=True,
                )

                self.assertEqual(
                    formal_details["counterfactual_delta"].shape,
                    (2, num_attributes, 3, self.num_patches),
                )
                self.assertEqual(
                    formal_details["route_weights"].shape,
                    (2, num_attributes, 4, self.num_patches),
                )
                self.assertEqual(
                    formal_details["residual_bank"].shape,
                    (2, 3, self.num_patches, self.hidden_dim),
                )
                self.assertTrue(
                    torch.equal(
                        formal_details["counterfactual_tokens"],
                        patch_tokens.unsqueeze(1)
                        - formal_details["residual_bank"],
                    )
                )

                comparisons = (
                    (
                        legacy_details["ablated_tokens"],
                        formal_details["counterfactual_tokens"],
                    ),
                    (
                        legacy_details["counterfactual_delta"],
                        formal_details["counterfactual_delta"],
                    ),
                    (legacy_details["band_gates"], formal_details["band_gate"]),
                    (
                        legacy_details["filtered_high_tokens"],
                        formal_details["filtered_high_tokens"],
                    ),
                    (
                        legacy_details["x_tilde_pre_norm"],
                        formal_details["x_tilde_pre_norm"],
                    ),
                    (legacy_cls, formal_cls),
                )
                for expected, actual in comparisons:
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=1.0e-5,
                        rtol=1.0e-4,
                    )

    def test_formal_path_uses_one_idwt_and_preserves_autograd(self):
        torch.manual_seed(7)
        num_attributes = 5
        formal = self._aggregator(
            ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
            num_attributes,
        )
        counting_idwt = _CountingIDWT(formal.idwt)
        formal.idwt = counting_idwt
        projector = nn.Linear(self.hidden_dim, self.attribute_dim)
        patch_tokens = torch.randn(
            2,
            self.num_patches,
            self.hidden_dim,
            requires_grad=True,
        )
        attribute_embeddings = torch.randn(
            num_attributes, self.attribute_dim
        )

        new_cls, details = formal(
            patch_tokens,
            attribute_embeddings,
            projector,
            layer_index=0,
            return_details=True,
        )
        self.assertEqual(counting_idwt.calls, 1)

        loss = new_cls.square().mean() + details["x_tilde_pre_norm"].square().mean()
        loss.backward()
        gradients = (
            patch_tokens.grad,
            projector.weight.grad,
            formal.layer_attribute_residuals.grad,
            formal.raw_high_feature_scales.grad,
            formal.raw_high_score_weights.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
