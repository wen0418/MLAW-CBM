#!/usr/bin/env python3
"""CPU contract tests for the Fold04 positive-result V5-4 variant."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_isic2018_baseline_cv10 as baseline  # noqa: E402
from model.mvpcbm_attribute_wavelet_last_v5_4 import (  # noqa: E402
    mvpcbm as AttributeWaveletLaSTV5_4,
)
from model.mvpcbm_attribute_wavelet_last_v5_4_positive_result import (  # noqa: E402
    mvpcbm as PositiveResultV5_4,
)
from training.concept_supervision import (  # noqa: E402
    train_fold04_v5_4_positive_result as trainer,
)


@unittest.skipUnless(
    Path(trainer.DEFAULT_PSEUDO_LABEL_CSV).is_file(),
    "private pseudo-label CSV is not present",
)
class PositiveResultV5_4DataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fold_frame, _, cls.fold_audit = baseline.load_fold_audit(
            trainer.DEFAULT_MANIFEST_DIR,
            4,
        )

    def test_positive_result_labels_cover_fold04_exactly(self):
        lookup, audit = trainer.positive_common.load_pseudo_labels(
            trainer.DEFAULT_PSEUDO_LABEL_CSV,
            "positive_result",
        )
        joined, join_audit = trainer.positive_common.attach_fold_labels(
            self.fold_frame,
            lookup,
        )
        self.assertEqual(audit["row_count"], 10015)
        self.assertEqual(len(joined), 10015)
        self.assertEqual(
            join_audit["role_counts"],
            {"train": 9013, "holdout": 1002},
        )
        self.assertTrue(audit["all_rows_passed_invariants"])
        self.assertTrue(join_audit["join_verified"])


class PositiveResultV5_4HeadContractTest(unittest.TestCase):
    def test_forward_preserves_v5_4_disease_logits_verbatim(self):
        model = PositiveResultV5_4.__new__(PositiveResultV5_4)
        torch.nn.Module.__init__(model)
        model.positive_result_attribute_order = ("a", "b")
        model.nothing_logits = torch.nn.Parameter(torch.zeros(2))
        disease_logits = torch.randn(2, 7)
        original_heads = {
            "a": torch.randn(2, 2),
            "b": torch.randn(2, 3),
        }
        sparse_loss = torch.tensor(0.25)
        with mock.patch.object(
            AttributeWaveletLaSTV5_4,
            "forward",
            return_value=(disease_logits, original_heads, sparse_loss),
        ) as parent_forward:
            result = model.forward(torch.randn(2, 3, 4, 4))
        parent_forward.assert_called_once()
        self.assertIs(result[0], disease_logits)
        self.assertIs(result[2], sparse_loss)
        self.assertEqual(tuple(result[1]["a"].shape), (2, 3))
        self.assertEqual(tuple(result[1]["b"].shape), (2, 4))

    def test_v5_4_pre_layernorm_concept_path_is_inherited(self):
        self.assertTrue(
            issubclass(PositiveResultV5_4, AttributeWaveletLaSTV5_4)
        )
        self.assertEqual(
            PositiveResultV5_4.concept_pooling_input,
            "X_plus_scaled_counterfactual_H_star_before_layernorm",
        )
        self.assertFalse(
            PositiveResultV5_4.concept_pooling_uses_attribute_attention
        )
        self.assertFalse(PositiveResultV5_4.concept_pooling_uses_selector_topk)


class PositiveResultV5_4RecipeContractTest(unittest.TestCase):
    def test_original_fold04_hyperparameters_are_preserved(self):
        args = argparse.Namespace(
            pseudo_label_csv=str(trainer.DEFAULT_PSEUDO_LABEL_CSV),
            concept_label_column="positive_result",
            attribute_wavelet_top_k=98,
            attribute_temperature=0.07,
            counterfactual_route_temperature=0.05,
            initial_high_feature_scale=0.1,
            initial_high_score_weight=1.0,
            attribute_wavelet_eps=1e-6,
        )
        recipe = trainer.positive_result_recipe(args)
        differences = trainer.validate_training_recipe(recipe)
        self.assertEqual(set(differences), {"checkpoint_selection"})
        for key in trainer.common_trainer.TRAINING_RECIPE_KEYS:
            if key == "checkpoint_selection":
                continue
            self.assertEqual(recipe[key], baseline.FIXED_RECIPE[key])
        self.assertEqual(recipe["training_mode"], "joint_from_scratch_mvp_cbm")
        self.assertFalse(recipe["nothing_logits_used_by_disease_classifier"])
        self.assertEqual(recipe["original_concept_state_count"], 34)
        self.assertEqual(recipe["two_by_two_ablation_cell"], "C")
        self.assertFalse(recipe["concept_pooling_layernorm_applied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
