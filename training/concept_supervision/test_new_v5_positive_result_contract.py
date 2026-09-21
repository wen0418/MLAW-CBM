#!/usr/bin/env python3
"""CPU-only contract tests for the Fold04 positive-result New-V5 variant."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

import train_isic2018_baseline_cv10 as baseline
import train_isic2018_fold04_attribute_wavelet_last_new_v5_positive_result as trainer
from model.mvpcbm_attribute_wavelet_last_new_v5_positive_result import (
    AttributeWaveletLaSTNewV5,
    mvpcbm as PositiveResultNewV5,
    prepend_nothing_logits,
)


@unittest.skipUnless(
    Path(trainer.DEFAULT_PSEUDO_LABEL_CSV).is_file(),
    "private pseudo-label CSV is not present",
)
class PositiveResultDataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fold_frame, _, cls.fold_audit = baseline.load_fold_audit(
            trainer.DEFAULT_MANIFEST_DIR,
            4,
        )

    def test_positive_result_labels_cover_fold04_exactly(self):
        lookup, audit = trainer.load_pseudo_labels(
            trainer.DEFAULT_PSEUDO_LABEL_CSV,
            "positive_result",
        )
        joined, join_audit = trainer.attach_fold_labels(
            self.fold_frame,
            lookup,
        )
        self.assertEqual(audit["row_count"], 10015)
        self.assertEqual(len(joined), 10015)
        self.assertEqual(join_audit["role_counts"], {"train": 9013, "holdout": 1002})
        self.assertTrue(audit["all_rows_passed_invariants"])
        self.assertTrue(join_audit["join_verified"])
        self.assertEqual(audit["all_nothing_rows"], 23)

    def test_hard_result_uses_the_same_shifted_contract(self):
        lookup, audit = trainer.load_pseudo_labels(
            trainer.DEFAULT_PSEUDO_LABEL_CSV,
            "hard_result",
        )
        _, join_audit = trainer.attach_fold_labels(self.fold_frame, lookup)
        self.assertEqual(len(lookup), 10015)
        self.assertEqual(audit["label_column"], "hard_result")
        self.assertTrue(join_audit["join_verified"])


class NothingHeadContractTest(unittest.TestCase):
    def test_nothing_is_prepended_and_gradients_reach_both_paths(self):
        state_a = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        state_b = torch.tensor(
            [[-1.0, 0.5, 1.5], [2.0, -2.0, 1.0]],
            requires_grad=True,
        )
        nothing = torch.nn.Parameter(torch.zeros(2))
        heads = prepend_nothing_logits(
            {"a": state_a, "b": state_b},
            ("a", "b"),
            nothing,
        )
        self.assertEqual(tuple(heads["a"].shape), (2, 3))
        self.assertEqual(tuple(heads["b"].shape), (2, 4))
        self.assertTrue(torch.equal(heads["a"][:, 0], torch.zeros(2)))
        loss = F.cross_entropy(heads["a"], torch.tensor([0, 2]))
        loss = loss + F.cross_entropy(heads["b"], torch.tensor([3, 0]))
        loss.backward()
        self.assertIsNotNone(nothing.grad)
        self.assertIsNotNone(state_a.grad)
        self.assertIsNotNone(state_b.grad)
        self.assertTrue(torch.isfinite(nothing.grad).all())
        self.assertTrue(torch.isfinite(state_a.grad).all())
        self.assertTrue(torch.isfinite(state_b.grad).all())

    def test_head_order_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "head order"):
            prepend_nothing_logits(
                {"b": torch.zeros(1, 2), "a": torch.zeros(1, 2)},
                ("a", "b"),
                torch.zeros(2),
            )

    def test_forward_preserves_parent_disease_logits_verbatim(self):
        model = PositiveResultNewV5.__new__(PositiveResultNewV5)
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
            AttributeWaveletLaSTNewV5,
            "forward",
            return_value=(disease_logits, original_heads, sparse_loss),
        ) as parent_forward:
            result = model.forward(torch.randn(2, 3, 4, 4))
        parent_forward.assert_called_once()
        self.assertIs(result[0], disease_logits)
        self.assertIs(result[2], sparse_loss)
        self.assertEqual(tuple(result[1]["a"].shape), (2, 3))
        self.assertEqual(tuple(result[1]["b"].shape), (2, 4))


class Fold04RecipeContractTest(unittest.TestCase):
    def test_original_training_hyperparameters_are_preserved(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
