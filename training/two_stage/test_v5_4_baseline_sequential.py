#!/usr/bin/env python3
"""CPU contract tests for V5-4 original-label sequential training."""

from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path

import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_isic2018_baseline_cv10 as baseline  # noqa: E402
trainer = importlib.import_module(  # noqa: E402
    "training.two_stage.train_fold04_v5_4_baseline_sequential"
)


class DummySequentialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.fixed_reference = torch.nn.Parameter(
            torch.ones(1),
            requires_grad=False,
        )
        self.cls_head = torch.nn.Linear(4, 2)
        self.concept_token_dict = {"a": None, "b": None}


class BaselineConceptContractTest(unittest.TestCase):
    def test_original_fold04_labels_have_seven_valid_zero_based_states(self):
        head_sizes = [len(states) for states in baseline.CONCEPTS.values()]
        self.assertEqual(len(baseline.CONCEPT_LABEL_MAP), 7)
        for row in baseline.CONCEPT_LABEL_MAP:
            self.assertEqual(len(row), 7)
            for value, size in zip(row, head_sizes):
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, size)

    def test_raw34_concatenation_preserves_attribute_order(self):
        model = DummySequentialModel()
        model.concept_token_dict = {key: None for key in baseline.CONCEPTS}
        heads = {
            key: torch.full((2, len(states)), float(index))
            for index, (key, states) in enumerate(baseline.CONCEPTS.items())
        }
        features = trainer.concatenate_original_concept_features(model, heads)
        self.assertEqual(tuple(features.shape), (2, 34))
        start = 0
        for index, states in enumerate(baseline.CONCEPTS.values()):
            end = start + len(states)
            self.assertTrue(torch.equal(
                features[:, start:end],
                torch.full((2, len(states)), float(index)),
            ))
            start = end

    def test_stage_freezing_is_exact(self):
        model = DummySequentialModel()
        trainer.set_stage1_trainable(model)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(model.fixed_reference.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.cls_head.parameters()))
        trainer.set_stage2_trainable(model)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(model.fixed_reference.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.cls_head.parameters()))

    def test_detailed_disease_metrics_cover_overall_and_every_class(self):
        truth = np.asarray([0, 0, 1, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        prediction = np.asarray([0, 1, 1, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        metrics = trainer.detailed_disease_metrics(truth, prediction)
        self.assertEqual(
            set(metrics["per_class"]),
            set(baseline.CLASS_NAMES),
        )
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ):
            self.assertIn(key, metrics["overall"])
        for class_metrics in metrics["per_class"].values():
            for key in (
                "accuracy_ovr",
                "balanced_accuracy_ovr",
                "precision",
                "recall",
                "specificity",
                "f1",
                "support",
            ):
                self.assertIn(key, class_metrics)
        self.assertAlmostEqual(
            metrics["overall"]["balanced_accuracy"],
            metrics["overall"]["macro_recall"],
        )

    def test_runtime_format(self):
        self.assertEqual(trainer.format_runtime(3661), "01:01:01")


class BaselineSequentialRecipeContractTest(unittest.TestCase):
    def test_fold04_recipe_and_label_source_are_preserved(self):
        args = argparse.Namespace(
            attribute_wavelet_top_k=98,
            attribute_temperature=0.07,
            counterfactual_route_temperature=0.05,
            initial_high_feature_scale=0.1,
            initial_high_score_weight=1.0,
            attribute_wavelet_eps=1e-6,
        )
        recipe = trainer.sequential_recipe(args)
        differences = trainer.validate_training_recipe(recipe)
        self.assertEqual(set(differences), {"checkpoint_selection"})
        self.assertEqual(
            recipe["concept_target_source"],
            "baseline.CONCEPT_LABEL_MAP; disease-fixed zero-based labels",
        )
        self.assertFalse(recipe["positive_result_used"])
        self.assertFalse(recipe["pseudo_label_csv_used"])
        self.assertFalse(recipe["nothing_class_used"])
        self.assertEqual(recipe["stage2_classifier_input"],
                         "unchanged concatenated 34 raw MCSAF concept activations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
