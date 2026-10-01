#!/usr/bin/env python3
"""CPU contract tests for scalp MLAW-CBM per-image VLM supervision."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.mlaw_cbm_configurable import ConfigurableMLAWCBM
from model.mlaw_cbm_scalp_positive_result import ScalpPositiveResultMLAWCBM
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
)
from training.scalp.config import CONCEPTS, FIXED_RECIPE
from training.scalp_positive_result import train_mlaw_cbm as trainer
from training.scalp_positive_result.dataset import build_joined_manifests


DATA_ROOT = Path("/home/wen/Desktop/CBM/scalp_dataset/New_scalp")
LABEL_ROOT = Path(
    "/home/wen/Desktop/CBM/filtered data (VLM check concept label)/scalp_concept"
)


@unittest.skipUnless(
    DATA_ROOT.is_dir() and LABEL_ROOT.is_dir(),
    "private scalp images/VLM labels are not present",
)
class ScalpPositiveResultDataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests, cls.audit = build_joined_manifests(
            DATA_ROOT, LABEL_ROOT, "inclusive_output_list"
        )

    def test_split_and_class_counts_match_exactly(self):
        self.assertEqual(len(self.manifests["train"]), 7645)
        self.assertEqual(len(self.manifests["test"]), 3568)
        self.assertTrue(
            self.audit["count_class_id_and_label_invariants_verified"]
        )
        self.assertEqual(
            self.audit["mapping_sha256"],
            "1e2cba79368ea63b6f0dda386d5cb1e94d3e4b9b8389d713c63f3a8cc088f4be",
        )

    def test_known_sorted_filename_pairs_reconstruct_vlm_ids(self):
        train_by_id = {
            row["image_id"]: row for row in self.manifests["train"]
        }
        self.assertEqual(
            train_by_id["SCALP_TRAIN_DRY_DANDRUFF_00001"]["relative_path"],
            "train/Dry-Dandruff/2014.09.16丁先生-原生髮區(清潔後).jpg",
        )
        self.assertEqual(
            train_by_id["SCALP_TRAIN_FOLLICULITIS_00248"]["relative_path"],
            "train/Folliculitis/SC_20190514_205726_278.jpg",
        )

    def test_inclusive_targets_have_nothing_plus_shifted_encoding(self):
        head_sizes = [len(states) + 1 for states in CONCEPTS.values()]
        for split in ("train", "test"):
            for row in self.manifests[split]:
                for index, target in enumerate(row["concept_target"]):
                    self.assertIn(target, (0, row["shifted"][index]))
                    self.assertGreaterEqual(target, 0)
                    self.assertLess(target, head_sizes[index])


class ScalpPositiveResultModelContractTest(unittest.TestCase):
    def test_forward_preserves_mlaw_disease_logits_verbatim(self):
        model = ScalpPositiveResultMLAWCBM.__new__(
            ScalpPositiveResultMLAWCBM
        )
        torch.nn.Module.__init__(model)
        model.positive_result_attribute_order = ("a", "b")
        model.nothing_logits = torch.nn.Parameter(torch.zeros(2))
        disease_logits = torch.randn(2, 6)
        original_heads = {"a": torch.randn(2, 2), "b": torch.randn(2, 3)}
        sparse_loss = torch.tensor(0.25)
        with mock.patch.object(
            ConfigurableMLAWCBM,
            "forward",
            return_value=(disease_logits, original_heads, sparse_loss),
        ) as parent_forward:
            result = model.forward(torch.randn(2, 3, 4, 4))
        parent_forward.assert_called_once()
        self.assertIs(result[0], disease_logits)
        self.assertIs(result[2], sparse_loss)
        self.assertEqual(tuple(result[1]["a"].shape), (2, 3))
        self.assertEqual(tuple(result[1]["b"].shape), (2, 4))

    def test_scalp_head_sizes_and_disease_state_count(self):
        self.assertEqual(
            [len(states) + 1 for states in CONCEPTS.values()],
            [5, 5, 5, 5, 4, 4],
        )
        self.assertEqual(sum(len(states) for states in CONCEPTS.values()), 22)
        self.assertTrue(
            issubclass(ScalpPositiveResultMLAWCBM, ConfigurableMLAWCBM)
        )

    def test_wrapper_uses_formal_residual_bank_acfs(self):
        self.assertIs(
            ConfigurableMLAWCBM.attribute_wavelet_aggregator_class,
            ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
        )


class ScalpPositiveResultRecipeContractTest(unittest.TestCase):
    def test_joint_recipe_preserves_controlled_scalp_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                concept_label_column="inclusive_output_list",
                pseudo_label_dir=temp_dir,
            )
            recipe = trainer.resolved_recipe(args)
        for key, value in FIXED_RECIPE.items():
            self.assertEqual(recipe[key], value)
        self.assertEqual(recipe["training_mode"], "joint_from_scratch_mvp_cbm")
        self.assertTrue(recipe["dynamic_concepts_used"])
        self.assertEqual(
            recipe["base_model"], "formal_mlaw_cbm_residual_bank_acfs"
        )
        self.assertEqual(recipe["counterfactual_construction"], "X_minus_R_band")
        self.assertFalse(recipe["nothing_logits_used_by_disease_classifier"])
        self.assertEqual(recipe["original_concept_state_count"], 22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
