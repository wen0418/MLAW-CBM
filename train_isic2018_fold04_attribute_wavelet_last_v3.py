#!/usr/bin/env python3
"""Train Attribute-Guided Wavelet LaST V3 on audited ISIC2018 Fold04.

This is a controlled architecture experiment.  It imports the complete Fold04
data, augmentation, optimizer, schedule, class weighting, concept objective,
seed, and BMAC checkpoint policy from ``train_isic2018_baseline_cv10.py``.
V3 changes only the per-layer CLS source used by the attribute-preference path;
the downstream concept-state scoring path remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_isic2018_baseline_cv10 as baseline
from training.common import isic2018 as common_trainer
from model.mvpcbm_attribute_wavelet_last_v3 import (
    ATTRIBUTE_PROMPTS,
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_TOP_K,
    mvpcbm as AttributeWaveletLaSTMVPCBM,
)


PROTOCOL = "baseline_cv10_attribute_wavelet_last_v3_shared_val_test_diagnostic"
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v3.py"
PROJECT_ROOT = Path(__file__).resolve().parent
ORIGIN_ROOT = PROJECT_ROOT
DEFAULT_MANIFEST_DIR = (
    ORIGIN_ROOT / "splits" / "isic2018_baseline_cv10_imagelevel"
)
DEFAULT_REFERENCE = (
    ORIGIN_ROOT / "reference_results" / "isic2018_fold04_baseline_lam2p5.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(10), default=4)
    parser.add_argument("--data-path", default="./dataset/ISIC2018")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tensorboard-dir",
        default="./log/isic2018/baseline_cv10_attribute_wavelet_last_v3",
    )
    parser.add_argument(
        "--baseline-reference-metrics", default=str(DEFAULT_REFERENCE)
    )
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument(
        "--attribute-wavelet-top-k", type=int, default=DEFAULT_TOP_K
    )
    parser.add_argument(
        "--attribute-temperature",
        type=float,
        default=DEFAULT_ATTRIBUTE_TEMPERATURE,
    )
    parser.add_argument(
        "--initial-high-feature-scale",
        type=float,
        default=DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    )
    parser.add_argument(
        "--initial-high-score-weight",
        type=float,
        default=DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    )
    parser.add_argument(
        "--attribute-wavelet-eps", type=float, default=DEFAULT_EPS
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-config-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.attribute_wavelet_top_k <= 196:
        parser.error("--attribute-wavelet-top-k must be within [1, 196]")
    for name in (
        "attribute_temperature",
        "initial_high_feature_scale",
        "initial_high_score_weight",
        "attribute_wavelet_eps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def model_source_path() -> Path:
    imported = Path(inspect.getfile(AttributeWaveletLaSTMVPCBM)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected V3 model import: {imported}; expected {expected}"
        )
    return imported


def v3_recipe(args) -> dict:
    recipe = copy.deepcopy(baseline.FIXED_RECIPE)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v3",
            "architecture_ablation": (
                "replace every ICPM CLS input with seven attribute-only text "
                "queries over high-only inverse-Haar enhanced patches, followed "
                "by attribute-union channel-wise Top-K selective aggregation"
            ),
            "base_model_source": "model/mvpcbm_refined.py",
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "concept_target_source": "diagnosis-derived prototype labels",
            "dynamic_concepts_used": False,
            "old_cls_used_for_selection": False,
            "selector_uses_concept_states": False,
            "selector_attribute_names": list(ATTRIBUTE_PROMPTS.keys()),
            "selector_attribute_prompts": dict(ATTRIBUTE_PROMPTS),
            "selector_attribute_prompt_count": len(ATTRIBUTE_PROMPTS),
            "selector_semantic_projection": (
                "shared MVP-CBM fc_confidence 768-to-512 projection"
            ),
            "selector_attribute_interaction": (
                "cosine similarity at temperature, spatial softmax per attribute, "
                "then max union across seven attributes"
            ),
            "wavelet_transform": "fixed one-level orthonormal 2-D Haar",
            "wavelet_high_reconstruction": (
                "exact inverse Haar of zero-LL plus separate signed LH/HL/HH"
            ),
            "wavelet_signed_high_sum_used": False,
            "attribute_wavelet_top_k": int(args.attribute_wavelet_top_k),
            "attribute_temperature": float(args.attribute_temperature),
            "initial_high_feature_scale": float(
                args.initial_high_feature_scale
            ),
            "initial_high_score_weight": float(
                args.initial_high_score_weight
            ),
            "attribute_wavelet_eps": float(args.attribute_wavelet_eps),
            "attribute_wavelet_patch_count": 196,
            "attribute_wavelet_hidden_dim": 768,
            "attribute_wavelet_attribute_dim": 512,
            "attribute_wavelet_selection_score": (
                "attribute_union * (1 + positive_layer_weight * "
                "channelwise_normalized_high_energy)"
            ),
            "attribute_wavelet_topk_axis": (
                "196 patch positions independently for each of 768 channels"
            ),
            "attribute_wavelet_value_source": "original ViT patch values",
            "attribute_wavelet_aggregation": (
                "positive selected-score-normalized weighted mean over Top-K"
            ),
            "attribute_wavelet_application": "all ViT layers consumed by ICPM",
            "downstream_patch_to_concept_path_changed": False,
            "additional_auxiliary_loss": None,
        }
    )
    return recipe


def validate_training_recipe_unchanged(recipe: dict) -> None:
    changed = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "v3": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if changed:
        raise RuntimeError(f"V3 changed the audited Fold04 recipe: {changed}")


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("V3 training requires a CUDA-capable PyTorch setup")
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "V3 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = AttributeWaveletLaSTMVPCBM(
        concept_list=baseline.CONCEPTS,
        model_name="biomedclip",
        config=args,
    )
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=len(baseline.CLASS_NAMES),
    )
    vit.head = nn.Identity()
    model.model.visual.trunk.load_state_dict(vit.state_dict())
    model.cuda()
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    return model


def run_smoke_test(model, args, fold_frame) -> None:
    train_loader, _, _ = baseline.make_dataloaders(model, args, fold_frame)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(baseline.CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    data, labels, concept_targets = next(iter(train_loader))
    data = data.float().cuda(non_blocking=True)
    labels = labels.long().cuda(non_blocking=True)
    concept_targets = concept_targets.long().cuda(non_blocking=True)

    logits, concept_logits, sparse_loss = model(data)
    concept_loss_sum = torch.zeros((), device=data.device)
    for concept_index, key in enumerate(model.concept_token_dict.keys()):
        concept_loss_sum = concept_loss_sum + F.cross_entropy(
            concept_logits[key], concept_targets[:, concept_index]
        )
    concept_loss = concept_loss_sum / len(model.concept_token_dict)
    classification_loss = criterion(logits, labels)
    total_loss = (
        classification_loss
        + baseline.FIXED_RECIPE["lambda_cpt"] * concept_loss
        + sparse_loss
    )
    if not torch.isfinite(total_loss):
        raise FloatingPointError("V3 smoke-test loss is non-finite")
    total_loss.backward()

    selector = model.attribute_wavelet_aggregator
    required_gradients = {
        "layer_attribute_residuals": selector.layer_attribute_residuals.grad,
        "raw_high_feature_scales": selector.raw_high_feature_scales.grad,
        "raw_high_score_weights": selector.raw_high_score_weights.grad,
    }
    missing = [name for name, gradient in required_gradients.items() if gradient is None]
    nonfinite = [
        name
        for name, gradient in required_gradients.items()
        if gradient is not None and not torch.isfinite(gradient).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            f"V3 selector gradient failure: missing={missing}, nonfinite={nonfinite}"
        )

    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": "refined_mvpcbm_attribute_wavelet_last_v3",
                "batch_size": len(labels),
                "classification_loss": classification_loss.item(),
                "concept_loss_mean": concept_loss.item(),
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": sparse_loss.item(),
                "total_loss": total_loss.item(),
                "attribute_count": len(ATTRIBUTE_PROMPTS),
                "attribute_wavelet_top_k": args.attribute_wavelet_top_k,
                "selector_uses_concept_states": False,
                "old_cls_used_for_selection": False,
                "selector_gradients_verified": True,
                "output_directory_created": False,
            },
            indent=2,
        )
    )


def rewrite_completed_metrics(
    output_dir: Path,
    args,
    recipe: dict,
    reference: dict,
) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    base_trainer_source = metrics["trainer_source"]
    base_trainer_sha256 = metrics["trainer_sha256"]
    trainer_path = Path(__file__).resolve()
    variant_source = model_source_path()
    base_source = baseline.model_source_path()
    reference_path = Path(args.baseline_reference_metrics).resolve()

    metrics.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v3",
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "base_model_source": str(base_source),
            "base_model_source_sha256": baseline.sha256_file(base_source),
            "trainer_source": str(trainer_path),
            "trainer_sha256": baseline.sha256_file(trainer_path),
            "base_trainer_source": base_trainer_source,
            "base_trainer_sha256": base_trainer_sha256,
            "attribute_wavelet_last_v3": {
                "attribute_names": recipe["selector_attribute_names"],
                "attribute_prompts": recipe["selector_attribute_prompts"],
                "selector_uses_concept_states": False,
                "semantic_projection": recipe[
                    "selector_semantic_projection"
                ],
                "attribute_interaction": recipe[
                    "selector_attribute_interaction"
                ],
                "wavelet_transform": recipe["wavelet_transform"],
                "high_reconstruction": recipe[
                    "wavelet_high_reconstruction"
                ],
                "signed_high_sum_used": False,
                "top_k": recipe["attribute_wavelet_top_k"],
                "attribute_temperature": recipe["attribute_temperature"],
                "initial_high_feature_scale": recipe[
                    "initial_high_feature_scale"
                ],
                "initial_high_score_weight": recipe[
                    "initial_high_score_weight"
                ],
                "selection_score": recipe[
                    "attribute_wavelet_selection_score"
                ],
                "topk_axis": recipe["attribute_wavelet_topk_axis"],
                "value_source": recipe["attribute_wavelet_value_source"],
                "aggregation": recipe["attribute_wavelet_aggregation"],
                "application": recipe["attribute_wavelet_application"],
                "old_cls_used_for_selection": False,
                "downstream_patch_to_concept_path_changed": False,
            },
            "concept_target_source": "diagnosis-derived prototype labels",
            "dynamic_concepts_used": False,
            "baseline_fold_reference_metrics": str(reference_path),
            "baseline_fold_reference_metrics_sha256": baseline.sha256_file(
                reference_path
            ),
            "baseline_fold_reference": {
                key: reference[key]
                for key in (
                    "acc",
                    "bmac",
                    "macro_f1",
                    "weighted_f1",
                    "best_epoch",
                )
            },
            "delta_vs_original_mvpcbm_fold": {
                key: metrics[key] - reference[key]
                for key in ("acc", "bmac", "macro_f1", "weighted_f1")
            },
        }
    )
    metrics["runtime_config"].update(
        {
            "attribute_wavelet_top_k": recipe["attribute_wavelet_top_k"],
            "attribute_temperature": recipe["attribute_temperature"],
            "initial_high_feature_scale": recipe[
                "initial_high_feature_scale"
            ],
            "initial_high_score_weight": recipe[
                "initial_high_score_weight"
            ],
            "attribute_wavelet_eps": recipe["attribute_wavelet_eps"],
            "selector_uses_concept_states": False,
            "old_cls_used_for_selection": False,
        }
    )
    for key in (
        "historical_reference_metrics",
        "historical_reference_metrics_sha256",
        "historical_reference_result",
    ):
        metrics.pop(key, None)

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }
    for name in baseline.CLASS_NAMES:
        flat[f"f1_{name}"] = metrics["per_class_f1"][name]
        flat[f"recall_{name}"] = metrics["per_class_recall"][name]
    baseline.write_csv(output_dir / "metrics.csv", [flat])


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = model_source_path()
    source_hash = baseline.sha256_file(source)
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "V3 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(), args.fold, fold_audit
    )
    recipe = v3_recipe(args)
    validate_training_recipe_unchanged(recipe)
    output_dir = Path(args.output_dir).resolve()
    if not (args.validate_config_only or args.smoke_test) and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    static_audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "fold": args.fold,
        "fold_interpretation": (
            f"one-based split {args.fold + 1} means zero-based fold{args.fold:02d}"
        ),
        "variant_model_source": str(source),
        "variant_model_source_sha256": source_hash,
        "base_model_source": str(baseline.model_source_path()),
        "base_model_source_sha256": baseline.sha256_file(
            baseline.model_source_path()
        ),
        "base_trainer_source": str(Path(baseline.__file__).resolve()),
        "base_trainer_sha256": baseline.sha256_file(baseline.__file__),
        "v3_trainer_source": str(Path(__file__).resolve()),
        "v3_trainer_sha256": baseline.sha256_file(__file__),
        "recipe": recipe,
        "training_recipe_difference_vs_origin": {},
        "architecture_difference_vs_origin": recipe["architecture_ablation"],
        "selector_uses_concept_states": False,
        "downstream_patch_to_concept_path_changed": False,
        "manifest_sha256": fold_audit["manifest_sha256"],
        "train_size": fold_audit["train_size"],
        "holdout_size": fold_audit["test_size"],
        "val_test_overlap": fold_audit["val_test_overlap"],
        "baseline_reference": {
            "path": str(Path(args.baseline_reference_metrics).resolve()),
            "sha256": baseline.sha256_file(args.baseline_reference_metrics),
            **{
                key: reference[key]
                for key in (
                    "acc",
                    "bmac",
                    "macro_f1",
                    "weighted_f1",
                    "best_epoch",
                )
            },
        },
        "planned_output": str(output_dir),
        "planned_output_exists": output_dir.exists(),
    }
    print(json.dumps(static_audit, indent=2, ensure_ascii=False))
    if args.validate_config_only:
        print("Configuration validation completed; no training/output created.")
        return

    baseline.FIXED_RECIPE = copy.deepcopy(recipe)
    args.historical_metrics = args.baseline_reference_metrics
    model = construct_model(args)
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    print(json.dumps({"optimizer_groups": optimizer_summary}, indent=2))
    del optimizer

    if args.smoke_test:
        run_smoke_test(model, args, fold_frame)
        return

    baseline.train(
        model,
        args,
        fold_frame,
        manifest_metadata,
        fold_audit,
        reference,
    )
    rewrite_completed_metrics(output_dir, args, recipe, reference)
    print("Completed Attribute-Guided Wavelet LaST V3 Fold04 metrics and deltas.")


if __name__ == "__main__":
    main()
