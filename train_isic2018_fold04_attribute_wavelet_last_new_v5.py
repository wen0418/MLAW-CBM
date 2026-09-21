#!/usr/bin/env python3
"""Train counterfactual band-selected new V5 on ISIC2018 Fold04.

The optimization/data recipe is identical to the audited Fold04 baseline and
the V3/V4/old-V5 comparisons.  New V5 first uses counterfactual band ablation
to build a filtered high-frequency residual, then keeps V3's semantic/high-
frequency score and original-patch channel-wise Top-K aggregation.  Four
complementary checkpoints are retained so a high-ACC or high-Macro-F1 epoch is
not discarded when BMAC is nearly tied.
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
from training.common import checkpointing
from training.common import isic2018 as common_trainer
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    ATTRIBUTE_PROMPTS,
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_ROUTE_TEMPERATURE,
    DEFAULT_TOP_K,
    ROUTE_NAMES,
    mvpcbm as AttributeWaveletLaSTNewV5,
)


PROTOCOL = "baseline_cv10_attribute_wavelet_last_new_v5_shared_val_test_diagnostic"
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_new_v5.py"
PROJECT_ROOT = Path(__file__).resolve().parent
ORIGIN_ROOT = PROJECT_ROOT
DEFAULT_MANIFEST_DIR = (
    ORIGIN_ROOT / "splits" / "isic2018_baseline_cv10_imagelevel"
)
DEFAULT_REFERENCE = (
    ORIGIN_ROOT / "reference_results" / "isic2018_fold04_baseline_lam2p5.json"
)
TRADEOFF_WEIGHTS = checkpointing.TRADEOFF_WEIGHTS
CHECKPOINT_ALIASES = checkpointing.CHECKPOINT_ALIASES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(10), default=4)
    parser.add_argument("--data-path", default="./dataset/ISIC2018")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tensorboard-dir",
        default="./log/isic2018/baseline_cv10_attribute_wavelet_last_new_v5",
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
        "--counterfactual-route-temperature",
        type=float,
        default=DEFAULT_ROUTE_TEMPERATURE,
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
    parser.add_argument("--attribute-wavelet-eps", type=float, default=DEFAULT_EPS)
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
        "counterfactual_route_temperature",
        "initial_high_feature_scale",
        "initial_high_score_weight",
        "attribute_wavelet_eps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def model_source_path() -> Path:
    imported = Path(inspect.getfile(AttributeWaveletLaSTNewV5)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected new-V5 model import: {imported}; expected {expected}"
        )
    return imported


def new_v5_recipe(args) -> dict:
    recipe = copy.deepcopy(baseline.FIXED_RECIPE)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_new_v5",
            "checkpoint_selection": (
                "standard report uses maximum validation BMAC; additionally "
                "retain maximum ACC, maximum Macro-F1, and weighted tradeoff "
                "checkpoints; later epoch wins exact ties"
            ),
            "standard_report_checkpoint": "best_bmac.pth",
            "retained_checkpoint_aliases": dict(CHECKPOINT_ALIASES),
            "checkpoint_tradeoff_weights": dict(TRADEOFF_WEIGHTS),
            "architecture_ablation": (
                "counterfactual Sparsemax selection over None/LH/HL/HH builds "
                "a filtered high-frequency residual before the unchanged V3 "
                "semantic/high-frequency channel-wise Top-K selector"
            ),
            "base_model_source": "model/mvpcbm_refined.py",
            "v3_model_source": "model/mvpcbm_attribute_wavelet_last_v3.py",
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "concept_target_source": "diagnosis-derived prototype labels",
            "dynamic_concepts_used": False,
            "old_cls_used_for_selection": False,
            "selector_uses_concept_states": False,
            "selector_attribute_names": list(ATTRIBUTE_PROMPTS.keys()),
            "selector_attribute_prompts": dict(ATTRIBUTE_PROMPTS),
            "wavelet_transform": "fixed one-level orthonormal 2-D Haar",
            "wavelet_detail_handling": (
                "exact independent LH/HL/HH-only inverse-Haar residuals are "
                "spatially gated and summed into filtered H-star"
            ),
            "counterfactual_ablation": (
                "direct inverse Haar with LH, HL, or HH coefficient band "
                "zeroed; no residual subtraction"
            ),
            "counterfactual_delta": (
                "cosine(attribute, project(original patch)) minus "
                "cosine(attribute, project(band-ablated patch))"
            ),
            "counterfactual_routes": list(ROUTE_NAMES),
            "counterfactual_routing": (
                "sparsemax over None/LH/HL/HH for every attribute and patch"
            ),
            "counterfactual_route_temperature": float(
                args.counterfactual_route_temperature
            ),
            "counterfactual_negative_effect": (
                "Delta <= 0 excludes that band from H-star; it does not "
                "directly penalize the final selector score"
            ),
            "counterfactual_none_effect": (
                "None consumes Sparsemax probability mass and contributes no "
                "wavelet residual"
            ),
            "attribute_route_fusion": (
                "raw-X spatial attention P0 averages positive band-route "
                "weights over the seven attributes"
            ),
            "counterfactual_delta_shape": (
                "batch x 7 attributes x 3 bands x 196 patches"
            ),
            "counterfactual_route_shape": (
                "batch x 7 attributes x 4 routes x 196 patches"
            ),
            "wavelet_band_gate_shape": "batch x 3 bands x 196 patches",
            "filtered_high_residual_shape": "batch x 196 patches x 768 channels",
            "band_to_channel_assignment": None,
            "raw_high_frequency_magnitude_used": True,
            "ll_guided_conv_gate_used": False,
            "high_frequency_residual_injected_into_patch": True,
            "v3_semantic_and_high_frequency_score_retained": True,
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
                "U_from_filtered_wavelet_patch * (1 + "
                "softplus(layer_high_score_weight) * "
                "sigmoid(channelwise_zscore(abs(filtered_H_star))))"
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


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "new_v5": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "new V5 changed unauthorized Fold04 training settings: "
            f"{differences}"
        )
    return differences


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("new-V5 training requires a CUDA-capable PyTorch setup")
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "new-V5 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = AttributeWaveletLaSTNewV5(
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
        raise FloatingPointError("new-V5 smoke-test loss is non-finite")
    total_loss.backward()

    selector = model.attribute_wavelet_aggregator
    required_gradients = {
        "layer_attribute_residuals": selector.layer_attribute_residuals.grad,
        "raw_high_feature_scales": selector.raw_high_feature_scales.grad,
        "raw_high_score_weights": selector.raw_high_score_weights.grad,
    }
    missing = [name for name, value in required_gradients.items() if value is None]
    nonfinite = [
        name
        for name, value in required_gradients.items()
        if value is not None and not torch.isfinite(value).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            "new-V5 selector gradient failure: "
            f"missing={missing}, nonfinite={nonfinite}"
        )

    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": "refined_mvpcbm_attribute_wavelet_last_new_v5",
                "batch_size": len(labels),
                "classification_loss": classification_loss.item(),
                "concept_loss_mean": concept_loss.item(),
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": sparse_loss.item(),
                "total_loss": total_loss.item(),
                "attribute_count": len(ATTRIBUTE_PROMPTS),
                "attribute_wavelet_top_k": args.attribute_wavelet_top_k,
                "counterfactual_route_temperature": (
                    args.counterfactual_route_temperature
                ),
                "counterfactual_routes": list(ROUTE_NAMES),
                "initial_high_feature_scale": args.initial_high_feature_scale,
                "initial_high_score_weight": args.initial_high_score_weight,
                "raw_high_frequency_magnitude_used": True,
                "high_frequency_residual_injected_into_patch": True,
                "v3_semantic_and_high_frequency_score_retained": True,
                "selector_gradients_verified": True,
                "output_directory_created": False,
            },
            indent=2,
        )
    )


def train_with_retained_checkpoints(
    model,
    args,
    fold_frame,
    manifest_metadata,
    fold_audit,
    reference,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = baseline.SummaryWriter(
        str(Path(args.tensorboard_dir).resolve() / f"fold_{args.fold:02d}")
    )
    report_path = output_dir / "training_report.txt"
    train_loader, val_loader, test_loader = baseline.make_dataloaders(
        model, args, fold_frame
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(baseline.CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "warning": "val=test; checkpoint-selected scores are optimistic",
                    "fold": args.fold,
                    "recipe": baseline.FIXED_RECIPE,
                    "optimizer_groups": optimizer_summary,
                    "fold_audit": fold_audit,
                    "checkpoint_policies": checkpointing.checkpoint_index_payload(
                        {}, output_dir
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n")

    best_ranks = {}
    checkpoint_records = {}
    history = []
    for epoch_index in range(baseline.FIXED_RECIPE["epochs"]):
        epoch = epoch_index + 1
        model.train()
        baseline.schedule_group_lrs(optimizer, epoch_index)
        classification_losses = []
        concept_losses = []
        total_losses = []
        progress = baseline.tqdm(
            train_loader,
            desc=f"new V5 fold {args.fold:02d} epoch {epoch}/100",
            leave=False,
        )
        for data, label, concept_label in progress:
            data = data.float().cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            concept_label = concept_label.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, concept_logits, sparse_loss = model(data)
            concept_loss_sum = torch.zeros((), device=data.device)
            for concept_index, key in enumerate(model.concept_token_dict.keys()):
                concept_loss_sum = concept_loss_sum + F.cross_entropy(
                    concept_logits[key], concept_label[:, concept_index]
                )
            average_concept_loss = concept_loss_sum / len(model.concept_token_dict)
            classification_loss = criterion(logits, label)
            total_loss = (
                classification_loss
                + baseline.FIXED_RECIPE["lambda_cpt"] * average_concept_loss
                + sparse_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite new-V5 loss at fold {args.fold}, epoch {epoch}"
                )
            total_loss.backward()
            optimizer.step()
            classification_losses.append(classification_loss.item())
            concept_losses.append(concept_loss_sum.item())
            total_losses.append(total_loss.item())
            progress.set_postfix(
                cls=f"{classification_loss.item():.4f}",
                concept=f"{concept_loss_sum.item():.4f}",
            )

        val_metrics, _, _ = baseline.evaluate(model, val_loader, criterion)
        row = {
            "epoch": epoch,
            "train_cls_loss": float(np.mean(classification_losses)),
            "train_concept_loss_sum": float(np.mean(concept_losses)),
            "train_total_loss": float(np.mean(total_losses)),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_bmac": val_metrics["bmac"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_tradeoff": checkpointing.tradeoff_score(val_metrics),
            **{
                f"lr_{group['name']}": float(group["lr"])
                for group in optimizer.param_groups
            },
        }
        history.append(row)
        print(
            f"Fold {args.fold:02d} epoch {epoch}: "
            f"{baseline.format_metrics(val_metrics)} | "
            f"Tradeoff={row['val_tradeoff']:.2f}"
        )
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"Epoch {epoch}: {baseline.format_metrics(val_metrics)} | "
                f"Tradeoff={row['val_tradeoff']:.4f}\n"
            )

        improved = checkpointing.update_retained_checkpoints(
            model,
            val_metrics,
            epoch,
            best_ranks,
            checkpoint_records,
            output_dir,
        )
        if improved:
            print(f"Retained checkpoint epoch={epoch}: {', '.join(improved)}")

    checkpointing.write_checkpoint_index(checkpoint_records, output_dir)
    best_record = checkpoint_records["best_bmac"]
    best_path = output_dir / CHECKPOINT_ALIASES["best_bmac"]
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)
    )
    model.cuda()
    best_val_metrics, val_true, val_pred = baseline.evaluate(
        model, val_loader, criterion
    )
    test_metrics, y_true, y_pred = baseline.evaluate(model, test_loader, criterion)
    if not np.array_equal(val_true, y_true) or not np.array_equal(val_pred, y_pred):
        raise RuntimeError("Shared validation/test predictions unexpectedly differ")
    sample_frame = test_loader.dataset.df
    result, class_report = baseline.save_artifacts(
        output_dir,
        args,
        model,
        manifest_metadata,
        fold_audit,
        optimizer_summary,
        reference,
        int(best_record["epoch"]),
        float(best_record["bmac"]),
        best_val_metrics,
        test_metrics,
        y_true,
        y_pred,
        sample_frame,
        history,
    )
    print(f"Fold {args.fold:02d} final best-BMAC: {baseline.format_metrics(result)}")
    print(class_report)
    writer.close()


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
    v3_source = PROJECT_ROOT / "model/mvpcbm_attribute_wavelet_last_v3.py"
    reference_path = Path(args.baseline_reference_metrics).resolve()
    checkpoint_index_path = output_dir / "checkpoint_index.json"
    checkpoint_index = json.loads(
        checkpoint_index_path.read_text(encoding="utf-8")
    )

    metrics.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_new_v5",
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "v3_model_source": str(v3_source),
            "v3_model_source_sha256": baseline.sha256_file(v3_source),
            "base_model_source": str(base_source),
            "base_model_source_sha256": baseline.sha256_file(base_source),
            "trainer_source": str(trainer_path),
            "trainer_sha256": baseline.sha256_file(trainer_path),
            "base_trainer_source": base_trainer_source,
            "base_trainer_sha256": base_trainer_sha256,
            "checkpoint_index": str(checkpoint_index_path),
            "checkpoint_index_sha256": baseline.sha256_file(
                checkpoint_index_path
            ),
            "retained_checkpoints": checkpoint_index,
            "attribute_wavelet_last_new_v5": {
                "attribute_names": recipe["selector_attribute_names"],
                "attribute_prompts": recipe["selector_attribute_prompts"],
                "selector_uses_concept_states": False,
                "wavelet_transform": recipe["wavelet_transform"],
                "detail_handling": recipe["wavelet_detail_handling"],
                "counterfactual_ablation": recipe["counterfactual_ablation"],
                "counterfactual_delta": recipe["counterfactual_delta"],
                "routes": recipe["counterfactual_routes"],
                "routing": recipe["counterfactual_routing"],
                "route_temperature": recipe[
                    "counterfactual_route_temperature"
                ],
                "negative_effect": recipe["counterfactual_negative_effect"],
                "none_effect": recipe["counterfactual_none_effect"],
                "attribute_route_fusion": recipe["attribute_route_fusion"],
                "delta_shape": recipe["counterfactual_delta_shape"],
                "route_shape": recipe["counterfactual_route_shape"],
                "band_gate_shape": recipe["wavelet_band_gate_shape"],
                "filtered_high_residual_shape": recipe[
                    "filtered_high_residual_shape"
                ],
                "band_to_channel_assignment": None,
                "raw_high_frequency_magnitude_used": True,
                "ll_guided_conv_gate_used": False,
                "high_frequency_residual_injected_into_patch": True,
                "v3_semantic_and_high_frequency_score_retained": True,
                "initial_high_feature_scale": recipe[
                    "initial_high_feature_scale"
                ],
                "initial_high_score_weight": recipe[
                    "initial_high_score_weight"
                ],
                "top_k": recipe["attribute_wavelet_top_k"],
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
            "counterfactual_route_temperature": recipe[
                "counterfactual_route_temperature"
            ],
            "initial_high_feature_scale": recipe[
                "initial_high_feature_scale"
            ],
            "initial_high_score_weight": recipe[
                "initial_high_score_weight"
            ],
            "attribute_wavelet_eps": recipe["attribute_wavelet_eps"],
            "selector_uses_concept_states": False,
            "old_cls_used_for_selection": False,
            "raw_high_frequency_magnitude_used": True,
            "high_frequency_residual_injected_into_patch": True,
            "v3_semantic_and_high_frequency_score_retained": True,
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
            "new-V5 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(), args.fold, fold_audit
    )
    recipe = new_v5_recipe(args)
    recipe_differences = validate_training_recipe(recipe)
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
        "v3_model_source": str(
            PROJECT_ROOT / "model/mvpcbm_attribute_wavelet_last_v3.py"
        ),
        "v3_model_source_sha256": baseline.sha256_file(
            PROJECT_ROOT / "model/mvpcbm_attribute_wavelet_last_v3.py"
        ),
        "base_model_source": str(baseline.model_source_path()),
        "base_model_source_sha256": baseline.sha256_file(
            baseline.model_source_path()
        ),
        "base_trainer_source": str(Path(baseline.__file__).resolve()),
        "base_trainer_sha256": baseline.sha256_file(baseline.__file__),
        "new_v5_trainer_source": str(Path(__file__).resolve()),
        "new_v5_trainer_sha256": baseline.sha256_file(__file__),
        "recipe": recipe,
        "training_recipe_difference_vs_origin": recipe_differences,
        "architecture_difference_vs_v3": (
            "counterfactual None/LH/HL/HH routing filters the detail residual "
            "before V3 enhancement; V3 U/E/score and Top-K remain intact"
        ),
        "architecture_difference_vs_v4": (
            "use attribute-semantic counterfactual routing instead of an LL "
            "convolutional gate/threshold"
        ),
        "architecture_difference_vs_old_v5": (
            "move counterfactual routing upstream into wavelet enhancement; "
            "remove signed utility, 3x768 band assignment, tanh/exp scoring, "
            "and restore V3 semantic/high-frequency scoring"
        ),
        "selector_uses_concept_states": False,
        "downstream_patch_to_concept_path_changed": False,
        "checkpoint_policies": checkpointing.checkpoint_index_payload(
            {}, output_dir
        ),
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

    train_with_retained_checkpoints(
        model,
        args,
        fold_frame,
        manifest_metadata,
        fold_audit,
        reference,
    )
    rewrite_completed_metrics(output_dir, args, recipe, reference)
    print("Completed counterfactual band-selected new-V5 Fold04 outputs.")


if __name__ == "__main__":
    main()
