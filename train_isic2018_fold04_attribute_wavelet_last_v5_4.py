#!/usr/bin/env python3
"""Train V5-4 (2x2 ablation C) on ISIC2018 Fold04.

All audited Fold04 settings are identical to new-V5 and V5-3.  The new-V5
selector/CLS branch remains unchanged.  The only architecture change is the
concept AP input:

    V5-3 / cell D:  LN(X + alpha*H*) -> baseline AP -> inherited MLP
    V5-4 / cell C:     X + alpha*H*  -> baseline AP -> inherited MLP

No detach, absolute value, added gate, added projector, or auxiliary loss is
used.  Four complementary checkpoint policies are retained exactly as V5-3.
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
import train_isic2018_fold04_attribute_wavelet_last_new_v5 as new_v5_trainer
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_ROUTE_TEMPERATURE,
    DEFAULT_TOP_K,
)
from model.mvpcbm_attribute_wavelet_last_v5_4 import (
    mvpcbm as AttributeWaveletLaSTV5_4,
)


PROTOCOL = "baseline_cv10_attribute_wavelet_last_v5_4_shared_val_test_diagnostic"
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v5_4.py"
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
        default="./log/isic2018/baseline_cv10_attribute_wavelet_last_v5_4",
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
    imported = Path(inspect.getfile(AttributeWaveletLaSTV5_4)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected V5-4 model import: {imported}; expected {expected}"
        )
    return imported


def v5_4_recipe(args) -> dict:
    recipe = new_v5_trainer.new_v5_recipe(args)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_4",
            "architecture_ablation": (
                "2x2 ablation cell C: retain the complete new-V5 selector/CLS "
                "branch, but feed pre-LayerNorm X + alpha*H-star into the "
                "unchanged baseline concept AP and MLP projector"
            ),
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "downstream_patch_to_concept_path_changed": True,
            "concept_pooling": (
                "the complete inherited AvgPoolProjector applies "
                "AdaptiveAvgPool1d(A) then its unchanged MLP to pre-LayerNorm "
                "X + softplus(s_l) * H-star"
            ),
            "concept_pooling_input_source": (
                "pre-LayerNorm counterfactually enhanced patch "
                "X + softplus(s_l) * H-star"
            ),
            "concept_pooling_layernorm_applied": False,
            "concept_pooling_residual_detached": False,
            "concept_pooling_residual_absolute_value": False,
            "concept_pooling_additional_gate": False,
            "concept_pooling_uses_attribute_attention": False,
            "concept_pooling_uses_selector_topk": False,
            "baseline_adaptive_average_pool_retained": True,
            "concept_projector": (
                "reuse complete inherited AvgPoolProjector unchanged"
            ),
            "concept_similarity": (
                "retain inherited logit-scaled dot product; no added normalize"
            ),
            "additional_auxiliary_loss": None,
            "two_by_two_ablation_cell": "C",
            "selector_cls_input": (
                "unchanged post-LayerNorm W-star = "
                "LN(X + softplus(s_l) * H-star)"
            ),
        }
    )
    return recipe


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "v5_4": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "V5-4 changed unauthorized Fold04 training settings: "
            f"{differences}"
        )
    return differences


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("V5-4 training requires a CUDA-capable PyTorch setup")
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "V5-4 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = AttributeWaveletLaSTV5_4(
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

    # Directly prove the concept AP input is pre-LayerNorm X + alpha*H*, and
    # that the concept path remains differentiable through both X and H*.
    original_probe = torch.randn(
        1, 196, 768, device=data.device, requires_grad=True
    )
    high_probe = torch.randn(
        1, 196, 768, device=data.device, requires_grad=True
    )
    scale_probe = torch.tensor(0.1, device=data.device, requires_grad=True)
    concept_input_probe, projected_probe = (
        model._pre_norm_wavelet_ap_concept_features(
            original_probe,
            high_probe,
            scale_probe,
        )
    )
    expected_probe = original_probe + scale_probe * high_probe
    if not torch.allclose(concept_input_probe, expected_probe):
        raise RuntimeError("V5-4 concept input is not exact pre-LN X + alpha*H*")
    post_norm_probe = model.attribute_wavelet_aggregator.wavelet_patch_norms[0](
        expected_probe
    )
    if torch.allclose(concept_input_probe, post_norm_probe, rtol=1e-5, atol=1e-6):
        raise RuntimeError("V5-4 concept input unexpectedly applies LayerNorm")
    probe_gradients = torch.autograd.grad(
        projected_probe.square().mean(),
        (original_probe, high_probe, scale_probe),
        allow_unused=True,
    )
    if any(value is None for value in probe_gradients):
        raise RuntimeError("V5-4 detached X, H*, or alpha from concept pooling")
    if not all(torch.isfinite(value).all() for value in probe_gradients):
        raise RuntimeError("V5-4 concept pooling probe gradient is non-finite")

    pooling_calls = []
    sampler_calls = []
    original_pooling = model._pre_norm_wavelet_ap_concept_features

    def audited_pooling(patch_tokens, filtered_high_tokens, high_feature_scale):
        concept_input, projected_features = original_pooling(
            patch_tokens,
            filtered_high_tokens,
            high_feature_scale,
        )
        expected = patch_tokens + high_feature_scale * filtered_high_tokens
        if not torch.allclose(concept_input, expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError("V5-4 full forward did not use pre-LN X + alpha*H*")
        pooling_calls.append(
            {
                "original_patch": list(patch_tokens.shape),
                "filtered_high": list(filtered_high_tokens.shape),
                "concept_input": list(concept_input.shape),
                "projected": list(projected_features.shape),
                "high_feature_scale": float(high_feature_scale.detach()),
                "concept_input_requires_grad": concept_input.requires_grad,
            }
        )
        return concept_input, projected_features

    model._pre_norm_wavelet_ap_concept_features = audited_pooling
    sampler_hook = model.Avg.sampler.register_forward_hook(
        lambda _module, inputs, output: sampler_calls.append(
            {"input": list(inputs[0].shape), "output": list(output.shape)}
        )
    )
    logits, concept_logits, sparse_loss = model(data)
    sampler_hook.remove()
    num_layers = len(model.model.visual.trunk.blocks)
    if len(pooling_calls) != num_layers or len(sampler_calls) != num_layers:
        raise RuntimeError(
            "V5-4 must run one pre-LN concept AP per ViT layer; "
            f"pooling={len(pooling_calls)}, sampler={len(sampler_calls)}, "
            f"expected={num_layers}"
        )
    if not all(call["concept_input_requires_grad"] for call in pooling_calls):
        raise RuntimeError("V5-4 detached pre-LN concept patch")

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
        raise FloatingPointError("V5-4 smoke-test loss is non-finite")
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
            "V5-4 selector gradient failure: "
            f"missing={missing}, nonfinite={nonfinite}"
        )

    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": "refined_mvpcbm_attribute_wavelet_last_v5_4",
                "two_by_two_ablation_cell": "C",
                "batch_size": len(labels),
                "classification_loss": classification_loss.item(),
                "concept_loss_mean": concept_loss.item(),
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": sparse_loss.item(),
                "total_loss": total_loss.item(),
                "selector_cls_uses_post_layernorm_w_star": True,
                "concept_input": "X + softplus(s_l) * H-star",
                "concept_input_layernorm_applied": False,
                "concept_residual_detached": False,
                "concept_residual_absolute_value": False,
                "concept_additional_gate": False,
                "concept_pooling_calls": pooling_calls,
                "adaptive_average_pool_calls": sampler_calls,
                "selector_gradients_verified": True,
                "concept_gradient_into_x_verified": True,
                "concept_gradient_into_h_star_verified": True,
                "concept_gradient_into_scale_verified": True,
                "baseline_adaptive_average_pool_retained": True,
                "concept_pooling_uses_attribute_attention": False,
                "concept_pooling_uses_selector_topk": False,
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
            desc=f"V5-4 fold {args.fold:02d} epoch {epoch}/100",
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
                    f"Non-finite V5-4 loss at fold {args.fold}, epoch {epoch}"
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
    # Reuse the audited new-V5 artifact completion, then replace every
    # variant-specific field with MLAW-CBM provenance and cell-C semantics.
    new_v5_trainer.rewrite_completed_metrics(output_dir, args, recipe, reference)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trainer_path = Path(__file__).resolve()
    variant_source = model_source_path()
    old_metadata = metrics.pop("attribute_wavelet_last_new_v5")
    old_metadata.update(
        {
            "concept_pooling": recipe["concept_pooling"],
            "concept_pooling_input_source": recipe[
                "concept_pooling_input_source"
            ],
            "concept_pooling_layernorm_applied": False,
            "concept_pooling_residual_detached": False,
            "concept_pooling_residual_absolute_value": False,
            "concept_pooling_additional_gate": False,
            "two_by_two_ablation_cell": "C",
            "selector_cls_input": recipe["selector_cls_input"],
        }
    )
    metrics.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_4",
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "trainer_source": str(trainer_path),
            "trainer_sha256": baseline.sha256_file(trainer_path),
            "attribute_wavelet_last_v5_4": old_metadata,
            "two_by_two_ablation_cell": "C",
        }
    )
    metrics["runtime_config"].update(
        {
            "concept_pooling": recipe["concept_pooling"],
            "concept_pooling_input_source": recipe[
                "concept_pooling_input_source"
            ],
            "concept_pooling_layernorm_applied": False,
            "concept_pooling_residual_detached": False,
            "concept_pooling_residual_absolute_value": False,
            "concept_pooling_additional_gate": False,
            "two_by_two_ablation_cell": "C",
            "selector_cls_input": recipe["selector_cls_input"],
        }
    )
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
            "V5-4 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(), args.fold, fold_audit
    )
    recipe = v5_4_recipe(args)
    recipe_differences = validate_training_recipe(recipe)
    output_dir = Path(args.output_dir).resolve()
    if not (args.validate_config_only or args.smoke_test) and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    static_audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "fold": args.fold,
        "variant": recipe["variant"],
        "two_by_two_ablation_cell": "C",
        "variant_model_source": str(source),
        "variant_model_source_sha256": source_hash,
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": baseline.sha256_file(__file__),
        "recipe": recipe,
        "training_recipe_difference_vs_origin": recipe_differences,
        "architecture_difference_vs_new_v5": (
            "concept AP receives pre-LayerNorm X + alpha*H-star instead of X; "
            "selector/CLS remains unchanged"
        ),
        "architecture_difference_vs_v5_3": (
            "remove only concept-side LayerNorm; retain identical X + alpha*H-star"
        ),
        "selector_cls_uses_post_layernorm_w_star": True,
        "concept_pooling_input": "X + softplus(s_l) * H-star",
        "concept_pooling_layernorm_applied": False,
        "concept_pooling_residual_detached": False,
        "concept_pooling_residual_absolute_value": False,
        "concept_pooling_additional_gate": False,
        "manifest_sha256": fold_audit["manifest_sha256"],
        "train_size": fold_audit["train_size"],
        "holdout_size": fold_audit["test_size"],
        "val_test_overlap": fold_audit["val_test_overlap"],
        "baseline_reference": {
            key: reference[key]
            for key in ("acc", "bmac", "macro_f1", "weighted_f1", "best_epoch")
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
    print("Completed V5-4 Fold04 outputs.")


if __name__ == "__main__":
    main()
