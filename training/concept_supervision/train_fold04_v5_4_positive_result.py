#!/usr/bin/env python3
"""Joint-train V5-4 on Fold04 with per-image positive-result labels.

All audited V5-4 Fold04 data, optimization, disease-classification, and
checkpoint settings are preserved.  The only experimental change is the same
concept-supervision interface used by the earlier New-V5 positive-result run:
each of seven Attribute heads receives a learnable ``nothing`` class at index
0 and is trained against ``positive_result`` (or optionally ``hard_result``).

Validation and test intentionally remain the shared Fold04 holdout for direct
diagnostic comparison with the existing runs.  They are not an independent
test estimate.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import train_isic2018_baseline_cv10 as baseline  # noqa: E402
import train_isic2018_fold04_attribute_wavelet_last_new_v5_positive_result as positive_common  # noqa: E402,E501
import train_isic2018_fold04_attribute_wavelet_last_v5_4 as v5_4_trainer  # noqa: E402,E501
from training.common import checkpointing  # noqa: E402
from training.common import isic2018 as common_trainer  # noqa: E402
from model.mvpcbm_attribute_wavelet_last_new_v5 import (  # noqa: E402
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_ROUTE_TEMPERATURE,
    DEFAULT_TOP_K,
)
from model.mvpcbm_attribute_wavelet_last_new_v5_positive_result import (  # noqa: E402
    NOTHING_LABEL,
    NOTHING_STATE_NAME,
)
from model.mvpcbm_attribute_wavelet_last_v5_4_positive_result import (  # noqa: E402
    mvpcbm as PositiveResultV5_4,
)


PROTOCOL = (
    "baseline_cv10_attribute_wavelet_last_v5_4_positive_result_"
    "joint_shared_val_test_diagnostic"
)
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v5_4_positive_result.py"
BASE_MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v5_4.py"
DEFAULT_MANIFEST_DIR = positive_common.DEFAULT_MANIFEST_DIR
DEFAULT_REFERENCE = positive_common.DEFAULT_REFERENCE
DEFAULT_PSEUDO_LABEL_CSV = positive_common.DEFAULT_PSEUDO_LABEL_CSV
SUPPORTED_LABEL_COLUMNS = positive_common.SUPPORTED_LABEL_COLUMNS
CHECKPOINT_ALIASES = checkpointing.CHECKPOINT_ALIASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(10), default=4)
    parser.add_argument("--data-path", default="./dataset/ISIC2018")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument(
        "--pseudo-label-csv",
        default=str(DEFAULT_PSEUDO_LABEL_CSV),
    )
    parser.add_argument(
        "--concept-label-column",
        choices=SUPPORTED_LABEL_COLUMNS,
        default="positive_result",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tensorboard-dir",
        default=(
            "./log/isic2018/"
            "baseline_cv10_attribute_wavelet_last_v5_4_positive_result"
        ),
    )
    parser.add_argument(
        "--baseline-reference-metrics",
        default=str(DEFAULT_REFERENCE),
    )
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument(
        "--attribute-wavelet-top-k",
        type=int,
        default=DEFAULT_TOP_K,
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
    parser.add_argument(
        "--attribute-wavelet-eps",
        type=float,
        default=DEFAULT_EPS,
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
        "counterfactual_route_temperature",
        "initial_high_feature_scale",
        "initial_high_score_weight",
        "attribute_wavelet_eps",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and positive"
            )
    return args


def model_source_path() -> Path:
    imported = Path(inspect.getfile(PositiveResultV5_4)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected positive-result V5-4 import: {imported}; "
            f"expected {expected}"
        )
    return imported


def parent_model_source_path() -> Path:
    return PROJECT_ROOT / "model" / BASE_MODEL_FILENAME


def positive_result_recipe(args) -> dict:
    recipe = v5_4_trainer.v5_4_recipe(args)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_4_positive_result",
            "training_mode": "joint_from_scratch_mvp_cbm",
            "concept_target_source": (
                "per-image disease-conditioned VLM pseudo-label"
            ),
            "concept_label_csv": str(Path(args.pseudo_label_csv).resolve()),
            "concept_label_column": args.concept_label_column,
            "concept_label_encoding": (
                "0=nothing; nonzero=original zero-based state plus one"
            ),
            "dynamic_concepts_used": True,
            "concept_loss": (
                "mean cross-entropy over seven Attribute heads with a "
                "prepended nothing class"
            ),
            "nothing_class_implementation": (
                "one learnable scalar reference per Attribute, prepended after "
                "MCSAF; no nothing text embedding"
            ),
            "nothing_label": NOTHING_LABEL,
            "nothing_state_name": NOTHING_STATE_NAME,
            "original_concept_state_count": sum(
                len(states) for states in baseline.CONCEPTS.values()
            ),
            "positive_result_head_sizes": {
                name: len(states) + 1
                for name, states in baseline.CONCEPTS.items()
            },
            "disease_classifier_input": (
                "unchanged original 34-dimensional V5-4 MCSAF activation"
            ),
            "nothing_logits_used_by_disease_classifier": False,
            "joint_total_loss": (
                "class-weighted disease CE + 2.5 * mean seven-head concept CE "
                "+ inherited MCSAF sparse loss"
            ),
            "classifier_gradient_reaches_concept_branch": True,
            "sequential_training_used": False,
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "parent_v5_4_model_source": f"model/{BASE_MODEL_FILENAME}",
        }
    )
    return recipe


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "positive_result_v5_4": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "Positive-result V5-4 changed unauthorized Fold04 settings: "
            f"{differences}"
        )
    return differences


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Positive-result V5-4 training requires a CUDA-capable PyTorch setup"
        )
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "Positive-result V5-4 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )
    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = PositiveResultV5_4(
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
    train_loader, _, _ = positive_common.make_dataloaders(
        model,
        args,
        fold_frame,
        smoke_test=True,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            baseline.CLASS_WEIGHTS,
            dtype=torch.float32,
        ).cuda()
    )
    data, labels, concept_targets = next(iter(train_loader))
    data = data.float().cuda(non_blocking=True)
    labels = labels.long().cuda(non_blocking=True)
    concept_targets = concept_targets.long().cuda(non_blocking=True)
    losses = positive_common.compute_losses(
        model,
        data,
        labels,
        concept_targets,
        criterion,
    )
    if not torch.isfinite(losses["total_loss"]):
        raise FloatingPointError("Positive-result V5-4 smoke-test loss is non-finite")
    losses["total_loss"].backward()

    selector = model.attribute_wavelet_aggregator
    required_gradients = {
        "layer_attribute_residuals": selector.layer_attribute_residuals.grad,
        "raw_high_feature_scales": selector.raw_high_feature_scales.grad,
        "raw_high_score_weights": selector.raw_high_score_weights.grad,
        "nothing_logits": model.nothing_logits.grad,
        "disease_classifier": model.cls_head.weight.grad,
    }
    missing = [name for name, value in required_gradients.items() if value is None]
    nonfinite = [
        name
        for name, value in required_gradients.items()
        if value is not None and not torch.isfinite(value).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            "Positive-result V5-4 joint gradient failure: "
            f"missing={missing}, nonfinite={nonfinite}"
        )

    original_probe = torch.randn(1, 196, 768, device=data.device)
    high_probe = torch.randn(1, 196, 768, device=data.device)
    scale_probe = torch.tensor(0.1, device=data.device)
    concept_input, _ = model._pre_norm_wavelet_ap_concept_features(
        original_probe,
        high_probe,
        scale_probe,
    )
    if not torch.allclose(
        concept_input,
        original_probe + scale_probe * high_probe,
    ):
        raise RuntimeError("Positive-result V5-4 no longer uses pre-LN X+alpha*H*")

    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": (
                    "refined_mvpcbm_attribute_wavelet_last_v5_4_positive_result"
                ),
                "training_mode": "joint_from_scratch_mvp_cbm",
                "batch_size": len(labels),
                "concept_label_column": args.concept_label_column,
                "concept_head_shapes": {
                    key: list(value.shape)
                    for key, value in losses["concept_logits"].items()
                },
                "classification_loss": losses["classification_loss"].item(),
                "concept_loss_mean": losses["concept_loss"].item(),
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": losses["sparse_loss"].item(),
                "total_loss": losses["total_loss"].item(),
                "joint_gradients_verified": True,
                "v5_4_pre_layernorm_concept_input_verified": True,
                "nothing_logits_used_by_disease_classifier": False,
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
    pseudo_label_audit: dict,
    label_join_audit: dict,
    best_val_metrics: dict,
    test_metrics: dict,
) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trainer_path = Path(__file__).resolve()
    variant_source = model_source_path()
    parent_source = parent_model_source_path()
    reference_path = Path(args.baseline_reference_metrics).resolve()
    checkpoint_index_path = output_dir / "checkpoint_index.json"
    checkpoint_index = json.loads(
        checkpoint_index_path.read_text(encoding="utf-8")
    )
    metrics.update(
        {
            "variant": (
                "refined_mvpcbm_attribute_wavelet_last_v5_4_positive_result"
            ),
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "parent_v5_4_model_source": str(parent_source),
            "parent_v5_4_model_source_sha256": baseline.sha256_file(
                parent_source
            ),
            "trainer_source": str(trainer_path),
            "trainer_sha256": baseline.sha256_file(trainer_path),
            "pseudo_label_audit": pseudo_label_audit,
            "label_join_audit": label_join_audit,
            "checkpoint_index": str(checkpoint_index_path),
            "checkpoint_index_sha256": baseline.sha256_file(
                checkpoint_index_path
            ),
            "retained_checkpoints": checkpoint_index,
            "training_mode": "joint_from_scratch_mvp_cbm",
            "concept_supervision": {
                "label_column": args.concept_label_column,
                "label_encoding": (
                    "0=nothing; nonzero=original zero-based state plus one"
                ),
                "attribute_order": list(baseline.CONCEPTS),
                "nothing_label": NOTHING_LABEL,
                "nothing_state_name": NOTHING_STATE_NAME,
                "head_sizes": {
                    name: len(states) + 1
                    for name, states in baseline.CONCEPTS.items()
                },
                "loss": "mean seven-head cross entropy",
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "nothing_logits_used_by_disease_classifier": False,
            },
            "concept_metrics_at_best_checkpoint": {
                key: test_metrics[key]
                for key in (
                    "concept_loss",
                    "concept_accuracy",
                    "concept_mean_attribute_accuracy",
                    "concept_mean_attribute_macro_f1",
                    "concept_per_attribute",
                )
            },
            "validation_joint_metrics_at_best_checkpoint": {
                key: best_val_metrics[key]
                for key in (
                    "classification_loss",
                    "concept_loss",
                    "sparse_loss",
                    "joint_total_loss",
                    "concept_accuracy",
                    "concept_mean_attribute_accuracy",
                    "concept_mean_attribute_macro_f1",
                )
            },
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
            "pseudo_label_csv": str(Path(args.pseudo_label_csv).resolve()),
            "concept_label_column": args.concept_label_column,
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


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = model_source_path()
    source_hash = baseline.sha256_file(source)
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "Positive-result V5-4 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(),
        args.fold,
    )
    target_lookup, pseudo_label_audit = positive_common.load_pseudo_labels(
        Path(args.pseudo_label_csv),
        args.concept_label_column,
    )
    fold_frame, label_join_audit = positive_common.attach_fold_labels(
        fold_frame,
        target_lookup,
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(),
        args.fold,
        fold_audit,
    )
    recipe = positive_result_recipe(args)
    recipe_differences = validate_training_recipe(recipe)
    output_dir = Path(args.output_dir).resolve()
    if not (args.validate_config_only or args.smoke_test) and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    static_audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "fold": args.fold,
        "training_mode": "joint_from_scratch_mvp_cbm",
        "variant": recipe["variant"],
        "two_by_two_ablation_cell": "C",
        "variant_model_source": str(source),
        "variant_model_source_sha256": source_hash,
        "parent_v5_4_model_source": str(parent_model_source_path()),
        "parent_v5_4_model_source_sha256": baseline.sha256_file(
            parent_model_source_path()
        ),
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": baseline.sha256_file(__file__),
        "recipe": recipe,
        "training_recipe_difference_vs_origin": recipe_differences,
        "pseudo_label_audit": pseudo_label_audit,
        "label_join_audit": label_join_audit,
        "classifier_path_changed": False,
        "disease_classifier_input": (
            "unchanged original 34-dimensional V5-4 MCSAF activation"
        ),
        "nothing_logits_used_by_disease_classifier": False,
        "selector_cls_uses_post_layernorm_w_star": True,
        "concept_pooling_input": "X + softplus(s_l) * H-star",
        "concept_pooling_layernorm_applied": False,
        "manifest_sha256": fold_audit["manifest_sha256"],
        "train_size": fold_audit["train_size"],
        "holdout_size": fold_audit["test_size"],
        "val_test_overlap": fold_audit["val_test_overlap"],
        "checkpoint_policies": checkpointing.checkpoint_index_payload(
            {}, output_dir
        ),
        "planned_output": str(output_dir),
        "planned_output_exists": output_dir.exists(),
    }
    print(json.dumps(static_audit, indent=2, ensure_ascii=False))
    if args.validate_config_only:
        print("Configuration validation completed; no training/output created.")
        return

    baseline.FIXED_RECIPE = copy.deepcopy(recipe)
    args.historical_metrics = args.baseline_reference_metrics
    args.positive_result_display_name = "positive-result V5-4"
    model = construct_model(args)
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    print(json.dumps({"optimizer_groups": optimizer_summary}, indent=2))
    del optimizer

    if args.smoke_test:
        run_smoke_test(model, args, fold_frame)
        return

    best_val_metrics, test_metrics = (
        positive_common.train_with_retained_checkpoints(
            model,
            args,
            fold_frame,
            manifest_metadata,
            fold_audit,
            reference,
            recipe,
            pseudo_label_audit,
            label_join_audit,
        )
    )
    rewrite_completed_metrics(
        output_dir,
        args,
        recipe,
        reference,
        pseudo_label_audit,
        label_join_audit,
        best_val_metrics,
        test_metrics,
    )
    print("Completed joint positive-result V5-4 Fold04 outputs.")


if __name__ == "__main__":
    main()
