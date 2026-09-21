#!/usr/bin/env python3
"""Train V5-7 original-AP plus wavelet-energy fusion on ISIC2018 Fold04.

All audited Fold04 settings and the complete V5-4 selector/CLS branch remain
unchanged.  The concept path is the requested ablation (option 3):

    content = inherited_MLP(AP_7(X))
    R       = softplus(s_l) * H*
    energy  = energy_MLP(sqrt(AP_7(R^2) + eps) - sqrt(eps))
    fused   = content + softplus(g[l,a]) * energy

The seven-bin baseline AP therefore receives original patches only.  Pure
wavelet RMS energy is pooled over the same bins, independently projected, and
added immediately before the unchanged concept-state similarity calculation.
No std branch is included in this experiment.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_isic2018_baseline_cv10 as baseline
import train_isic2018_fold04_attribute_wavelet_last_v5_4 as v5_4_trainer
from training.common import isic2018 as common_trainer
from model.mvpcbm_attribute_wavelet_last_v5_7 import (
    INITIAL_WAVELET_ENERGY_GATE,
    mvpcbm as AttributeWaveletLaSTV5_7,
)


PROTOCOL = "baseline_cv10_attribute_wavelet_last_v5_7_shared_val_test_diagnostic"
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v5_7.py"
PROJECT_ROOT = Path(__file__).resolve().parent
ORIGIN_ROOT = PROJECT_ROOT
DEFAULT_MANIFEST_DIR = (
    ORIGIN_ROOT / "splits" / "isic2018_baseline_cv10_imagelevel"
)
DEFAULT_REFERENCE = (
    ORIGIN_ROOT / "reference_results" / "isic2018_fold04_baseline_lam2p5.json"
)
V5_4_TENSORBOARD_DEFAULT = (
    "./log/isic2018/baseline_cv10_attribute_wavelet_last_v5_4"
)
V5_7_TENSORBOARD_DEFAULT = (
    "./log/isic2018/baseline_cv10_attribute_wavelet_last_v5_7"
)
ENERGY_PROJECTOR_PARAMETER_COUNT = 656384
ENERGY_GATE_PARAMETER_COUNT = 84


def parse_args():
    # Reuse V5-4's validated selector hyperparameters and Fold04 CLI.  Only
    # redirect the default TensorBoard directory to keep runs isolated.
    args = v5_4_trainer.parse_args()
    if args.tensorboard_dir == V5_4_TENSORBOARD_DEFAULT:
        args.tensorboard_dir = V5_7_TENSORBOARD_DEFAULT
    return args


def model_source_path() -> Path:
    imported = Path(inspect.getfile(AttributeWaveletLaSTV5_7)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected V5-7 model import: {imported}; expected {expected}"
        )
    return imported


def v5_7_recipe(args) -> dict:
    recipe = v5_4_trainer.v5_4_recipe(args)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_7",
            "architecture_ablation": (
                "option 3: restore the exact baseline AP(X) semantic path and "
                "add a separately projected RMS-energy branch computed only "
                "from R = softplus(s_l) * H-star"
            ),
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "two_by_two_ablation_cell": "original_AP_plus_wavelet_RMS_energy",
            "selector_cls_input": (
                "unchanged post-LayerNorm W-star = "
                "LN(X + softplus(s_l) * H-star)"
            ),
            "selector_input_layernorm_applied": True,
            "selective_cls_output_layernorm_applied": True,
            "concept_pooling": (
                "inherited seven-bin AdaptiveAvgPool1d applies to original X; "
                "a second identical seven-bin pool computes zero-referenced "
                "RMS energy from only scaled counterfactual H-star"
            ),
            "concept_pooling_input_source": "original patch X only",
            "concept_wavelet_residual": "R = softplus(s_l) * H-star",
            "concept_wavelet_energy": (
                "sqrt(AdaptiveAvgPool1d(R squared) + eps) - sqrt(eps)"
            ),
            "concept_content_layernorm_applied": False,
            "concept_energy_layernorm_applied": False,
            "concept_pooling_spatial_support": (
                "the same seven fixed baseline sequence bins for content and "
                "energy; 196 patches divide into seven bins of 28"
            ),
            "baseline_adaptive_average_pool_retained": True,
            "concept_content_projector": (
                "reuse inherited AvgPoolProjector.mlp_projector unchanged"
            ),
            "concept_energy_projector": (
                "independent two-linear-layer 768-to-512-to-512 MLP; all "
                "biases zeroed so zero energy maps to exact zero"
            ),
            "concept_energy_fusion": (
                "projected_content + softplus(per-layer-per-Attribute gate) "
                "times projected_energy"
            ),
            "concept_energy_gate_shape": [12, 7],
            "concept_energy_gate_initial_value": INITIAL_WAVELET_ENERGY_GATE,
            "concept_pooling_additional_parameters": (
                ENERGY_PROJECTOR_PARAMETER_COUNT + ENERGY_GATE_PARAMETER_COUNT
            ),
            "concept_projector": (
                "inherited content MLP plus an independent zero-bias two-layer "
                "wavelet-energy MLP"
            ),
            "concept_pooling_additional_gate": True,
            "concept_pooling_residual_squared_for_energy": True,
            "concept_std_branch_included": False,
            "concept_pooling_uses_attribute_attention": False,
            "concept_pooling_uses_selector_topk": False,
            "concept_similarity": (
                "retain inherited logit-scaled dot product after fusion"
            ),
            "shared_training_loop_source": (
                "train_isic2018_fold04_attribute_wavelet_last_v5_4.py"
            ),
            "shared_training_loop_sha256": baseline.sha256_file(
                Path(inspect.getfile(v5_4_trainer)).resolve()
            ),
        }
    )
    return recipe


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {"baseline": baseline.FIXED_RECIPE.get(key), "v5_7": recipe.get(key)}
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "V5-7 changed unauthorized Fold04 training settings: "
            f"{differences}"
        )
    return differences


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("V5-7 training requires a CUDA-capable PyTorch setup")
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "V5-7 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = AttributeWaveletLaSTV5_7(
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

    selector = model.attribute_wavelet_aggregator
    if not all(
        isinstance(module, nn.LayerNorm) for module in selector.wavelet_patch_norms
    ):
        raise RuntimeError("V5-7 must retain V5-4 selector LN1 at every layer")
    if not isinstance(selector.output_norm, nn.LayerNorm):
        raise RuntimeError("V5-7 must retain selective-CLS output_norm")
    if any(
        left is right
        for left, right in zip(
            model.Avg.mlp_projector.parameters(),
            model.wavelet_energy_projector.parameters(),
        )
    ):
        raise RuntimeError("V5-7 content and energy projectors share parameters")
    for module in model.wavelet_energy_projector.modules():
        if (
            isinstance(module, nn.Linear)
            and module.bias is not None
            and not torch.equal(module.bias, torch.zeros_like(module.bias))
        ):
            raise RuntimeError("V5-7 energy-projector biases must start at zero")

    gates = model.wavelet_energy_gates()
    expected_gates = torch.full_like(gates, INITIAL_WAVELET_ENERGY_GATE)
    if tuple(gates.shape) != (12, 7) or not torch.allclose(
        gates, expected_gates, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "V5-7 energy gates must initialize to one with shape [12,7]"
        )

    zero_patch = torch.randn(1, 196, 768, device=data.device)
    zero_high = torch.zeros_like(zero_patch)
    zero_scale = torch.ones((), device=data.device)
    _, zero_fused, zero_details = (
        model._original_ap_plus_wavelet_energy_concept_features(
            zero_patch,
            zero_high,
            zero_scale,
            layer_index=0,
            return_details=True,
        )
    )
    if not torch.equal(
        zero_details["energy_visual"],
        torch.zeros_like(zero_details["energy_visual"]),
    ):
        raise RuntimeError("V5-7 zero residual did not produce zero RMS energy")
    if not torch.equal(
        zero_details["energy_projected"],
        torch.zeros_like(zero_details["energy_projected"]),
    ):
        raise RuntimeError("V5-7 zero energy did not map to zero evidence")
    if not torch.allclose(
        zero_fused,
        zero_details["content_projected"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("V5-7 zero energy changed the baseline content path")

    pooling_calls = []
    sampler_calls = []
    original_pooling = model._original_ap_plus_wavelet_energy_concept_features

    def audited_pooling(
        patch_tokens,
        filtered_high_tokens,
        high_feature_scale,
        layer_index,
        return_details=False,
    ):
        content_input, fused, details = original_pooling(
            patch_tokens,
            filtered_high_tokens,
            high_feature_scale,
            layer_index,
            return_details=True,
        )
        residual = high_feature_scale * filtered_high_tokens
        expected_content = model.Avg.sampler(
            patch_tokens.transpose(1, 2)
        ).transpose(1, 2)
        expected_mean_square = model.Avg.sampler(
            residual.square().transpose(1, 2)
        ).transpose(1, 2)
        epsilon = float(model.attribute_wavelet_eps)
        expected_energy = torch.sqrt(expected_mean_square + epsilon) - (
            epsilon**0.5
        )
        expected_content_projected = model.Avg.mlp_projector(expected_content)
        expected_energy_projected = model.wavelet_energy_projector(
            expected_energy
        )
        expected_gate = model.wavelet_energy_gates()[layer_index].view(1, 7, 1)
        expected_fused = (
            expected_content_projected
            + expected_gate * expected_energy_projected
        )
        checks = (
            (content_input, patch_tokens, "content input is not original X"),
            (details["residual"], residual, "energy residual is not alpha*H*"),
            (details["content_visual"], expected_content, "AP(X) mismatch"),
            (details["energy_visual"], expected_energy, "RMS energy mismatch"),
            (fused, expected_fused, "projected fusion mismatch"),
        )
        for actual, expected, message in checks:
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"V5-7 {message} at layer {layer_index}")
        pooling_calls.append(
            {
                "layer": layer_index,
                "content_input": list(content_input.shape),
                "content_ap": list(details["content_visual"].shape),
                "wavelet_energy": list(details["energy_visual"].shape),
                "fused_projected": list(fused.shape),
                "mean_energy_gate": float(expected_gate.mean().detach()),
            }
        )
        if return_details:
            return content_input, fused, details
        return content_input, fused

    model._original_ap_plus_wavelet_energy_concept_features = audited_pooling
    sampler_hook = model.Avg.sampler.register_forward_hook(
        lambda _module, inputs, output: sampler_calls.append(
            {"input": list(inputs[0].shape), "output": list(output.shape)}
        )
    )
    logits, concept_logits, sparse_loss = model(data)
    sampler_hook.remove()
    num_layers = len(model.model.visual.trunk.blocks)
    # Two audited calls inside the wrapper plus two calls in the actual helper.
    if len(pooling_calls) != num_layers or len(sampler_calls) != num_layers * 4:
        raise RuntimeError(
            "V5-7 must compute AP(X) and AP(R^2) once per layer; "
            f"pooling={len(pooling_calls)}, sampler_audit_calls={len(sampler_calls)}"
        )

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
        raise FloatingPointError("V5-7 smoke-test loss is non-finite")
    total_loss.backward()

    first_energy_linear = next(
        module
        for module in model.wavelet_energy_projector.modules()
        if isinstance(module, nn.Linear)
    )
    required_gradients = {
        "wavelet_energy_projector": first_energy_linear.weight.grad,
        "raw_wavelet_energy_gates": model.raw_wavelet_energy_gates.grad,
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
    zero_grad = [
        name
        for name, value in required_gradients.items()
        if value is not None and float(value.norm()) <= 0.0
    ]
    if missing or nonfinite or zero_grad:
        raise RuntimeError(
            "V5-7 gradient failure: "
            f"missing={missing}, nonfinite={nonfinite}, zero={zero_grad}"
        )

    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": "refined_mvpcbm_attribute_wavelet_last_v5_7",
                "batch_size": len(labels),
                "classification_loss": classification_loss.item(),
                "concept_loss_mean": concept_loss.item(),
                "lambda_cpt": baseline.FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": sparse_loss.item(),
                "total_loss": total_loss.item(),
                "selector_input": "LN(X + softplus(s_l) * H-star)",
                "concept_content_input": "original X only",
                "concept_wavelet_input": "R = softplus(s_l) * H-star",
                "content_pool": "baseline AdaptiveAvgPool1d(7)",
                "energy_pool": "zero-referenced RMS over the same seven bins",
                "std_branch_included": False,
                "energy_gate_shape": list(gates.shape),
                "energy_gate_initial_value": INITIAL_WAVELET_ENERGY_GATE,
                "pooling_calls": pooling_calls,
                "required_gradients_verified": True,
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
    # Run the exact audited V5-4 loader/loss/optimizer/checkpoint loop.  Only
    # rename its progress label; this has no effect on training state.
    original_tqdm = baseline.tqdm

    def v5_7_tqdm(*positional, **keywords):
        description = keywords.get("desc")
        if isinstance(description, str):
            keywords["desc"] = description.replace("V5-4", "V5-7")
        return original_tqdm(*positional, **keywords)

    baseline.tqdm = v5_7_tqdm
    try:
        v5_4_trainer.train_with_retained_checkpoints(
            model,
            args,
            fold_frame,
            manifest_metadata,
            fold_audit,
            reference,
        )
    finally:
        baseline.tqdm = original_tqdm


def rewrite_completed_metrics(
    output_dir: Path,
    args,
    recipe: dict,
    reference: dict,
) -> None:
    v5_4_trainer.rewrite_completed_metrics(output_dir, args, recipe, reference)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trainer_path = Path(__file__).resolve()
    variant_source = model_source_path()
    variant_metadata = metrics.pop("attribute_wavelet_last_v5_4")
    variant_metadata.update(
        {
            "concept_pooling": recipe["concept_pooling"],
            "concept_pooling_input_source": recipe[
                "concept_pooling_input_source"
            ],
            "concept_wavelet_residual": recipe["concept_wavelet_residual"],
            "concept_wavelet_energy": recipe["concept_wavelet_energy"],
            "concept_content_layernorm_applied": False,
            "concept_energy_layernorm_applied": False,
            "concept_pooling_spatial_support": recipe[
                "concept_pooling_spatial_support"
            ],
            "baseline_adaptive_average_pool_retained": True,
            "concept_content_projector": recipe["concept_content_projector"],
            "concept_energy_projector": recipe["concept_energy_projector"],
            "concept_projector": recipe["concept_projector"],
            "concept_energy_fusion": recipe["concept_energy_fusion"],
            "concept_energy_gate_shape": recipe[
                "concept_energy_gate_shape"
            ],
            "concept_energy_gate_initial_value": recipe[
                "concept_energy_gate_initial_value"
            ],
            "concept_std_branch_included": False,
            "concept_pooling_additional_gate": True,
            "concept_pooling_residual_squared_for_energy": True,
            "concept_pooling_additional_parameters": recipe[
                "concept_pooling_additional_parameters"
            ],
            "two_by_two_ablation_cell": recipe["two_by_two_ablation_cell"],
            "selector_cls_input": recipe["selector_cls_input"],
            "selector_input_layernorm_applied": True,
            "selective_cls_output_layernorm_applied": True,
        }
    )
    metrics.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_7",
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "trainer_source": str(trainer_path),
            "trainer_sha256": baseline.sha256_file(trainer_path),
            "attribute_wavelet_last_v5_7": variant_metadata,
            "two_by_two_ablation_cell": recipe["two_by_two_ablation_cell"],
        }
    )
    metrics["runtime_config"].update(
        {
            "concept_pooling": recipe["concept_pooling"],
            "concept_pooling_input_source": recipe[
                "concept_pooling_input_source"
            ],
            "concept_wavelet_residual": recipe["concept_wavelet_residual"],
            "concept_wavelet_energy": recipe["concept_wavelet_energy"],
            "concept_content_layernorm_applied": False,
            "concept_energy_layernorm_applied": False,
            "concept_pooling_spatial_support": recipe[
                "concept_pooling_spatial_support"
            ],
            "baseline_adaptive_average_pool_retained": True,
            "concept_content_projector": recipe["concept_content_projector"],
            "concept_energy_projector": recipe["concept_energy_projector"],
            "concept_projector": recipe["concept_projector"],
            "concept_energy_fusion": recipe["concept_energy_fusion"],
            "concept_energy_gate_shape": recipe[
                "concept_energy_gate_shape"
            ],
            "concept_energy_gate_initial_value": recipe[
                "concept_energy_gate_initial_value"
            ],
            "concept_std_branch_included": False,
            "concept_pooling_additional_gate": True,
            "concept_pooling_residual_squared_for_energy": True,
            "concept_pooling_additional_parameters": recipe[
                "concept_pooling_additional_parameters"
            ],
            "two_by_two_ablation_cell": recipe["two_by_two_ablation_cell"],
            "selector_cls_input": recipe["selector_cls_input"],
            "selector_input_layernorm_applied": True,
            "selective_cls_output_layernorm_applied": True,
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
            "V5-7 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(), args.fold, fold_audit
    )
    recipe = v5_7_recipe(args)
    recipe_differences = validate_training_recipe(recipe)
    output_dir = Path(args.output_dir).resolve()
    if not (args.validate_config_only or args.smoke_test) and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    static_audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "fold": args.fold,
        "variant": recipe["variant"],
        "variant_model_source": str(source),
        "variant_model_source_sha256": source_hash,
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": baseline.sha256_file(__file__),
        "shared_training_loop_source": recipe["shared_training_loop_source"],
        "shared_training_loop_sha256": recipe[
            "shared_training_loop_sha256"
        ],
        "recipe": recipe,
        "training_recipe_difference_vs_origin": recipe_differences,
        "architecture_difference_vs_v5_4": (
            "concept content becomes exact baseline AP(X); scaled H-star is "
            "removed from the signed content path and retained only as a "
            "separately projected seven-bin RMS-energy residual"
        ),
        "selector_cls_unchanged_from_v5_4": True,
        "selector_input_layernorm_applied": True,
        "concept_content_input": "original X only",
        "concept_wavelet_input": "R = softplus(s_l) * H-star",
        "concept_content_layernorm_applied": False,
        "concept_energy_layernorm_applied": False,
        "baseline_adaptive_average_pool_retained": True,
        "concept_std_branch_included": False,
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
    print("Completed V5-7 Fold04 outputs.")


if __name__ == "__main__":
    main()
