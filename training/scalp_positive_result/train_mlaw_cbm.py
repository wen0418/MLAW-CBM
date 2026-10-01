#!/usr/bin/env python3
"""Joint-train formal residual-bank scalp MLAW-CBM with per-image VLM labels.

This is the scalp counterpart of the ISIC2018 V5-4 positive-result experiment.
It preserves the controlled scalp recipe, disease path, and validation/test
protocol while using the formal residual-bank ACFS model.  Only concept
supervision changes: the fixed disease prototype is replaced by each image's
``inclusive_output_list`` (or the optional strict ``hard_output_list``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from model.mlaw_cbm_scalp_positive_result import ScalpPositiveResultMLAWCBM
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
)
from training.scalp import common as scalp_common
from training.scalp.config import (
    ATTRIBUTE_PROMPTS,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    CONCEPTS,
    DATALOADER_SEEDS,
    FIXED_RECIPE,
    NEW_V5_RECIPE,
)
from training.scalp_positive_result.dataset import (
    NOTHING_LABEL,
    SUPPORTED_LABEL_COLUMNS,
    ScalpPositiveResultDataset,
    build_joined_manifests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAME = "mlaw_cbm_scalp_positive_result.py"
PROTOCOL = (
    "scalp_mlaw_cbm_residual_bank_joint_per_image_vlm_"
    "shared_val_test_diagnostic_v2"
)
DEFAULT_DATA_ROOT = Path("/home/wen/Desktop/CBM/scalp_dataset/New_scalp")
DEFAULT_PSEUDO_LABEL_DIR = Path(
    "/home/wen/Desktop/CBM/filtered data (VLM check concept label)/scalp_concept"
)
DEFAULT_OUTPUT_NAME = (
    "scalp_mlaw_cbm_residual_bank_joint_inclusive_k98_fold04_recipe_seed43"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--pseudo-label-dir", default=str(DEFAULT_PSEUDO_LABEL_DIR)
    )
    parser.add_argument(
        "--concept-label-column",
        choices=SUPPORTED_LABEL_COLUMNS,
        default="inclusive_output_list",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "scalp" / DEFAULT_OUTPUT_NAME),
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=str(PROJECT_ROOT / "log" / "scalp" / DEFAULT_OUTPUT_NAME),
    )
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-config-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.train_workers < 0 or args.eval_workers < 0:
        parser.error("worker counts must be nonnegative")
    if len(args.expected_model_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF"
        for character in args.expected_model_sha256
    ):
        parser.error("--expected-model-sha256 must be a 64-character hex digest")
    return args


def model_source_path() -> Path:
    imported = Path(inspect.getfile(ScalpPositiveResultMLAWCBM)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected scalp positive-result model: {imported}; expected {expected}"
        )
    return imported


def dependency_sources() -> list[Path]:
    base_spec = scalp_common.get_variant("mlaw")
    paths = scalp_common.dependency_sources(base_spec)
    paths.extend(
        [
            PROJECT_ROOT / "model" / MODEL_FILENAME,
            PROJECT_ROOT
            / "model"
            / "mvpcbm_attribute_wavelet_last_new_v5_positive_result.py",
            Path(__file__).with_name("dataset.py"),
        ]
    )
    return list(dict.fromkeys(path.resolve() for path in paths))


def resolved_recipe(args: argparse.Namespace) -> dict:
    recipe = scalp_common.resolved_recipe(scalp_common.get_variant("mlaw"))
    recipe.update(
        {
            "variant": "scalp_mlaw_cbm_residual_bank_joint_per_image_vlm",
            "protocol": PROTOCOL,
            "base_model": "formal_mlaw_cbm_residual_bank_acfs",
            "counterfactual_construction": "X_minus_R_band",
            "training_mode": "joint_from_scratch_mvp_cbm",
            "concept_target_source": "per-image disease-conditioned VLM label",
            "concept_label_column": args.concept_label_column,
            "concept_label_encoding": (
                "0=nothing; nonzero=original zero-based state plus one"
            ),
            "dynamic_concepts_used": True,
            "concept_loss": "mean cross-entropy over six Attribute heads",
            "positive_result_head_sizes": {
                name: len(states) + 1 for name, states in CONCEPTS.items()
            },
            "original_concept_state_count": sum(
                len(states) for states in CONCEPTS.values()
            ),
            "nothing_logits_used_by_disease_classifier": False,
            "disease_classifier_path_changed": False,
            "joint_total_loss": (
                "class-weighted disease CE + 2.5 * mean six-head concept CE "
                "+ inherited sparse loss"
            ),
            "sequential_training_used": False,
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "pseudo_label_directory": str(Path(args.pseudo_label_dir).resolve()),
            "image_label_join": (
                "per split/class lexicographically sorted filenames joined to "
                "one-based sequential VLM IDs"
            ),
        }
    )
    return recipe


def validate_static_contract() -> dict:
    base_contract = scalp_common.validate_static_model_contract(
        scalp_common.get_variant("mlaw")
    )
    expected_heads = {
        name: len(states) + 1 for name, states in CONCEPTS.items()
    }
    if list(expected_heads.values()) != [5, 5, 5, 5, 4, 4]:
        raise RuntimeError(f"Unexpected scalp positive head sizes: {expected_heads}")
    if not issubclass(
        ScalpPositiveResultMLAWCBM,
        scalp_common.ConfigurableMLAWCBM,
    ):
        raise RuntimeError("Positive-result model no longer inherits scalp MLAW-CBM")
    if (
        scalp_common.ConfigurableMLAWCBM.attribute_wavelet_aggregator_class
        is not ResidualSubtractionCounterfactualBandSelectedWaveletAggregator
    ):
        raise RuntimeError(
            "ConfigurableMLAWCBM no longer selects the formal residual-bank "
            "ACFS implementation"
        )
    base_contract.update(
        {
            "positive_result_model_inherits_mlaw_cbm": True,
            "formal_residual_bank_acfs_verified": True,
            "counterfactual_construction": "X_minus_R_band",
            "positive_result_head_sizes": expected_heads,
            "nothing_label": NOTHING_LABEL,
            "nothing_logits_used_by_disease_classifier": False,
            "disease_classifier_input_state_count": sum(
                len(states) for states in CONCEPTS.values()
            ),
            "unused_original_state_retained_for_architecture_compatibility": {
                "attribute": "Sebum_and_Moisture",
                "state": "other",
                "reason": (
                    "present in original scalp MVP-CBM but absent from VLM "
                    "candidate vocabulary and all class-derived targets"
                ),
            },
        }
    )
    return base_contract


def make_dataloaders(
    model,
    args: argparse.Namespace,
    manifests: dict[str, list[dict]],
    smoke_test: bool = False,
):
    train_transform, eval_transform = scalp_common.make_transforms(model)
    train_set = ScalpPositiveResultDataset(
        manifests["train"], transforms=train_transform
    )
    val_set = ScalpPositiveResultDataset(
        manifests["test"], transforms=eval_transform
    )
    test_set = ScalpPositiveResultDataset(
        manifests["test"], transforms=copy.deepcopy(eval_transform)
    )
    batch_size = 2 if smoke_test else FIXED_RECIPE["batch_size"]
    worker_count_train = 0 if smoke_test else args.train_workers
    worker_count_eval = 0 if smoke_test else args.eval_workers
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count_train,
        drop_last=FIXED_RECIPE["drop_last_train"] and not smoke_test,
        worker_init_fn=scalp_common.seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["train"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count_eval,
        drop_last=False,
        worker_init_fn=scalp_common.seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["val"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count_eval,
        drop_last=False,
        worker_init_fn=scalp_common.seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["test"]),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def construct_model(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("Scalp joint-positive training requires CUDA")
    args.dataset = "scalp"
    args.num_class = len(CLASS_NAMES)
    for name, value in NEW_V5_RECIPE.items():
        setattr(args, name, value)
    scalp_common.set_seed(FIXED_RECIPE["model_seed"])
    model = ScalpPositiveResultMLAWCBM(
        CONCEPTS,
        model_name=FIXED_RECIPE["model_name"],
        config=args,
        attribute_prompts=ATTRIBUTE_PROMPTS,
    )
    if not isinstance(
        model.attribute_wavelet_aggregator,
        ResidualSubtractionCounterfactualBandSelectedWaveletAggregator,
    ):
        raise RuntimeError("Constructed model is not the formal residual-bank ACFS")
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=len(CLASS_NAMES),
    )
    vit.head = nn.Identity()
    model.model.visual.trunk.load_state_dict(vit.state_dict())
    model.cuda()
    scalp_common.set_seed(FIXED_RECIPE["model_seed"])
    return model


def compute_losses(model, data, labels, concept_targets, criterion) -> dict:
    logits, concept_logits, sparse_loss = model(data)
    expected_order = tuple(model.positive_result_attribute_order)
    if tuple(concept_logits) != expected_order:
        raise RuntimeError("Runtime positive-result head order changed")
    concept_loss_sum = torch.zeros((), device=data.device)
    for attribute_index, key in enumerate(expected_order):
        head = concept_logits[key]
        target = concept_targets[:, attribute_index]
        expected_size = model.positive_result_head_sizes[attribute_index]
        if head.ndim != 2 or head.size(1) != expected_size:
            raise RuntimeError(
                f"Unexpected {key} head shape: {tuple(head.shape)}; "
                f"expected second dimension {expected_size}"
            )
        if target.numel() and (
            int(target.min()) < 0 or int(target.max()) >= head.size(1)
        ):
            raise RuntimeError(f"{key} target lies outside [0,{head.size(1)})")
        concept_loss_sum = concept_loss_sum + F.cross_entropy(head, target)
    concept_loss = concept_loss_sum / len(expected_order)
    classification_loss = criterion(logits, labels)
    checkpoint_loss = classification_loss + sparse_loss
    total_loss = (
        classification_loss
        + FIXED_RECIPE["lambda_cpt"] * concept_loss
        + sparse_loss
    )
    return {
        "logits": logits,
        "concept_logits": concept_logits,
        "classification_loss": classification_loss,
        "concept_loss": concept_loss,
        "sparse_loss": sparse_loss,
        "checkpoint_loss": checkpoint_loss,
        "total_loss": total_loss,
    }


def _concept_metrics(
    concept_true: dict[str, list[int]],
    concept_pred: dict[str, list[int]],
) -> dict:
    per_attribute = {}
    attribute_accuracies = []
    attribute_macro_f1s = []
    all_true = []
    all_pred = []
    target_matrix = []
    pred_matrix = []
    for key, states in CONCEPTS.items():
        target = np.asarray(concept_true[key], dtype=np.int64)
        prediction = np.asarray(concept_pred[key], dtype=np.int64)
        labels = list(range(len(states) + 1))
        precision, recall, per_f1, support = precision_recall_fscore_support(
            target,
            prediction,
            labels=labels,
            zero_division=0,
        )
        accuracy = 100.0 * accuracy_score(target, prediction)
        macro_f1 = 100.0 * f1_score(
            target,
            prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        state_names = ["nothing"] + list(states)
        per_attribute[key] = {
            "accuracy": float(accuracy),
            "macro_f1_fixed_labels": float(macro_f1),
            "labels": labels,
            "state_names": state_names,
            "target_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(target.tolist()).items())
            },
            "prediction_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(prediction.tolist()).items())
            },
            "per_state": {
                state_name: {
                    "label": label,
                    "precision": 100.0 * float(precision[label]),
                    "recall": 100.0 * float(recall[label]),
                    "f1": 100.0 * float(per_f1[label]),
                    "support": int(support[label]),
                }
                for label, state_name in enumerate(state_names)
            },
        }
        attribute_accuracies.append(accuracy)
        attribute_macro_f1s.append(macro_f1)
        all_true.extend(target.tolist())
        all_pred.extend(prediction.tolist())
        target_matrix.append(target)
        pred_matrix.append(prediction)
    target_matrix_np = np.stack(target_matrix, axis=1)
    pred_matrix_np = np.stack(pred_matrix, axis=1)
    return {
        "concept_accuracy": float(
            100.0 * accuracy_score(all_true, all_pred)
        ),
        "concept_mean_attribute_accuracy": float(
            np.mean(attribute_accuracies)
        ),
        "concept_mean_attribute_macro_f1": float(
            np.mean(attribute_macro_f1s)
        ),
        "concept_exact_match_all_six": float(
            100.0 * np.mean(np.all(target_matrix_np == pred_matrix_np, axis=1))
        ),
        "concept_per_attribute": per_attribute,
    }


def evaluate_joint(model, dataloader, criterion):
    model.eval()
    totals = Counter()
    sample_count = 0
    disease_true = []
    disease_pred = []
    probabilities = []
    concept_true = {key: [] for key in CONCEPTS}
    concept_pred = {key: [] for key in CONCEPTS}
    with torch.no_grad():
        for data, labels, concept_targets in scalp_common.tqdm(
            dataloader, desc="Evaluate", leave=False
        ):
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            losses = compute_losses(
                model, data, labels, concept_targets, criterion
            )
            batch_size = labels.size(0)
            sample_count += batch_size
            for key in (
                "classification_loss",
                "concept_loss",
                "sparse_loss",
                "checkpoint_loss",
                "total_loss",
            ):
                totals[key] += float(losses[key].item()) * batch_size
            probs = losses["logits"].softmax(dim=1)
            disease_true.extend(labels.cpu().tolist())
            disease_pred.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            for attribute_index, key in enumerate(CONCEPTS):
                concept_true[key].extend(
                    concept_targets[:, attribute_index].cpu().tolist()
                )
                concept_pred[key].extend(
                    losses["concept_logits"][key].argmax(dim=1).cpu().tolist()
                )
    if sample_count == 0:
        raise RuntimeError("Evaluation dataloader is empty")
    truth = np.asarray(disease_true, dtype=np.int64)
    prediction = np.asarray(disease_pred, dtype=np.int64)
    metrics = scalp_common.calculate_metrics(truth, prediction)
    metrics.update(
        {
            "loss": totals["checkpoint_loss"] / sample_count,
            "classification_loss": totals["classification_loss"] / sample_count,
            "concept_loss": totals["concept_loss"] / sample_count,
            "sparse_loss": totals["sparse_loss"] / sample_count,
            "total_loss": totals["total_loss"] / sample_count,
            "joint_total_loss": totals["total_loss"] / sample_count,
            "sample_count": sample_count,
        }
    )
    metrics.update(_concept_metrics(concept_true, concept_pred))
    concept_truth = np.stack(
        [np.asarray(concept_true[key], dtype=np.int64) for key in CONCEPTS],
        axis=1,
    )
    concept_prediction = np.stack(
        [np.asarray(concept_pred[key], dtype=np.int64) for key in CONCEPTS],
        axis=1,
    )
    return (
        metrics,
        truth,
        prediction,
        np.asarray(probabilities, dtype=np.float32),
        concept_truth,
        concept_prediction,
    )


def run_smoke_test(
    model,
    args: argparse.Namespace,
    manifests: dict[str, list[dict]],
    output_existed_before: bool,
) -> None:
    train_loader, _val_loader, _test_loader = make_dataloaders(
        model, args, manifests, smoke_test=True
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    data, labels, concept_targets = next(iter(train_loader))
    data = data.float().cuda(non_blocking=True)
    labels = labels.long().cuda(non_blocking=True)
    concept_targets = concept_targets.long().cuda(non_blocking=True)
    losses = compute_losses(model, data, labels, concept_targets, criterion)
    expected_head_shapes = {
        key: [len(labels), len(states) + 1] for key, states in CONCEPTS.items()
    }
    found_head_shapes = {
        key: list(value.shape) for key, value in losses["concept_logits"].items()
    }
    if found_head_shapes != expected_head_shapes:
        raise RuntimeError(
            f"Positive-result head shapes changed: {found_head_shapes}"
        )
    if not torch.isfinite(losses["total_loss"]):
        raise FloatingPointError("Smoke-test total loss is non-finite")
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
            f"Joint gradient failure: missing={missing}, nonfinite={nonfinite}"
        )
    if model.nothing_logits_used_by_disease_classifier:
        raise RuntimeError("Nothing logits unexpectedly enter disease classifier")
    if Path(args.output_dir).exists() != output_existed_before:
        raise RuntimeError("Smoke test changed the formal output directory")
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "variant": "scalp_mlaw_cbm_residual_bank_joint_per_image_vlm",
                "training_mode": "joint_from_scratch_mvp_cbm",
                "batch_size": len(labels),
                "concept_label_column": args.concept_label_column,
                "concept_targets": concept_targets.cpu().tolist(),
                "classification_logits": list(losses["logits"].shape),
                "concept_head_shapes": found_head_shapes,
                "classification_loss": float(
                    losses["classification_loss"].item()
                ),
                "concept_loss_mean": float(losses["concept_loss"].item()),
                "lambda_cpt": FIXED_RECIPE["lambda_cpt"],
                "sparse_loss": float(losses["sparse_loss"].item()),
                "total_loss": float(losses["total_loss"].item()),
                "joint_gradients_verified": True,
                "nothing_logits_used_by_disease_classifier": False,
                "formal_output_directory_created": False,
                "cuda_peak_memory_mib": (
                    torch.cuda.max_memory_allocated() / 2**20
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_manifest_rows(entries: list[dict]) -> list[dict]:
    result = []
    for entry in entries:
        result.append(
            {
                "split": entry["split"],
                "image_id": entry["image_id"],
                "relative_path": entry["relative_path"],
                "class_id": entry["class_id"],
                "class_name": entry["class_name"],
                "concept_target": json.dumps(entry["concept_target"]),
                "original_zero_based": json.dumps(
                    entry["original_zero_based"]
                ),
                "shifted": json.dumps(entry["shifted"]),
                "hard": json.dumps(entry["hard"]),
                "inclusive": json.dumps(entry["inclusive"]),
            }
        )
    return result


def save_final_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    recipe: dict,
    data_audit: dict,
    mapping_audit: dict,
    static_contract: dict,
    source_audit: dict,
    optimizer_summary: dict,
    manifests: dict[str, list[dict]],
    best_record: dict,
    best_val_metrics: dict,
    test_metrics: dict,
    disease_true: np.ndarray,
    disease_pred: np.ndarray,
    probabilities: np.ndarray,
    concept_true: np.ndarray,
    concept_pred: np.ndarray,
    history: list[dict],
    runtime_seconds: float,
) -> None:
    _write_csv(output_dir / "history.csv", history)
    _write_csv(
        output_dir / "image_label_manifest.csv",
        _jsonable_manifest_rows(manifests["train"] + manifests["test"]),
    )
    prediction_rows = []
    for entry, true_id, pred_id, probs, true_concepts, pred_concepts in zip(
        manifests["test"],
        disease_true,
        disease_pred,
        probabilities,
        concept_true,
        concept_pred,
    ):
        row = {
            "image_id": entry["image_id"],
            "image": entry["relative_path"],
            "true_id": int(true_id),
            "true_class": CLASS_NAMES[int(true_id)],
            "pred_id": int(pred_id),
            "pred_class": CLASS_NAMES[int(pred_id)],
            "all_six_concepts_correct": bool(
                np.array_equal(true_concepts, pred_concepts)
            ),
        }
        row.update(
            {
                f"prob_{class_name}": float(probs[class_id])
                for class_id, class_name in enumerate(CLASS_NAMES)
            }
        )
        for attribute_index, attribute_name in enumerate(CONCEPTS):
            row[f"target_{attribute_name}"] = int(
                true_concepts[attribute_index]
            )
            row[f"pred_{attribute_name}"] = int(
                pred_concepts[attribute_index]
            )
        prediction_rows.append(row)
    _write_csv(output_dir / "predictions.csv", prediction_rows)

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(disease_true, disease_pred, labels=labels)
    _write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true\\pred": name, **dict(zip(CLASS_NAMES, row.tolist()))}
            for name, row in zip(CLASS_NAMES, matrix)
        ],
    )
    class_report = classification_report(
        disease_true,
        disease_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        class_report, encoding="utf-8"
    )
    concept_summary_rows = []
    for attribute_name, values in test_metrics["concept_per_attribute"].items():
        concept_summary_rows.append(
            {
                "attribute": attribute_name,
                "accuracy": values["accuracy"],
                "macro_f1_fixed_labels": values["macro_f1_fixed_labels"],
                "target_counts": json.dumps(values["target_counts"]),
                "prediction_counts": json.dumps(values["prediction_counts"]),
            }
        )
    _write_csv(output_dir / "concept_metrics.csv", concept_summary_rows)

    shared_metrics_verified = all(
        abs(float(best_val_metrics[key]) - float(test_metrics[key])) < 1e-10
        for key in (
            "acc",
            "bmac",
            "macro_f1",
            "weighted_f1",
            "classification_loss",
            "concept_loss",
            "sparse_loss",
            "total_loss",
            "concept_accuracy",
            "concept_mean_attribute_accuracy",
            "concept_mean_attribute_macro_f1",
            "concept_exact_match_all_six",
        )
    )
    if not shared_metrics_verified:
        raise RuntimeError("Validation/test metrics differ for identical files")

    checkpoint_index_path = output_dir / "checkpoint_index.json"
    checkpoint_index = json.loads(
        checkpoint_index_path.read_text(encoding="utf-8")
    )
    model_path = model_source_path()
    dataset_path = Path(__file__).with_name("dataset.py").resolve()
    metrics = {
        "variant": "scalp_mlaw_cbm_residual_bank_joint_per_image_vlm",
        "protocol": PROTOCOL,
        "result_scope": (
            "Validation and final test reuse the same scalp test split; metrics "
            "are diagnostic rather than an independent test estimate."
        ),
        "metric_unit": "percent",
        "best_epoch": int(best_record["epoch"]),
        "acc": float(test_metrics["acc"]),
        "bmac": float(test_metrics["bmac"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "per_class_precision": test_metrics["per_class_precision"],
        "per_class_recall": test_metrics["per_class_recall"],
        "per_class_f1": test_metrics["per_class_f1"],
        "per_class_metrics": test_metrics["per_class"],
        "losses": {
            key: float(test_metrics[key])
            for key in (
                "classification_loss",
                "concept_loss",
                "sparse_loss",
                "total_loss",
            )
        },
        "concept_metrics_at_best_checkpoint": {
            key: test_metrics[key]
            for key in (
                "concept_accuracy",
                "concept_mean_attribute_accuracy",
                "concept_mean_attribute_macro_f1",
                "concept_exact_match_all_six",
                "concept_per_attribute",
            )
        },
        "validation_metrics_at_best_checkpoint": best_val_metrics,
        "shared_val_test_metrics_verified": shared_metrics_verified,
        "training_runtime_seconds": float(runtime_seconds),
        "training_runtime_hours": float(runtime_seconds / 3600.0),
        "data_path": str(Path(args.data_path).resolve()),
        "pseudo_label_directory": str(Path(args.pseudo_label_dir).resolve()),
        "train_size": len(manifests["train"]),
        "val_size": len(manifests["test"]),
        "test_size": len(manifests["test"]),
        "val_test_overlap": "100%: both use data-path/test",
        "class_order": list(CLASS_NAMES),
        "class_counts": data_audit["counts"],
        "class_weights": list(CLASS_WEIGHTS),
        "concept_supervision": {
            "label_column": args.concept_label_column,
            "label_encoding": (
                "0=nothing; nonzero=original zero-based state plus one"
            ),
            "attribute_order": list(CONCEPTS),
            "head_sizes": {
                name: len(states) + 1 for name, states in CONCEPTS.items()
            },
            "nothing_label": NOTHING_LABEL,
            "nothing_logits_used_by_disease_classifier": False,
            "loss": "mean six-head cross entropy",
            "lambda_cpt": FIXED_RECIPE["lambda_cpt"],
        },
        "mapping_audit": mapping_audit,
        "hyperparameters": recipe,
        "seed": FIXED_RECIPE["model_seed"],
        "dataloader_seeds": dict(DATALOADER_SEEDS),
        "checkpoint_index": checkpoint_index,
        "checkpoint_index_path": str(checkpoint_index_path),
        "checkpoint_index_sha256": scalp_common.sha256_file(
            checkpoint_index_path
        ),
        "model_source": str(model_path),
        "model_source_sha256": scalp_common.sha256_file(model_path),
        "model_dependency_sources": {
            str(path): scalp_common.sha256_file(path)
            for path in dependency_sources()
        },
        "dataset_source": str(dataset_path),
        "dataset_source_sha256": scalp_common.sha256_file(dataset_path),
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": scalp_common.sha256_file(__file__),
        "entry_source": str(Path(args.entry_source).resolve()),
        "entry_sha256": scalp_common.sha256_file(args.entry_source),
        "data_inventory_sha256": data_audit["inventory_sha256"],
        "static_model_contract": static_contract,
        "preserved_source_audit": source_audit,
        "optimizer_groups": optimizer_summary,
        "runtime_config": {
            "gpu_argument": args.gpu,
            "train_workers": args.train_workers,
            "eval_workers": args.eval_workers,
            "output_dir": str(output_dir),
            "tensorboard_dir": str(Path(args.tensorboard_dir).resolve()),
            "concept_label_column": args.concept_label_column,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32_enabled": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat = {
        "variant": metrics["variant"],
        "best_epoch": metrics["best_epoch"],
        "acc": metrics["acc"],
        "bmac": metrics["bmac"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "concept_accuracy": test_metrics["concept_accuracy"],
        "concept_mean_attribute_accuracy": test_metrics[
            "concept_mean_attribute_accuracy"
        ],
        "concept_mean_attribute_macro_f1": test_metrics[
            "concept_mean_attribute_macro_f1"
        ],
        "concept_exact_match_all_six": test_metrics[
            "concept_exact_match_all_six"
        ],
        "training_runtime_seconds": runtime_seconds,
    }
    for class_name, values in test_metrics["per_class"].items():
        for key in ("precision", "recall", "f1", "support"):
            flat[f"{key}_{class_name}"] = values[key]
    _write_csv(output_dir / "metrics.csv", [flat])
    (output_dir / "README.md").write_text(
        "# Scalp MLAW-CBM joint per-image VLM result\n\n"
        f"- Concept label: `{args.concept_label_column}`\n"
        "- Training: disease and six concept heads jointly from scratch\n"
        "- Standard checkpoint: `best_bmac.pth`\n"
        "- Validation and test both use `test/`; scores are diagnostic.\n\n"
        "Main files: `metrics.json`, `metrics.csv`, `concept_metrics.csv`, "
        "`classification_report.txt`, `predictions.csv`, "
        "`image_label_manifest.csv`, and `checkpoint_index.json`.\n",
        encoding="utf-8",
    )


def train(
    model,
    args: argparse.Namespace,
    manifests: dict[str, list[dict]],
    recipe: dict,
    data_audit: dict,
    mapping_audit: dict,
    static_contract: dict,
    source_audit: dict,
    optimizer_summary: dict,
    configuration_audit: dict,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_dir = Path(args.tensorboard_dir).resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = scalp_common.SummaryWriter(str(tensorboard_dir))
    (output_dir / "configuration_audit.json").write_text(
        json.dumps(configuration_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "image_label_manifest.csv",
        _jsonable_manifest_rows(manifests["train"] + manifests["test"]),
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        model, args, manifests
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    optimizer, _ = scalp_common.build_optimizer(model)
    history = []
    best_ranks = {}
    checkpoint_records = {}
    report_path = output_dir / "training_report.txt"
    report_path.write_text(
        "Scalp MLAW-CBM joint per-image VLM concept supervision\n"
        f"Protocol: {PROTOCOL}\n"
        f"Concept label: {args.concept_label_column}\n"
        "WARNING: Validation and final test both use data-path/test. "
        "Scores are diagnostic.\n\n",
        encoding="utf-8",
    )
    start_time = time.perf_counter()

    for epoch_index in range(FIXED_RECIPE["epochs"]):
        epoch = epoch_index + 1
        scalp_common.schedule_group_lrs(optimizer, epoch_index)
        model.train()
        sums = Counter()
        sample_count = 0
        progress = scalp_common.tqdm(
            train_loader,
            desc=f"MLAW+VLM scalp epoch {epoch}/{FIXED_RECIPE['epochs']}",
            leave=False,
        )
        for data, labels, concept_targets in progress:
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(
                model, data, labels, concept_targets, criterion
            )
            if not torch.isfinite(losses["total_loss"]):
                raise FloatingPointError(
                    f"Non-finite joint loss at epoch {epoch}"
                )
            losses["total_loss"].backward()
            optimizer.step()
            batch_size = labels.size(0)
            sample_count += batch_size
            for key in (
                "classification_loss",
                "concept_loss",
                "sparse_loss",
                "total_loss",
            ):
                sums[key] += float(losses[key].item()) * batch_size
            progress.set_postfix(
                cls=f"{losses['classification_loss'].item():.4f}",
                concept=f"{losses['concept_loss'].item():.4f}",
                total=f"{losses['total_loss'].item():.4f}",
            )
        if sample_count == 0:
            raise RuntimeError("Training dataloader produced no samples")
        train_losses = {
            key: value / sample_count for key, value in sums.items()
        }
        (
            val_metrics,
            _val_true,
            _val_pred,
            _val_probabilities,
            _val_concept_true,
            _val_concept_pred,
        ) = evaluate_joint(model, val_loader, criterion)
        row = {
            "epoch": epoch,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "bridge_lr": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{
                f"val_{key}": val_metrics[key]
                for key in (
                    "classification_loss",
                    "concept_loss",
                    "sparse_loss",
                    "total_loss",
                    "acc",
                    "bmac",
                    "macro_f1",
                    "weighted_f1",
                    "concept_accuracy",
                    "concept_mean_attribute_accuracy",
                    "concept_mean_attribute_macro_f1",
                    "concept_exact_match_all_six",
                )
            },
            "val_tradeoff": scalp_common.tradeoff_score(val_metrics),
        }
        history.append(row)
        _write_csv(output_dir / "history.csv", history)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        improved = scalp_common.update_retained_checkpoints(
            model,
            val_metrics,
            epoch,
            best_ranks,
            checkpoint_records,
            output_dir,
        )
        last_path = output_dir / "last.pth"
        torch.save(model.state_dict(), last_path)
        last_record = scalp_common.checkpoint_record(
            "last", epoch, val_metrics, last_path, last_path
        )
        scalp_common.write_checkpoint_index(
            checkpoint_records, last_record, output_dir
        )
        message = (
            f"Epoch {epoch:03d} | {scalp_common.format_metrics(val_metrics)} | "
            f"Concept-ACC={val_metrics['concept_accuracy']:.2f}% | "
            "Concept-Mean-Macro-F1="
            f"{val_metrics['concept_mean_attribute_macro_f1']:.2f}% | "
            f"Exact6={val_metrics['concept_exact_match_all_six']:.2f}% | "
            f"retained={','.join(improved) if improved else 'none'}"
        )
        print(message)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    runtime_seconds = time.perf_counter() - start_time
    writer.flush()
    writer.close()
    best_record = checkpoint_records["best_bmac"]
    best_path = output_dir / scalp_common.CHECKPOINT_ALIASES["best_bmac"]
    incompatible = model.load_state_dict(
        scalp_common.load_state(best_path), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict best-checkpoint reload failed: {incompatible}")
    model.cuda()
    (
        best_val_metrics,
        val_true,
        val_pred,
        val_probabilities,
        val_concept_true,
        val_concept_pred,
    ) = evaluate_joint(model, val_loader, criterion)
    (
        test_metrics,
        disease_true,
        disease_pred,
        probabilities,
        concept_true,
        concept_pred,
    ) = evaluate_joint(model, test_loader, criterion)
    arrays = (
        (val_true, disease_true),
        (val_pred, disease_pred),
        (val_probabilities, probabilities),
        (val_concept_true, concept_true),
        (val_concept_pred, concept_pred),
    )
    if not all(np.array_equal(left, right) for left, right in arrays):
        raise RuntimeError("Shared validation/test predictions unexpectedly differ")
    save_final_artifacts(
        output_dir,
        args,
        recipe,
        data_audit,
        mapping_audit,
        static_contract,
        source_audit,
        optimizer_summary,
        manifests,
        best_record,
        best_val_metrics,
        test_metrics,
        disease_true,
        disease_pred,
        probabilities,
        concept_true,
        concept_pred,
        history,
        runtime_seconds,
    )
    final_message = (
        f"Completed scalp MLAW+VLM; best epoch {best_record['epoch']}: "
        f"{scalp_common.format_metrics(test_metrics)} | "
        f"Concept-ACC={test_metrics['concept_accuracy']:.2f}% | "
        f"runtime={runtime_seconds / 3600.0:.2f} h\n"
        f"Standard checkpoint: {best_path}"
    )
    print(final_message)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + final_message + "\n")


def main() -> None:
    args = parse_args()
    args.entry_source = str(Path(__file__).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.pseudo_label_dir = str(
        Path(args.pseudo_label_dir).expanduser().resolve()
    )
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args.tensorboard_dir = str(
        Path(args.tensorboard_dir).expanduser().resolve()
    )

    source = model_source_path()
    source_hash = scalp_common.sha256_file(source)
    if source_hash.lower() != args.expected_model_sha256.lower():
        raise RuntimeError(
            "Scalp positive-result model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )
    data_root = Path(args.data_path)
    data_audit = scalp_common.validate_data_layout(data_root)
    manifests, mapping_audit = build_joined_manifests(
        data_root,
        Path(args.pseudo_label_dir),
        args.concept_label_column,
    )
    if len(manifests["train"]) != data_audit["train_size"]:
        raise RuntimeError("Joined train manifest changed controlled split size")
    if len(manifests["test"]) != data_audit["test_size"]:
        raise RuntimeError("Joined test manifest changed controlled split size")
    source_audit = scalp_common.validate_preserved_sources()
    static_contract = validate_static_contract()
    recipe = resolved_recipe(args)
    output_dir = Path(args.output_dir)
    output_existed_before = output_dir.exists()
    if not (args.validate_config_only or args.smoke_test) and output_existed_before:
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "variant": recipe["variant"],
        "protocol": PROTOCOL,
        "training_mode": recipe["training_mode"],
        "model_source": str(source),
        "model_source_sha256": source_hash,
        "model_dependency_sources": {
            str(path): scalp_common.sha256_file(path)
            for path in dependency_sources()
        },
        "trainer_source": str(Path(__file__).resolve()),
        "data_path": str(data_root),
        "pseudo_label_directory": args.pseudo_label_dir,
        "concept_label_column": args.concept_label_column,
        "data_inventory_sha256": data_audit["inventory_sha256"],
        "class_counts": data_audit["counts"],
        "train_size": data_audit["train_size"],
        "val_size": data_audit["test_size"],
        "test_size": data_audit["test_size"],
        "val_test_overlap": "100%: both map to data-path/test",
        "mapping_audit": mapping_audit,
        "static_model_contract": static_contract,
        "preserved_source_audit": source_audit,
        "recipe": recipe,
        "planned_output": str(output_dir),
        "planned_output_exists": output_existed_before,
        "formal_output_will_be_created": not (
            args.validate_config_only or args.smoke_test
        ),
    }
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.validate_config_only:
        if output_dir.exists() != output_existed_before:
            raise RuntimeError("Configuration validation changed output directory")
        print("Configuration validation completed; no model/output created.")
        return

    model = construct_model(args)
    optimizer, optimizer_summary = scalp_common.build_optimizer(model)
    print(json.dumps({"optimizer_groups": optimizer_summary}, indent=2))
    del optimizer
    if args.smoke_test:
        run_smoke_test(model, args, manifests, output_existed_before)
        return
    train(
        model,
        args,
        manifests,
        recipe,
        data_audit,
        mapping_audit,
        static_contract,
        source_audit,
        optimizer_summary,
        audit,
    )


if __name__ == "__main__":
    main()
