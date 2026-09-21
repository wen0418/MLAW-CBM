#!/usr/bin/env python3
"""Standalone two-stage V5-4 training with original MVP-CBM concept labels.

Stage 1 trains the V5-4 concept branch with the disease-fixed, zero-based
``CONCEPT_LABEL_MAP`` used by the audited Fold04 baseline.  Disease CE is not
used and the classifier stays frozen.  Stage 2 reloads the best concept
checkpoint, freezes every non-classifier parameter, and trains the unchanged
linear classifier on the same 34 raw MCSAF concept activations.

No pseudo-label CSV is read.  In particular, ``positive_result``,
``hard_result``, and the additional ``nothing`` state are not used.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import precision_score, recall_score  # noqa: E402

import train_isic2018_baseline_cv10 as baseline  # noqa: E402
import train_isic2018_fold04_attribute_wavelet_last_v5_4 as v5_4_trainer  # noqa: E402,E501
from training.common import checkpointing  # noqa: E402
from training.common import isic2018 as common_trainer  # noqa: E402
from model.mvpcbm_attribute_wavelet_last_v5_4 import (  # noqa: E402
    mvpcbm as AttributeWaveletLaSTV5_4,
)


PROTOCOL = "baseline_cv10_attribute_wavelet_last_v5_4_baseline_concept_sequential"
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_v5_4.py"
CHECKPOINT_ALIASES = checkpointing.CHECKPOINT_ALIASES
DEFAULT_JOINT_REFERENCE = (
    PROJECT_ROOT
    / "output/isic2018/"
    "baseline_cv10_attribute_wavelet_last_v5_4_k98_imglevel_fold04_seed43/"
    "metrics.json"
)


def timed_start() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed_elapsed(start: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return float(time.perf_counter() - start)


def format_runtime(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detailed_disease_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    labels = list(range(len(baseline.CLASS_NAMES)))
    matrix = baseline.confusion_matrix(y_true, y_pred, labels=labels)
    standard = baseline.calculate_metrics(y_true, y_pred)
    overall = {
        "accuracy": float(standard["acc"]),
        "balanced_accuracy": float(standard["bmac"]),
        "macro_precision": float(
            100.0
            * precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            100.0
            * recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(standard["macro_f1"]),
        "weighted_precision": float(
            100.0
            * precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            100.0
            * recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(standard["weighted_f1"]),
        "sample_count": int(len(y_true)),
    }
    per_class = {}
    total = int(matrix.sum())
    for class_index, class_name in enumerate(baseline.CLASS_NAMES):
        true_positive = int(matrix[class_index, class_index])
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive

        def safe_ratio(numerator: int, denominator: int) -> float:
            return float(numerator / denominator) if denominator else 0.0

        precision = safe_ratio(
            true_positive,
            true_positive + false_positive,
        )
        recall = safe_ratio(
            true_positive,
            true_positive + false_negative,
        )
        specificity = safe_ratio(
            true_negative,
            true_negative + false_positive,
        )
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        per_class[class_name] = {
            "support": true_positive + false_negative,
            "accuracy_ovr": 100.0
            * safe_ratio(true_positive + true_negative, total),
            "balanced_accuracy_ovr": 50.0 * (recall + specificity),
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "specificity": 100.0 * specificity,
            "f1": 100.0 * f1,
            "tp": true_positive,
            "fn": false_negative,
            "fp": false_positive,
            "tn": true_negative,
        }
    return {
        "metric_unit": "percent except support/counts",
        "per_class_balanced_accuracy_definition": (
            "one-vs-rest mean of recall/sensitivity and specificity"
        ),
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(baseline.CLASS_NAMES),
    }


def print_detailed_disease_metrics(metrics: dict) -> None:
    overall = metrics["overall"]
    print("\nFinal overall disease metrics (%)")
    print(
        "ACC={accuracy:.2f} | BMAC={balanced_accuracy:.2f} | "
        "Macro-P={macro_precision:.2f} | Macro-R={macro_recall:.2f} | "
        "Macro-F1={macro_f1:.2f} | Weighted-P={weighted_precision:.2f} | "
        "Weighted-R={weighted_recall:.2f} | Weighted-F1={weighted_f1:.2f}".format(
            **overall
        )
    )
    print(
        "\nPer-class one-vs-rest metrics (%)\n"
        "Class   Support   ACC     BACC    Precision  Recall  Specificity  F1"
    )
    for class_name in baseline.CLASS_NAMES:
        row = metrics["per_class"][class_name]
        print(
            f"{class_name:<7} {row['support']:>7d}  "
            f"{row['accuracy_ovr']:>6.2f}  "
            f"{row['balanced_accuracy_ovr']:>6.2f}  "
            f"{row['precision']:>9.2f}  "
            f"{row['recall']:>6.2f}  "
            f"{row['specificity']:>11.2f}  "
            f"{row['f1']:>6.2f}"
        )


def parse_args():
    return v5_4_trainer.parse_args()


def model_source_path() -> Path:
    imported = Path(inspect.getfile(AttributeWaveletLaSTV5_4)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected V5-4 import: {imported}; expected {expected}"
        )
    return imported


def sequential_recipe(args) -> dict:
    recipe = v5_4_trainer.v5_4_recipe(args)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_v5_4_baseline_concept_sequential",
            "training_mode": "concept_first_sequential_raw34",
            "concept_target_source": (
                "baseline.CONCEPT_LABEL_MAP; disease-fixed zero-based labels"
            ),
            "concept_label_encoding": "original zero-based state per Attribute",
            "pseudo_label_csv_used": False,
            "positive_result_used": False,
            "hard_result_used": False,
            "nothing_class_used": False,
            "stage1_epochs": baseline.FIXED_RECIPE["epochs"],
            "stage1_trainable": (
                "visual backbone, V5-4 selector, ICPM, MCSAF, concept path"
            ),
            "stage1_frozen": "linear disease classifier",
            "stage1_loss": (
                "2.5 * mean seven-head concept CE + inherited MCSAF sparse loss"
            ),
            "stage1_checkpoint_selection": (
                "maximum validation mean per-Attribute macro-F1; later epoch wins ties"
            ),
            "stage2_epochs": baseline.FIXED_RECIPE["epochs"],
            "stage2_trainable": "unchanged linear disease classifier only",
            "stage2_frozen": "all visual and concept parameters",
            "stage2_classifier_input": (
                "unchanged concatenated 34 raw MCSAF concept activations"
            ),
            "stage2_loss": "class-weighted disease cross-entropy",
            "stage2_checkpoint_selection": recipe["checkpoint_selection"],
            "stage2_feature_cache_used": False,
            "stage2_train_augmentation_retained": True,
            "classifier_gradient_reaches_concept_branch": False,
            "variant_model_source": f"model/{MODEL_FILENAME}",
        }
    )
    return recipe


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "sequential": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "Sequential V5-4 changed unauthorized Fold04 settings: "
            f"{differences}"
        )
    return differences


def set_stage1_trainable(model) -> None:
    if not hasattr(model, "_sequential_stage1_trainable_names"):
        model._sequential_stage1_trainable_names = frozenset(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("cls_head.")
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            name in model._sequential_stage1_trainable_names
        )
        if not parameter.requires_grad:
            parameter.grad = None


def set_stage2_trainable(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    for parameter in model.cls_head.parameters():
        parameter.requires_grad = True


def trainable_parameter_names(model) -> list[str]:
    return sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def concatenate_original_concept_features(
    model,
    concept_logits: dict[str, torch.Tensor],
) -> torch.Tensor:
    expected_order = tuple(model.concept_token_dict)
    if tuple(concept_logits) != expected_order:
        raise RuntimeError(
            "Concept head order changed: "
            f"{tuple(concept_logits)} != {expected_order}"
        )
    features = torch.cat([concept_logits[key] for key in expected_order], dim=1)
    expected_width = sum(len(states) for states in baseline.CONCEPTS.values())
    if tuple(features.shape[1:]) != (expected_width,):
        raise RuntimeError(
            f"Expected [B,{expected_width}] classifier input, got "
            f"{tuple(features.shape)}"
        )
    return features


def state_subset_sha256(model, include_classifier: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        is_classifier = name.startswith("cls_head.")
        if is_classifier != include_classifier:
            continue
        digest.update(name.encode("utf-8"))
        try:
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.numpy().tobytes())
        except ValueError:
            digest.update(b"<uninitialized>")
    return digest.hexdigest()


def compute_concept_loss(model, concept_logits, concept_targets):
    expected_order = tuple(model.concept_token_dict)
    if tuple(concept_logits) != expected_order:
        raise RuntimeError("Runtime concept head order changed")
    loss_sum = torch.zeros((), device=concept_targets.device)
    for attribute_index, key in enumerate(expected_order):
        head = concept_logits[key]
        target = concept_targets[:, attribute_index]
        if head.size(1) != len(baseline.CONCEPTS[key]):
            raise RuntimeError(f"Unexpected baseline head size for {key}")
        if target.numel() and (
            int(target.min()) < 0 or int(target.max()) >= head.size(1)
        ):
            raise RuntimeError(f"Baseline target outside {key} head range")
        loss_sum = loss_sum + F.cross_entropy(head, target)
    return loss_sum / len(expected_order)


def evaluate_concepts(model, dataloader) -> dict:
    model.eval()
    totals = Counter()
    sample_count = 0
    truth = {key: [] for key in baseline.CONCEPTS}
    prediction = {key: [] for key in baseline.CONCEPTS}
    with torch.no_grad():
        for data, _labels, concept_targets in baseline.tqdm(
            dataloader,
            desc="Evaluate concepts",
            leave=False,
        ):
            data = data.float().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            _disease_logits, concept_logits, sparse_loss = model(data)
            concept_loss = compute_concept_loss(
                model,
                concept_logits,
                concept_targets,
            )
            batch_size = data.size(0)
            sample_count += batch_size
            totals["concept_loss"] += float(concept_loss.item()) * batch_size
            totals["sparse_loss"] += float(sparse_loss.item()) * batch_size
            for attribute_index, key in enumerate(baseline.CONCEPTS):
                truth[key].extend(
                    concept_targets[:, attribute_index].cpu().tolist()
                )
                prediction[key].extend(
                    concept_logits[key].argmax(dim=1).cpu().tolist()
                )
    if sample_count == 0:
        raise RuntimeError("Concept evaluation dataloader is empty")

    per_attribute = {}
    accuracies = []
    macro_f1s = []
    all_truth = []
    all_prediction = []
    for key in baseline.CONCEPTS:
        target = np.asarray(truth[key], dtype=np.int64)
        pred = np.asarray(prediction[key], dtype=np.int64)
        labels = list(range(len(baseline.CONCEPTS[key])))
        accuracy = 100.0 * baseline.accuracy_score(target, pred)
        macro_f1 = 100.0 * baseline.f1_score(
            target,
            pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        per_attribute[key] = {
            "accuracy": float(accuracy),
            "macro_f1_fixed_labels": float(macro_f1),
            "labels": labels,
            "target_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(target.tolist()).items())
            },
            "prediction_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(pred.tolist()).items())
            },
        }
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        all_truth.extend(target.tolist())
        all_prediction.extend(pred.tolist())

    return {
        "concept_loss": totals["concept_loss"] / sample_count,
        "sparse_loss": totals["sparse_loss"] / sample_count,
        "concept_accuracy": float(
            100.0 * baseline.accuracy_score(all_truth, all_prediction)
        ),
        "concept_mean_attribute_accuracy": float(np.mean(accuracies)),
        "concept_mean_attribute_macro_f1": float(np.mean(macro_f1s)),
        "concept_per_attribute": per_attribute,
    }


def build_classifier_optimizer(model):
    parameters = [
        parameter for parameter in model.cls_head.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Stage 2 classifier has no trainable parameters")
    selected = {id(parameter) for parameter in parameters}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if selected != trainable:
        raise RuntimeError("Stage 2 optimizer includes a non-classifier parameter")
    group = {
        "name": "classifier",
        "params": parameters,
        "lr": baseline.FIXED_RECIPE["bridge_lr"],
        "initial_lr": baseline.FIXED_RECIPE["bridge_lr"],
        "weight_decay": baseline.FIXED_RECIPE["weight_decay"],
    }
    optimizer = torch.optim.AdamW([group])
    summary = {
        "classifier": {
            "lr": group["initial_lr"],
            "weight_decay": group["weight_decay"],
            "parameter_tensors": len(parameters),
            "parameter_elements": int(sum(p.numel() for p in parameters)),
            "parameter_names": trainable_parameter_names(model),
        }
    }
    return optimizer, summary


def stage2_forward(model, data):
    model.eval()
    with torch.no_grad():
        forward_logits, concept_logits, _sparse_loss = model(data)
        features = concatenate_original_concept_features(model, concept_logits)
    classifier_logits = model.cls_head(features.detach())
    return classifier_logits, forward_logits.detach(), features.detach()


def evaluate_disease_stage2(model, dataloader, criterion):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, labels, _concept_targets in baseline.tqdm(
            dataloader,
            desc="Evaluate classifier",
            leave=False,
        ):
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            _unused, concept_logits, _sparse_loss = model(data)
            features = concatenate_original_concept_features(model, concept_logits)
            logits = model.cls_head(features)
            losses.append(float(criterion(logits, labels).item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    metrics = baseline.calculate_metrics(truth, prediction)
    metrics["loss"] = float(np.mean(losses))
    return metrics, truth, prediction


def run_smoke_test(model, args, fold_frame) -> None:
    train_loader, _, _ = baseline.make_dataloaders(model, args, fold_frame)
    data, labels, concept_targets = next(iter(train_loader))
    data = data.float().cuda(non_blocking=True)
    labels = labels.long().cuda(non_blocking=True)
    concept_targets = concept_targets.long().cuda(non_blocking=True)

    set_stage1_trainable(model)
    optimizer, _ = baseline.build_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    _logits, concept_logits, sparse_loss = model(data)
    concept_loss = compute_concept_loss(model, concept_logits, concept_targets)
    stage1_loss = baseline.FIXED_RECIPE["lambda_cpt"] * concept_loss + sparse_loss
    stage1_loss.backward()
    if any(parameter.grad is not None for parameter in model.cls_head.parameters()):
        raise RuntimeError("Stage 1 produced a classifier gradient")

    model.zero_grad(set_to_none=True)
    set_stage2_trainable(model)
    optimizer, _ = build_classifier_optimizer(model)
    optimizer.zero_grad(set_to_none=True)
    logits, forward_logits, features = stage2_forward(model, data)
    if not torch.allclose(logits.detach(), forward_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Raw34 reconstruction differs from V5-4 classifier input")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            baseline.CLASS_WEIGHTS,
            dtype=torch.float32,
            device=data.device,
        )
    )
    stage2_loss = criterion(logits, labels)
    stage2_loss.backward()
    non_classifier_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("cls_head.") and parameter.grad is not None
    ]
    if non_classifier_gradients:
        raise RuntimeError(
            f"Stage 2 leaked gradients into concept branch: {non_classifier_gradients}"
        )
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "variant": "V5-4 baseline-concept sequential Raw34",
                "stage1_label_source": "baseline.CONCEPT_LABEL_MAP",
                "positive_result_used": False,
                "stage1_classifier_gradient_absent": True,
                "stage2_feature_shape": list(features.shape),
                "stage2_only_classifier_trainable": True,
                "raw34_reconstruction_verified": True,
                "output_directory_created": False,
            },
            indent=2,
        )
    )


def train_stage1(model, args, fold_frame, output_dir: Path):
    stage_runtime_start = timed_start()
    stage_dir = output_dir / "stage1_concept"
    stage_dir.mkdir(parents=True, exist_ok=False)
    writer = baseline.SummaryWriter(
        str(Path(args.tensorboard_dir).resolve() / "stage1_concept")
    )
    train_loader, val_loader, _ = baseline.make_dataloaders(
        model,
        args,
        fold_frame,
    )
    set_stage1_trainable(model)
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    classifier_hash_before = state_subset_sha256(model, include_classifier=True)
    history = []
    best_score = float("-inf")
    best_epoch = None
    best_path = stage_dir / "best_concept_macro_f1.pth"

    for epoch_index in range(baseline.FIXED_RECIPE["epochs"]):
        epoch_runtime_start = timed_start()
        epoch = epoch_index + 1
        model.train()
        model.cls_head.eval()
        baseline.schedule_group_lrs(optimizer, epoch_index)
        sums = Counter()
        batches = 0
        progress = baseline.tqdm(
            train_loader,
            desc=f"V5-4 baseline-concept stage1 epoch {epoch}/100",
            leave=False,
        )
        train_runtime_start = timed_start()
        for data, _labels, concept_targets in progress:
            data = data.float().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _disease_logits, concept_logits, sparse_loss = model(data)
            concept_loss = compute_concept_loss(
                model,
                concept_logits,
                concept_targets,
            )
            total_loss = (
                baseline.FIXED_RECIPE["lambda_cpt"] * concept_loss
                + sparse_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite Stage 1 loss at epoch {epoch}"
                )
            total_loss.backward()
            if any(
                parameter.grad is not None
                for parameter in model.cls_head.parameters()
            ):
                raise RuntimeError("Stage 1 classifier received a gradient")
            optimizer.step()
            batches += 1
            sums["concept_loss"] += float(concept_loss.item())
            sums["sparse_loss"] += float(sparse_loss.item())
            sums["total_loss"] += float(total_loss.item())
            progress.set_postfix(concept=f"{concept_loss.item():.4f}")
        if batches == 0:
            raise RuntimeError("Stage 1 dataloader produced no batches")

        train_runtime_seconds = timed_elapsed(train_runtime_start)
        validation_runtime_start = timed_start()
        val_metrics = evaluate_concepts(model, val_loader)
        validation_runtime_seconds = timed_elapsed(validation_runtime_start)
        row = {
            "epoch": epoch,
            "train_concept_loss_mean": sums["concept_loss"] / batches,
            "train_sparse_loss": sums["sparse_loss"] / batches,
            "train_total_loss": sums["total_loss"] / batches,
            "val_concept_loss": val_metrics["concept_loss"],
            "val_sparse_loss": val_metrics["sparse_loss"],
            "val_concept_accuracy": val_metrics["concept_accuracy"],
            "val_concept_mean_attribute_accuracy": val_metrics[
                "concept_mean_attribute_accuracy"
            ],
            "val_concept_mean_attribute_macro_f1": val_metrics[
                "concept_mean_attribute_macro_f1"
            ],
            "train_runtime_seconds": train_runtime_seconds,
            "validation_runtime_seconds": validation_runtime_seconds,
            "epoch_runtime_seconds": timed_elapsed(epoch_runtime_start),
            **{
                f"lr_{group['name']}": float(group["lr"])
                for group in optimizer.param_groups
            },
        }
        history.append(row)
        score = val_metrics["concept_mean_attribute_macro_f1"]
        print(
            f"Stage 1 epoch {epoch}: Concept-ACC="
            f"{val_metrics['concept_accuracy']:.2f}% | Mean-Attribute-Macro-F1="
            f"{score:.2f}% | Runtime={format_runtime(row['epoch_runtime_seconds'])}"
        )
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        if score >= best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    if best_epoch is None:
        raise RuntimeError("Stage 1 did not retain a checkpoint")
    if classifier_hash_before != state_subset_sha256(model, include_classifier=True):
        raise RuntimeError("Frozen Stage 1 classifier parameters changed")
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)
    )
    model.cuda()
    final_evaluation_runtime_start = timed_start()
    best_metrics = evaluate_concepts(model, val_loader)
    final_evaluation_runtime_seconds = timed_elapsed(
        final_evaluation_runtime_start
    )
    stage_runtime_seconds = timed_elapsed(stage_runtime_start)
    baseline.write_csv(stage_dir / "history.csv", history)
    stage_metrics = {
        "stage": "concept_pretraining",
        "label_source": "baseline.CONCEPT_LABEL_MAP",
        "label_encoding": "original zero-based state per Attribute",
        "positive_result_used": False,
        "best_epoch": best_epoch,
        "best_concept_mean_attribute_macro_f1": best_score,
        "metrics_at_best_checkpoint": best_metrics,
        "checkpoint": str(best_path),
        "checkpoint_sha256": baseline.sha256_file(best_path),
        "optimizer": optimizer_summary,
        "classifier_frozen_verified": True,
        "classifier_state_sha256": classifier_hash_before,
        "runtime": {
            "stage_total_seconds": stage_runtime_seconds,
            "stage_total_hhmmss": format_runtime(stage_runtime_seconds),
            "final_evaluation_seconds": final_evaluation_runtime_seconds,
            "epoch_runtime_seconds": [
                row["epoch_runtime_seconds"] for row in history
            ],
        },
    }
    (stage_dir / "metrics.json").write_text(
        json.dumps(stage_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    writer.close()
    return stage_metrics


def train_stage2(
    model,
    args,
    fold_frame,
    manifest_metadata,
    fold_audit,
    reference,
    recipe,
    output_dir: Path,
    stage1_metrics: dict,
):
    stage_runtime_start = timed_start()
    stage_dir = output_dir / "stage2_classifier"
    stage_dir.mkdir(parents=True, exist_ok=False)
    writer = baseline.SummaryWriter(
        str(Path(args.tensorboard_dir).resolve() / "stage2_classifier")
    )
    train_loader, val_loader, test_loader = baseline.make_dataloaders(
        model,
        args,
        fold_frame,
    )
    set_stage2_trainable(model)
    concept_hash_before = state_subset_sha256(model, include_classifier=False)
    optimizer, optimizer_summary = build_classifier_optimizer(model)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            baseline.CLASS_WEIGHTS,
            dtype=torch.float32,
        ).cuda()
    )
    best_ranks = {}
    checkpoint_records = {}
    history = []
    reconstruction_verified = False

    for epoch_index in range(baseline.FIXED_RECIPE["epochs"]):
        epoch_runtime_start = timed_start()
        epoch = epoch_index + 1
        model.eval()
        model.cls_head.train()
        baseline.schedule_group_lrs(optimizer, epoch_index)
        losses = []
        progress = baseline.tqdm(
            train_loader,
            desc=f"V5-4 baseline-concept stage2 epoch {epoch}/100",
            leave=False,
        )
        train_runtime_start = timed_start()
        for data, labels, _concept_targets in progress:
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, forward_logits, _features = stage2_forward(model, data)
            if not reconstruction_verified:
                if not torch.allclose(
                    logits.detach(),
                    forward_logits,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        "Reconstructed Raw34 classifier input changed logits"
                    )
                reconstruction_verified = True
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Stage 2 loss at epoch {epoch}"
                )
            loss.backward()
            leaked = [
                name
                for name, parameter in model.named_parameters()
                if not name.startswith("cls_head.") and parameter.grad is not None
            ]
            if leaked:
                raise RuntimeError(
                    f"Stage 2 gradient leaked into concept branch: {leaked}"
                )
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(cls=f"{loss.item():.4f}")
        if not losses:
            raise RuntimeError("Stage 2 dataloader produced no batches")

        train_runtime_seconds = timed_elapsed(train_runtime_start)
        validation_runtime_start = timed_start()
        val_metrics, _, _ = evaluate_disease_stage2(
            model,
            val_loader,
            criterion,
        )
        validation_runtime_seconds = timed_elapsed(validation_runtime_start)
        row = {
            "epoch": epoch,
            "train_cls_loss": float(np.mean(losses)),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_bmac": val_metrics["bmac"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_tradeoff": checkpointing.tradeoff_score(val_metrics),
            "lr_classifier": float(optimizer.param_groups[0]["lr"]),
            "train_runtime_seconds": train_runtime_seconds,
            "validation_runtime_seconds": validation_runtime_seconds,
            "epoch_runtime_seconds": timed_elapsed(epoch_runtime_start),
        }
        history.append(row)
        print(
            f"Stage 2 epoch {epoch}: {baseline.format_metrics(val_metrics)} | "
            f"Tradeoff={row['val_tradeoff']:.2f} | "
            f"Runtime={format_runtime(row['epoch_runtime_seconds'])}"
        )
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        checkpointing.update_retained_checkpoints(
            model,
            val_metrics,
            epoch,
            best_ranks,
            checkpoint_records,
            stage_dir,
        )

    checkpointing.write_checkpoint_index(checkpoint_records, stage_dir)
    best_record = checkpoint_records["best_bmac"]
    best_path = stage_dir / CHECKPOINT_ALIASES["best_bmac"]
    model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)
    )
    model.cuda()
    concept_hash_after = state_subset_sha256(model, include_classifier=False)
    if concept_hash_after != concept_hash_before:
        raise RuntimeError("Stage 2 changed frozen visual/concept parameters")

    final_evaluation_runtime_start = timed_start()
    best_val_metrics, val_true, val_pred = evaluate_disease_stage2(
        model,
        val_loader,
        criterion,
    )
    test_metrics, y_true, y_pred = evaluate_disease_stage2(
        model,
        test_loader,
        criterion,
    )
    if not np.array_equal(val_true, y_true) or not np.array_equal(val_pred, y_pred):
        raise RuntimeError("Shared validation/test predictions unexpectedly differ")
    final_evaluation_runtime_seconds = timed_elapsed(
        final_evaluation_runtime_start
    )
    detailed_metrics = detailed_disease_metrics(y_true, y_pred)
    stage_runtime_seconds = timed_elapsed(stage_runtime_start)
    result, class_report = baseline.save_artifacts(
        stage_dir,
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
        test_loader.dataset.df,
        history,
    )
    metrics_path = stage_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "variant": recipe["variant"],
            "protocol": PROTOCOL,
            "training_mode": recipe["training_mode"],
            "hyperparameters": recipe,
            "model_source": str(model_source_path()),
            "model_source_sha256": baseline.sha256_file(model_source_path()),
            "trainer_source": str(Path(__file__).resolve()),
            "trainer_sha256": baseline.sha256_file(__file__),
            "concept_supervision": {
                "source": "baseline.CONCEPT_LABEL_MAP",
                "encoding": "original zero-based state per Attribute",
                "positive_result_used": False,
                "hard_result_used": False,
                "nothing_class_used": False,
            },
            "stage1": stage1_metrics,
            "stage2": {
                "classifier_input": "unchanged 34 raw MCSAF activations",
                "optimizer": optimizer_summary,
                "concept_branch_frozen_verified": True,
                "concept_branch_sha256_before": concept_hash_before,
                "concept_branch_sha256_after": concept_hash_after,
                "raw34_reconstruction_verified": reconstruction_verified,
                "checkpoint_index": checkpoint_records,
                "runtime": {
                    "stage_total_seconds": stage_runtime_seconds,
                    "stage_total_hhmmss": format_runtime(stage_runtime_seconds),
                    "final_validation_and_test_seconds": (
                        final_evaluation_runtime_seconds
                    ),
                    "epoch_runtime_seconds": [
                        row["epoch_runtime_seconds"] for row in history
                    ],
                },
            },
            "detailed_disease_metrics": detailed_metrics,
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    baseline.write_csv(
        stage_dir / "overall_metrics.csv",
        [detailed_metrics["overall"]],
    )
    baseline.write_csv(
        stage_dir / "per_class_metrics.csv",
        [
            {"class": class_name, **detailed_metrics["per_class"][class_name]}
            for class_name in baseline.CLASS_NAMES
        ],
    )
    writer.close()
    print(f"Sequential final: {baseline.format_metrics(result)}")
    print_detailed_disease_metrics(detailed_metrics)
    print(
        "\nRuntime | Stage 1="
        f"{stage1_metrics['runtime']['stage_total_hhmmss']} | Stage 2="
        f"{format_runtime(stage_runtime_seconds)}"
    )
    print(class_report)
    return metrics


def write_comparison(output_dir: Path, sequential_metrics: dict) -> None:
    payload = {
        "comparison": "V5-4 original-label joint vs sequential Raw34",
        "sequential_metrics": {
            key: sequential_metrics[key]
            for key in ("acc", "bmac", "macro_f1", "weighted_f1")
        },
        "joint_reference_path": str(DEFAULT_JOINT_REFERENCE),
        "joint_reference_available": DEFAULT_JOINT_REFERENCE.is_file(),
    }
    if DEFAULT_JOINT_REFERENCE.is_file():
        joint = json.loads(DEFAULT_JOINT_REFERENCE.read_text(encoding="utf-8"))
        payload["joint_metrics"] = {
            key: joint[key]
            for key in ("acc", "bmac", "macro_f1", "weighted_f1")
        }
        payload["sequential_minus_joint"] = {
            key: sequential_metrics[key] - joint[key]
            for key in ("acc", "bmac", "macro_f1", "weighted_f1")
        }
        payload["joint_reference_sha256"] = baseline.sha256_file(
            DEFAULT_JOINT_REFERENCE
        )
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    run_started_at = datetime.now(timezone.utc)
    run_runtime_start = timed_start()
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = model_source_path()
    source_hash = baseline.sha256_file(source)
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            f"V5-4 model SHA-256 mismatch: expected "
            f"{args.expected_model_sha256}, found {source_hash}"
        )
    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(),
        args.fold,
    )
    reference = common_trainer.load_baseline_reference(
        Path(args.baseline_reference_metrics).resolve(),
        args.fold,
        fold_audit,
    )
    recipe = sequential_recipe(args)
    recipe_differences = validate_training_recipe(recipe)
    output_dir = Path(args.output_dir).resolve()
    if not (args.validate_config_only or args.smoke_test) and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    static_audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "fold": args.fold,
        "variant": recipe["variant"],
        "training_mode": recipe["training_mode"],
        "model_source": str(source),
        "model_source_sha256": source_hash,
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": baseline.sha256_file(__file__),
        "concept_target_source": "baseline.CONCEPT_LABEL_MAP",
        "concept_label_map": baseline.CONCEPT_LABEL_MAP,
        "positive_result_used": False,
        "pseudo_label_csv_read": False,
        "recipe": recipe,
        "training_recipe_difference_vs_origin": recipe_differences,
        "manifest_sha256": fold_audit["manifest_sha256"],
        "train_size": fold_audit["train_size"],
        "holdout_size": fold_audit["test_size"],
        "val_test_overlap": fold_audit["val_test_overlap"],
        "joint_reference": str(DEFAULT_JOINT_REFERENCE),
        "joint_reference_available": DEFAULT_JOINT_REFERENCE.is_file(),
        "planned_output": str(output_dir),
        "planned_output_exists": output_dir.exists(),
    }
    print(json.dumps(static_audit, indent=2, ensure_ascii=False))
    if args.validate_config_only:
        print("Configuration validation completed; no training/output created.")
        return

    baseline.FIXED_RECIPE = copy.deepcopy(recipe)
    args.historical_metrics = args.baseline_reference_metrics
    model = v5_4_trainer.construct_model(args)
    if args.smoke_test:
        run_smoke_test(model, args, fold_frame)
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(static_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stage1_metrics = train_stage1(model, args, fold_frame, output_dir)
    sequential_metrics = train_stage2(
        model,
        args,
        fold_frame,
        manifest_metadata,
        fold_audit,
        reference,
        recipe,
        output_dir,
        stage1_metrics,
    )
    write_comparison(output_dir, sequential_metrics)
    total_runtime_seconds = timed_elapsed(run_runtime_start)
    runtime_payload = {
        "started_at_utc": run_started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage1": stage1_metrics["runtime"],
        "stage2": sequential_metrics["stage2"]["runtime"],
        "end_to_end_seconds": total_runtime_seconds,
        "end_to_end_hhmmss": format_runtime(total_runtime_seconds),
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"End-to-end runtime: {format_runtime(total_runtime_seconds)}")
    print("Completed V5-4 baseline-concept sequential Fold04 outputs.")


if __name__ == "__main__":
    main()
