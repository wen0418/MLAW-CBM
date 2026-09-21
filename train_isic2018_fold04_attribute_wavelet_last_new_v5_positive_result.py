#!/usr/bin/env python3
"""Joint-train New-V5 on Fold04 with per-image positive-result labels.

The audited New-V5 Fold04 data split, optimization recipe, disease path, and
checkpoint policies are preserved.  The sole experimental change is concept
supervision: each of the seven Attribute heads receives a ``nothing`` class at
index 0 and is trained against the per-image ``positive_result`` CSV column.

Validation and test intentionally remain the same Fold04 holdout so results
are directly comparable with the existing diagnostic New-V5 run.  They are
therefore optimistic and are not an independent test estimate.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import train_isic2018_baseline_cv10 as baseline
import train_isic2018_fold04_attribute_wavelet_last_new_v5 as parent_trainer
from training.common import checkpointing
from training.common import isic2018 as common_trainer
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_ROUTE_TEMPERATURE,
    DEFAULT_TOP_K,
)
from model.mvpcbm_attribute_wavelet_last_new_v5_positive_result import (
    NOTHING_LABEL,
    NOTHING_STATE_NAME,
    mvpcbm as PositiveResultNewV5,
)


PROTOCOL = (
    "baseline_cv10_attribute_wavelet_last_new_v5_positive_result_"
    "joint_shared_val_test_diagnostic"
)
MODEL_FILENAME = "mvpcbm_attribute_wavelet_last_new_v5_positive_result.py"
PROJECT_ROOT = Path(__file__).resolve().parent
ORIGIN_ROOT = PROJECT_ROOT
DEFAULT_MANIFEST_DIR = (
    ORIGIN_ROOT / "splits" / "isic2018_baseline_cv10_imagelevel"
)
DEFAULT_REFERENCE = (
    ORIGIN_ROOT / "reference_results" / "isic2018_fold04_baseline_lam2p5.json"
)
DEFAULT_PSEUDO_LABEL_CSV = (
    PROJECT_ROOT / "dataset" / "pseudo_labels" / "training.csv"
)
PSEUDO_LABEL_ENCODING = "cp1252"
SUPPORTED_LABEL_COLUMNS = ("positive_result", "hard_result")
CHECKPOINT_ALIASES = checkpointing.CHECKPOINT_ALIASES


@dataclass(frozen=True)
class PseudoLabelRecord:
    diagnosis: str
    original_zero_based: tuple[int, ...]
    target: tuple[int, ...]


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
            "baseline_cv10_attribute_wavelet_last_new_v5_positive_result"
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
    imported = Path(inspect.getfile(PositiveResultNewV5)).resolve()
    expected = (PROJECT_ROOT / "model" / MODEL_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            "Unexpected positive-result New-V5 import: "
            f"{imported}; expected {expected}"
        )
    return imported


def parent_model_source_path() -> Path:
    return PROJECT_ROOT / "model/mvpcbm_attribute_wavelet_last_new_v5.py"


def _parse_seven_ints(value, field: str, image_id: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Invalid {field} for {image_id}: {value!r}"
        ) from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(
        baseline.CONCEPTS
    ):
        raise ValueError(
            f"{field} for {image_id} must contain seven integers, "
            f"received {parsed!r}"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError(f"{field} for {image_id} contains a non-integer")
    return tuple(int(item) for item in parsed)


def load_pseudo_labels(
    path: Path,
    label_column: str,
) -> tuple[dict[str, PseudoLabelRecord], dict]:
    """Load and exhaustively audit the disease-conditioned per-image labels."""

    path = path.resolve()
    if label_column not in SUPPORTED_LABEL_COLUMNS:
        raise ValueError(f"Unsupported concept label column: {label_column}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing pseudo-label CSV: {path}")
    frame = pd.read_csv(path, encoding=PSEUDO_LABEL_ENCODING)
    required_columns = {
        "image_id",
        "diagnosis",
        "original_zero_based_list",
        "shifted_input_list",
        "hard_result",
        "positive_result",
        "num_strong",
        "num_weak_visible",
        "num_not_visibly_supported",
        "num_inclusive",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Pseudo-label CSV lacks required columns: {sorted(missing)}"
        )
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise RuntimeError("Pseudo-label image_id values must be non-null and unique")

    expected_original = {
        class_name: tuple(baseline.CONCEPT_LABEL_MAP[class_index])
        for class_index, class_name in enumerate(baseline.CLASS_NAMES)
    }
    head_sizes = tuple(len(states) + 1 for states in baseline.CONCEPTS.values())
    target_lookup: dict[str, PseudoLabelRecord] = {}
    label_counts = [Counter() for _ in baseline.CONCEPTS]
    class_counts = Counter()
    all_nothing = 0
    strong_total = 0
    weak_total = 0
    inclusive_total = 0

    for row_index, row in frame.iterrows():
        image_id = str(row["image_id"])
        diagnosis = str(row["diagnosis"])
        if diagnosis not in expected_original:
            raise RuntimeError(
                f"Unknown diagnosis {diagnosis!r} for {image_id} at row {row_index}"
            )
        original = _parse_seven_ints(
            row["original_zero_based_list"],
            "original_zero_based_list",
            image_id,
        )
        shifted = _parse_seven_ints(
            row["shifted_input_list"],
            "shifted_input_list",
            image_id,
        )
        hard = _parse_seven_ints(row["hard_result"], "hard_result", image_id)
        positive = _parse_seven_ints(
            row["positive_result"],
            "positive_result",
            image_id,
        )
        if original != expected_original[diagnosis]:
            raise RuntimeError(
                f"Original concept labels disagree with diagnosis for {image_id}: "
                f"{original} != {expected_original[diagnosis]}"
            )
        if shifted != tuple(value + 1 for value in original):
            raise RuntimeError(f"Shifted concept labels are invalid for {image_id}")
        for attribute_index, expected_value in enumerate(shifted):
            if hard[attribute_index] not in (NOTHING_LABEL, expected_value):
                raise RuntimeError(f"Invalid hard_result for {image_id}")
            if positive[attribute_index] not in (NOTHING_LABEL, expected_value):
                raise RuntimeError(f"Invalid positive_result for {image_id}")
            if hard[attribute_index] and (
                hard[attribute_index] != positive[attribute_index]
            ):
                raise RuntimeError(
                    f"hard_result is not a subset of positive_result for {image_id}"
                )
        num_strong = sum(value != NOTHING_LABEL for value in hard)
        num_inclusive = sum(value != NOTHING_LABEL for value in positive)
        num_weak = num_inclusive - num_strong
        if num_strong != int(row["num_strong"]):
            raise RuntimeError(f"num_strong mismatch for {image_id}")
        if num_weak != int(row["num_weak_visible"]):
            raise RuntimeError(f"num_weak_visible mismatch for {image_id}")
        if num_inclusive != int(row["num_inclusive"]):
            raise RuntimeError(f"num_inclusive mismatch for {image_id}")
        if len(baseline.CONCEPTS) - num_inclusive != int(
            row["num_not_visibly_supported"]
        ):
            raise RuntimeError(
                f"num_not_visibly_supported mismatch for {image_id}"
            )

        selected = positive if label_column == "positive_result" else hard
        for attribute_index, target in enumerate(selected):
            if not 0 <= target < head_sizes[attribute_index]:
                raise RuntimeError(
                    f"Target {target} is outside head {attribute_index} range "
                    f"[0, {head_sizes[attribute_index]}) for {image_id}"
                )
            label_counts[attribute_index][target] += 1
        if all(value == NOTHING_LABEL for value in selected):
            all_nothing += 1
        target_lookup[image_id] = PseudoLabelRecord(
            diagnosis=diagnosis,
            original_zero_based=original,
            target=selected,
        )
        class_counts[diagnosis] += 1
        strong_total += num_strong
        weak_total += num_weak
        inclusive_total += num_inclusive

    if len(target_lookup) != len(frame):
        raise RuntimeError("Pseudo-label lookup lost rows")
    audit = {
        "path": str(path),
        "sha256": baseline.sha256_file(path),
        "encoding": PSEUDO_LABEL_ENCODING,
        "row_count": len(frame),
        "unique_image_ids": len(target_lookup),
        "label_column": label_column,
        "attribute_order": list(baseline.CONCEPTS),
        "original_state_counts": {
            name: len(states) for name, states in baseline.CONCEPTS.items()
        },
        "positive_result_head_sizes": dict(
            zip(baseline.CONCEPTS, head_sizes)
        ),
        "nothing_label": NOTHING_LABEL,
        "nothing_state_name": NOTHING_STATE_NAME,
        "label_encoding": "0=nothing; nonzero=original_zero_based_label+1",
        "class_counts": dict(class_counts),
        "target_counts": {
            name: {
                str(label): int(count)
                for label, count in sorted(label_counts[index].items())
            }
            for index, name in enumerate(baseline.CONCEPTS)
        },
        "all_nothing_rows": all_nothing,
        "strong_positive_total": strong_total,
        "weak_positive_total": weak_total,
        "inclusive_positive_total": inclusive_total,
        "all_rows_passed_invariants": True,
    }
    return target_lookup, audit


def attach_fold_labels(
    fold_frame: pd.DataFrame,
    target_lookup: dict[str, PseudoLabelRecord],
) -> tuple[pd.DataFrame, dict]:
    """Join pseudo-label targets onto the exact audited Fold manifest."""

    frame = fold_frame.reset_index(drop=True).copy()
    manifest_ids = set(frame["image"].astype(str))
    label_ids = set(target_lookup)
    if manifest_ids != label_ids:
        raise RuntimeError(
            "Pseudo-label and Fold manifest image IDs differ: "
            f"missing_labels={len(manifest_ids - label_ids)}, "
            f"extra_labels={len(label_ids - manifest_ids)}"
        )
    frame["concept_target"] = [
        target_lookup[str(image_id)].target for image_id in frame["image"]
    ]
    for row in frame.itertuples():
        record = target_lookup[str(row.image)]
        if record.diagnosis != str(row.label_name):
            raise RuntimeError(
                f"Pseudo-label diagnosis differs from Fold manifest for "
                f"{row.image}: {record.diagnosis} != {row.label_name}"
            )
        expected_original = tuple(baseline.CONCEPT_LABEL_MAP[int(row.label_id)])
        if record.original_zero_based != expected_original:
            raise RuntimeError(
                f"Pseudo-label original concepts differ from Fold manifest for "
                f"{row.image}: {record.original_zero_based} != {expected_original}"
            )
        target = tuple(row.concept_target)
        for attribute_index, value in enumerate(target):
            expected_shifted = expected_original[attribute_index] + 1
            if value not in (NOTHING_LABEL, expected_shifted):
                raise RuntimeError(
                    f"Fold target no longer matches class-derived candidate for "
                    f"{row.image}: target={target}, original={expected_original}"
                )
    role_counts = Counter(frame["role"])
    target_counts_by_role = {}
    for role in ("train", "holdout"):
        subset = frame[frame["role"] == role]
        target_counts_by_role[role] = {
            name: dict(
                sorted(
                    Counter(
                        int(target[attribute_index])
                        for target in subset["concept_target"]
                    ).items()
                )
            )
            for attribute_index, name in enumerate(baseline.CONCEPTS)
        }
    return frame, {
        "manifest_rows": len(frame),
        "matched_rows": len(frame),
        "missing_labels": 0,
        "extra_labels": 0,
        "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "target_counts_by_role": target_counts_by_role,
        "join_verified": True,
    }


class PositiveResultManifestDataset(Dataset):
    def __init__(self, data_dir: Path, frame: pd.DataFrame, transform):
        self.image_dir = data_dir / "ISIC2018_Task3_Training_Input"
        self.df = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(self.image_dir / f"{row['image']}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            int(row["label_id"]),
            np.asarray(row["concept_target"], dtype=np.int64),
        )


def make_dataloaders(model, args, fold_frame: pd.DataFrame, smoke_test=False):
    train_transform, eval_transform = baseline.make_transforms(model)
    train_frame = fold_frame[fold_frame["role"] == "train"]
    holdout_frame = fold_frame[fold_frame["role"] == "holdout"]
    train_set = PositiveResultManifestDataset(
        Path(args.data_path),
        train_frame,
        train_transform,
    )
    val_set = PositiveResultManifestDataset(
        Path(args.data_path),
        holdout_frame,
        eval_transform,
    )
    test_set = PositiveResultManifestDataset(
        Path(args.data_path),
        holdout_frame,
        copy.deepcopy(eval_transform),
    )
    batch_size = 2 if smoke_test else baseline.FIXED_RECIPE["batch_size"]
    workers_train = 0 if smoke_test else args.train_workers
    workers_eval = 0 if smoke_test else args.eval_workers
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers_train,
        drop_last=baseline.FIXED_RECIPE["drop_last_train"] and not smoke_test,
        worker_init_fn=baseline.seed_worker,
        generator=torch.Generator().manual_seed(
            baseline.FIXED_RECIPE["dataloader_generator_seeds"]["train"]
        ),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers_eval,
        drop_last=False,
        worker_init_fn=baseline.seed_worker,
        generator=torch.Generator().manual_seed(
            baseline.FIXED_RECIPE["dataloader_generator_seeds"]["val"]
        ),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers_eval,
        drop_last=False,
        worker_init_fn=baseline.seed_worker,
        generator=torch.Generator().manual_seed(
            baseline.FIXED_RECIPE["dataloader_generator_seeds"]["test"]
        ),
    )
    return train_loader, val_loader, test_loader


def positive_result_recipe(args) -> dict:
    recipe = parent_trainer.new_v5_recipe(args)
    recipe.update(
        {
            "variant": "refined_mvpcbm_attribute_wavelet_last_new_v5_positive_result",
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
                "unchanged original 34-dimensional MCSAF activation"
            ),
            "nothing_logits_used_by_disease_classifier": False,
            "joint_total_loss": (
                "class-weighted disease CE + 2.5 * mean seven-head concept CE "
                "+ inherited MCSAF sparse loss"
            ),
            "classifier_gradient_reaches_concept_branch": True,
            "sequential_training_used": False,
            "variant_model_source": f"model/{MODEL_FILENAME}",
            "parent_new_v5_model_source": (
                "model/mvpcbm_attribute_wavelet_last_new_v5.py"
            ),
        }
    )
    return recipe


def validate_training_recipe(recipe: dict) -> dict:
    differences = {
        key: {
            "baseline": baseline.FIXED_RECIPE.get(key),
            "positive_result": recipe.get(key),
        }
        for key in common_trainer.TRAINING_RECIPE_KEYS
        if recipe.get(key) != baseline.FIXED_RECIPE.get(key)
    }
    if set(differences) - {"checkpoint_selection"}:
        raise RuntimeError(
            "Positive-result New-V5 changed unauthorized Fold04 settings: "
            f"{differences}"
        )
    return differences


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Positive-result New-V5 training requires a CUDA-capable PyTorch setup"
        )
    source_hash = baseline.sha256_file(model_source_path())
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            "Positive-result New-V5 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )
    args.dataset = "isic2018"
    args.num_class = len(baseline.CLASS_NAMES)
    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    model = PositiveResultNewV5(
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


def compute_losses(model, data, labels, concept_targets, criterion):
    logits, concept_logits, sparse_loss = model(data)
    expected_order = tuple(model.positive_result_attribute_order)
    if tuple(concept_logits) != expected_order:
        raise RuntimeError("Runtime positive-result head order changed")
    concept_loss_sum = torch.zeros((), device=data.device)
    for attribute_index, key in enumerate(expected_order):
        head = concept_logits[key]
        target = concept_targets[:, attribute_index]
        if head.size(1) != model.positive_result_head_sizes[attribute_index]:
            raise RuntimeError(
                f"Unexpected {key} head size: {head.size(1)} != "
                f"{model.positive_result_head_sizes[attribute_index]}"
            )
        if target.numel() and (
            int(target.min()) < 0 or int(target.max()) >= head.size(1)
        ):
            raise RuntimeError(
                f"{key} target lies outside [0, {head.size(1)})"
            )
        concept_loss_sum = concept_loss_sum + F.cross_entropy(head, target)
    concept_loss = concept_loss_sum / len(expected_order)
    classification_loss = criterion(logits, labels)
    checkpoint_loss = classification_loss + sparse_loss
    total_loss = (
        classification_loss
        + baseline.FIXED_RECIPE["lambda_cpt"] * concept_loss
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


def evaluate_joint(model, dataloader, criterion):
    model.eval()
    totals = Counter()
    sample_count = 0
    disease_true = []
    disease_pred = []
    concept_true = {key: [] for key in baseline.CONCEPTS}
    concept_pred = {key: [] for key in baseline.CONCEPTS}
    with torch.no_grad():
        for data, labels, concept_targets in baseline.tqdm(
            dataloader,
            desc="Evaluate",
            leave=False,
        ):
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            losses = compute_losses(
                model,
                data,
                labels,
                concept_targets,
                criterion,
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
            disease_true.extend(labels.cpu().tolist())
            disease_pred.extend(losses["logits"].argmax(dim=1).cpu().tolist())
            for attribute_index, key in enumerate(baseline.CONCEPTS):
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
    metrics = baseline.calculate_metrics(truth, prediction)
    metrics.update(
        {
            "loss": totals["checkpoint_loss"] / sample_count,
            "classification_loss": totals["classification_loss"] / sample_count,
            "concept_loss": totals["concept_loss"] / sample_count,
            "sparse_loss": totals["sparse_loss"] / sample_count,
            "joint_total_loss": totals["total_loss"] / sample_count,
        }
    )
    attribute_metrics = {}
    attribute_accuracies = []
    attribute_macro_f1s = []
    all_concept_true = []
    all_concept_pred = []
    for attribute_index, key in enumerate(baseline.CONCEPTS):
        target = np.asarray(concept_true[key], dtype=np.int64)
        pred = np.asarray(concept_pred[key], dtype=np.int64)
        labels_for_head = list(
            range(len(baseline.CONCEPTS[key]) + 1)
        )
        accuracy = 100.0 * baseline.accuracy_score(target, pred)
        macro_f1 = 100.0 * baseline.f1_score(
            target,
            pred,
            labels=labels_for_head,
            average="macro",
            zero_division=0,
        )
        attribute_metrics[key] = {
            "accuracy": float(accuracy),
            "macro_f1_fixed_labels": float(macro_f1),
            "labels": labels_for_head,
            "target_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(target.tolist()).items())
            },
            "prediction_counts": {
                str(label): int(count)
                for label, count in sorted(Counter(pred.tolist()).items())
            },
        }
        attribute_accuracies.append(accuracy)
        attribute_macro_f1s.append(macro_f1)
        all_concept_true.extend(target.tolist())
        all_concept_pred.extend(pred.tolist())
    metrics["concept_accuracy"] = float(
        100.0 * baseline.accuracy_score(all_concept_true, all_concept_pred)
    )
    metrics["concept_mean_attribute_accuracy"] = float(
        np.mean(attribute_accuracies)
    )
    metrics["concept_mean_attribute_macro_f1"] = float(
        np.mean(attribute_macro_f1s)
    )
    metrics["concept_per_attribute"] = attribute_metrics
    return metrics, truth, prediction


def run_smoke_test(model, args, fold_frame) -> None:
    train_loader, _, _ = make_dataloaders(
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
    losses = compute_losses(
        model,
        data,
        labels,
        concept_targets,
        criterion,
    )
    if not torch.isfinite(losses["total_loss"]):
        raise FloatingPointError("Positive-result smoke-test loss is non-finite")
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
            "Positive-result joint gradient failure: "
            f"missing={missing}, nonfinite={nonfinite}"
        )
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "model_variant": (
                    "refined_mvpcbm_attribute_wavelet_last_new_v5_positive_result"
                ),
                "training_mode": "joint_from_scratch_mvp_cbm",
                "batch_size": len(labels),
                "concept_label_column": args.concept_label_column,
                "concept_targets": concept_targets.cpu().tolist(),
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
                "nothing_logits_used_by_disease_classifier": False,
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
    recipe,
    pseudo_label_audit,
    label_join_audit,
) -> tuple[dict, dict]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = baseline.SummaryWriter(
        str(Path(args.tensorboard_dir).resolve() / f"fold_{args.fold:02d}")
    )
    report_path = output_dir / "training_report.txt"
    train_loader, val_loader, test_loader = make_dataloaders(
        model,
        args,
        fold_frame,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            baseline.CLASS_WEIGHTS,
            dtype=torch.float32,
        ).cuda()
    )
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "warning": "val=test; checkpoint-selected scores are optimistic",
                    "fold": args.fold,
                    "recipe": recipe,
                    "pseudo_label_audit": pseudo_label_audit,
                    "label_join_audit": label_join_audit,
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
    display_name = getattr(
        args,
        "positive_result_display_name",
        "positive-result New V5",
    )
    for epoch_index in range(baseline.FIXED_RECIPE["epochs"]):
        epoch = epoch_index + 1
        model.train()
        baseline.schedule_group_lrs(optimizer, epoch_index)
        sums = Counter()
        batches = 0
        progress = baseline.tqdm(
            train_loader,
            desc=f"{display_name} fold {args.fold:02d} epoch {epoch}/100",
            leave=False,
        )
        for data, labels, concept_targets in progress:
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(
                model,
                data,
                labels,
                concept_targets,
                criterion,
            )
            if not torch.isfinite(losses["total_loss"]):
                raise FloatingPointError(
                    f"Non-finite {display_name} loss at "
                    f"fold {args.fold}, epoch {epoch}"
                )
            losses["total_loss"].backward()
            optimizer.step()
            batches += 1
            for key in (
                "classification_loss",
                "concept_loss",
                "sparse_loss",
                "total_loss",
            ):
                sums[key] += float(losses[key].item())
            progress.set_postfix(
                cls=f"{losses['classification_loss'].item():.4f}",
                concept=f"{losses['concept_loss'].item():.4f}",
            )
        if batches == 0:
            raise RuntimeError("Training dataloader produced no batches")

        val_metrics, _, _ = evaluate_joint(model, val_loader, criterion)
        row = {
            "epoch": epoch,
            "train_cls_loss": sums["classification_loss"] / batches,
            "train_concept_loss_mean": sums["concept_loss"] / batches,
            "train_sparse_loss": sums["sparse_loss"] / batches,
            "train_total_loss": sums["total_loss"] / batches,
            "val_loss": val_metrics["loss"],
            "val_cls_loss": val_metrics["classification_loss"],
            "val_concept_loss": val_metrics["concept_loss"],
            "val_sparse_loss": val_metrics["sparse_loss"],
            "val_joint_total_loss": val_metrics["joint_total_loss"],
            "val_acc": val_metrics["acc"],
            "val_bmac": val_metrics["bmac"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_concept_accuracy": val_metrics["concept_accuracy"],
            "val_concept_mean_attribute_accuracy": val_metrics[
                "concept_mean_attribute_accuracy"
            ],
            "val_concept_mean_attribute_macro_f1": val_metrics[
                "concept_mean_attribute_macro_f1"
            ],
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
            f"Concept-ACC={val_metrics['concept_accuracy']:.2f}% | "
            "Concept-Mean-Macro-F1="
            f"{val_metrics['concept_mean_attribute_macro_f1']:.2f}% | "
            f"Tradeoff={row['val_tradeoff']:.2f}"
        )
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    best_val_metrics, val_true, val_pred = evaluate_joint(
        model,
        val_loader,
        criterion,
    )
    test_metrics, y_true, y_pred = evaluate_joint(
        model,
        test_loader,
        criterion,
    )
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
    print(
        "Fold "
        f"{args.fold:02d} final best-BMAC: {baseline.format_metrics(result)}"
    )
    print(class_report)
    writer.close()
    return best_val_metrics, test_metrics


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
                "refined_mvpcbm_attribute_wavelet_last_new_v5_positive_result"
            ),
            "protocol": PROTOCOL,
            "hyperparameters": recipe,
            "model_source": str(variant_source),
            "model_source_sha256": baseline.sha256_file(variant_source),
            "parent_new_v5_model_source": str(parent_source),
            "parent_new_v5_model_source_sha256": baseline.sha256_file(
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
            "Positive-result New-V5 model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )

    fold_frame, manifest_metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(),
        args.fold,
    )
    target_lookup, pseudo_label_audit = load_pseudo_labels(
        Path(args.pseudo_label_csv),
        args.concept_label_column,
    )
    fold_frame, label_join_audit = attach_fold_labels(
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
        "variant_model_source": str(source),
        "variant_model_source_sha256": source_hash,
        "parent_new_v5_model_source": str(parent_model_source_path()),
        "parent_new_v5_model_source_sha256": baseline.sha256_file(
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
            "unchanged original 34-dimensional MCSAF activation"
        ),
        "nothing_logits_used_by_disease_classifier": False,
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
    model = construct_model(args)
    optimizer, optimizer_summary = baseline.build_optimizer(model)
    print(json.dumps({"optimizer_groups": optimizer_summary}, indent=2))
    del optimizer

    if args.smoke_test:
        run_smoke_test(model, args, fold_frame)
        return

    best_val_metrics, test_metrics = train_with_retained_checkpoints(
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
    print("Completed joint positive-result New-V5 Fold04 outputs.")


if __name__ == "__main__":
    main()
