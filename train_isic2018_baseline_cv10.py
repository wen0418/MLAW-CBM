#!/usr/bin/env python3
"""Train exactly one refined MVP-CBM baseline on one audited CV10 fold.

The fold holdout is deliberately used for both validation/checkpoint selection
and final reporting.  These scores are diagnostic and optimistically biased.
This entry point imports no SSRG or wavelet model implementation.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import hashlib
import inspect
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import utils
from model.mvpcbm_refined import mvpcbm as BaselineMVPCBM


CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_WEIGHTS = [1.2855, 0.2134, 2.7835, 4.3753, 1.3018, 12.4410, 10.0755]
CONCEPT_LABEL_MAP = [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [2, 0, 2, 2, 2, 0, 1],
    [3, 0, 0, 3, 3, 0, 2],
    [4, 2, 1, 4, 4, 1, 3],
    [5, 1, 1, 5, 5, 1, 0],
    [6, 3, 1, 6, 1, 2, 0],
]
CONCEPTS = {
    "color": [
        "highly variable, often with multiple colors (black, brown, red, white, blue)",
        "uniformly tan, brown, or black",
        "translucent, pearly white, sometimes with blue, brown, or black areas",
        "red, pink, or brown, often with a scale",
        "light brown to black",
        "pink brown or red",
        "red, purple, or blue",
    ],
    "shape": ["irregular", "round", "round to irregular", "variable"],
    "border": [
        "often blurry and irregular",
        "sharp and well-defined",
        "rolled edges, often indistinct",
    ],
    "dermoscopic patterns": [
        "atypical pigment network, irregular streaks, blue-whitish veil, irregular",
        "regular pigment network, symmetric dots and globules",
        "arborizing vessels, leaf-like areas, blue-gray avoid nests",
        "strawberry pattern, glomerular vessels, scale",
        "cerebriform pattern, milia-like cysts, comedo-like openings",
        "central white patch, peripheral pigment network",
        "depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs",
    ],
    "texture": [
        "a raised or ulcerated surface",
        "smooth",
        "smooth, possibly with telangiectasias",
        "rough, scaly",
        "warty or greasy surface",
        "firm, may dimple when pinched",
    ],
    "symmetry": [
        "asymmetrical",
        "symmetrical",
        "can be symmetrical or asymmetrical depending on type",
    ],
    "elevation": [
        "flat to raised",
        "raised with possible central ulceration",
        "slightly raised",
        "slightly raised maybe thick",
    ],
}

FIXED_RECIPE = {
    "variant": "baseline",
    "epochs": 100,
    "batch_size": 64,
    "warmup_epoch": 5,
    "optimizer": "AdamW",
    "backbone_lr": 1e-5,
    "bridge_lr": 1e-4,
    "weight_decay": 0.01,
    "lambda_cpt": 2.5,
    "model_seed": 43,
    "checkpoint_selection": "maximum validation BMAC; later epoch wins ties",
    "post_model_seed_reset": True,
    "lr_schedule": "historical exponential warmup/poly decay per group",
    "drop_last_train": True,
    "model_name": "biomedclip",
    "visual_trunk_initialization": "timm vit_base_patch16_224 pretrained weights",
    "criterion": "class-weighted cross entropy",
    "class_weights": CLASS_WEIGHTS,
    "train_augmentation": {
        "random_resized_crop": {
            "size": [224, 224],
            "scale": [0.75, 1.0],
            "ratio": [0.75, 1.33],
            "interpolation": "bicubic",
        },
        "random_horizontal_flip": True,
        "random_vertical_flip": True,
    },
    "eval_transform": "BiomedCLIP pretrained evaluation preprocessing",
    "dataloader_generator_seeds": {"train": 43, "val": 44, "test": 45},
    "deterministic_algorithms": "warn_only",
    "tf32": False,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--data-path", default="./dataset/ISIC2018")
    parser.add_argument(
        "--manifest-dir", default="./splits/isic2018_baseline_cv10_imagelevel"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", default="./log/isic2018/baseline_cv10")
    parser.add_argument(
        "--historical-metrics",
        default="./output/isic2018/historical_repro_split60_baseline_seed43/metrics.json",
    )
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def model_source_path() -> Path:
    imported = Path(inspect.getfile(BaselineMVPCBM)).resolve()
    expected = (Path(__file__).resolve().parent / "model/mvpcbm_refined.py").resolve()
    if imported != expected:
        raise RuntimeError(f"Unexpected baseline import: {imported}; expected {expected}")
    return imported


def validate_historical_recipe(historical_metrics_path: Path) -> dict:
    if not historical_metrics_path.is_file():
        raise FileNotFoundError(f"Historical reference is missing: {historical_metrics_path}")
    with historical_metrics_path.open(encoding="utf-8") as handle:
        historical = json.load(handle)
    expected = {
        "variant": "baseline",
        "seed": 43,
        "epochs": 100,
        "batch_size": 64,
        "warmup_epoch": 5,
        "bridge_lr": 1e-4,
        "backbone_lr": 1e-5,
        "lambda_cpt": 2.5,
        "protocol": "historical_per_model_reproduction_v1",
        "backbone_lr_multiplier": 0.1,
        "bridge_lr_multiplier": 1.0,
        "post_model_seed_reset": True,
    }
    mismatches = {
        key: {"expected": value, "found": historical.get(key)}
        for key, value in expected.items()
        if historical.get(key) != value
    }
    source = model_source_path()
    if historical.get("model_source_sha256") != sha256_file(source):
        mismatches["model_source_sha256"] = {
            "expected": sha256_file(source),
            "found": historical.get("model_source_sha256"),
        }
    if mismatches:
        raise RuntimeError(f"Historical recipe validation failed: {mismatches}")
    return historical


def load_fold_audit(manifest_dir: Path, fold: int) -> tuple[pd.DataFrame, dict, dict]:
    metadata_path = manifest_dir / "split_metadata.json"
    manifest_path = manifest_dir / f"fold_{fold:02d}.csv"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "CV10 manifests are missing; run prepare_isic2018_baseline_cv10.py first"
        )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("validation", {}).get("each_image_holdout_exactly_once"):
        raise RuntimeError("Manifest metadata does not certify exact holdout coverage")
    fold_audit = metadata["folds"][fold]
    actual_hash = sha256_file(manifest_path)
    if fold_audit["fold"] != fold or fold_audit["manifest_sha256"] != actual_hash:
        raise RuntimeError(f"Fold {fold} manifest hash/audit mismatch")
    frame = pd.read_csv(manifest_path)
    if len(frame) != 10_015 or set(frame["role"]) != {"train", "holdout"}:
        raise RuntimeError(f"Invalid fold manifest: {manifest_path}")
    return frame, metadata, fold_audit


class ManifestDataset(Dataset):
    def __init__(self, data_dir: Path, frame: pd.DataFrame, transform):
        self.image_dir = data_dir / "ISIC2018_Task3_Training_Input"
        self.df = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(self.image_dir / f"{row['image']}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_id"])
        return image, label, np.asarray(CONCEPT_LABEL_MAP[label])


def make_transforms(model):
    train_transform = copy.deepcopy(model.config.preprocess)
    train_transform.transforms.pop(0)
    if model.model_name != "clip":
        train_transform.transforms.pop(0)
    train_transform.transforms.insert(0, transforms.RandomVerticalFlip())
    train_transform.transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transform.transforms.insert(
        0,
        transforms.RandomResizedCrop(
            size=(224, 224),
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33),
            interpolation=utils.get_interpolation_mode("bicubic"),
        ),
    )
    return train_transform, copy.deepcopy(model.config.preprocess)


def make_dataloaders(model, args, fold_frame: pd.DataFrame):
    train_transform, eval_transform = make_transforms(model)
    train_frame = fold_frame[fold_frame["role"] == "train"]
    holdout_frame = fold_frame[fold_frame["role"] == "holdout"]
    train_set = ManifestDataset(Path(args.data_path), train_frame, train_transform)
    val_set = ManifestDataset(Path(args.data_path), holdout_frame, eval_transform)
    test_set = ManifestDataset(
        Path(args.data_path), holdout_frame, copy.deepcopy(eval_transform)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=FIXED_RECIPE["batch_size"],
        shuffle=True,
        num_workers=args.train_workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(FIXED_RECIPE["model_seed"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=FIXED_RECIPE["batch_size"],
        shuffle=False,
        num_workers=args.eval_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(FIXED_RECIPE["model_seed"] + 1),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=FIXED_RECIPE["batch_size"],
        shuffle=False,
        num_workers=args.eval_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(FIXED_RECIPE["model_seed"] + 2),
    )
    return train_loader, val_loader, test_loader


def unique_parameters(parameters):
    result = []
    seen = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


def build_optimizer(model):
    backbone = unique_parameters(model.get_backbone_params())
    backbone_ids = {id(parameter) for parameter in backbone}
    bridge = [
        parameter
        for parameter in unique_parameters(model.get_bridge_params())
        if id(parameter) not in backbone_ids
    ]
    groups = [
        {
            "name": "backbone",
            "params": backbone,
            "lr": FIXED_RECIPE["backbone_lr"],
            "initial_lr": FIXED_RECIPE["backbone_lr"],
            "weight_decay": FIXED_RECIPE["weight_decay"],
        },
        {
            "name": "bridge",
            "params": bridge,
            "lr": FIXED_RECIPE["bridge_lr"],
            "initial_lr": FIXED_RECIPE["bridge_lr"],
            "weight_decay": FIXED_RECIPE["weight_decay"],
        },
    ]
    if not backbone or not bridge:
        raise RuntimeError("Historical optimizer requires nonempty backbone and bridge groups")
    if backbone_ids & {id(parameter) for parameter in bridge}:
        raise RuntimeError("Optimizer groups overlap")
    selected = backbone_ids | {id(parameter) for parameter in bridge}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if selected != trainable:
        raise RuntimeError(
            f"Optimizer does not cover trainable parameters: missing={len(trainable-selected)}, "
            f"extra={len(selected-trainable)}"
        )
    optimizer = optim.AdamW(groups)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    summary = {}
    for group in groups:
        initialized_elements = 0
        uninitialized_tensors = 0
        for parameter in group["params"]:
            try:
                initialized_elements += parameter.numel()
            except ValueError:
                uninitialized_tensors += 1
        summary[group["name"]] = {
            "lr": group["initial_lr"],
            "weight_decay": group["weight_decay"],
            "parameter_tensors": len(group["params"]),
            "initialized_parameter_elements": int(initialized_elements),
            "uninitialized_parameter_tensors": uninitialized_tensors,
            "parameter_names": sorted(parameter_names[id(p)] for p in group["params"]),
        }
    return optimizer, summary


def schedule_group_lrs(optimizer, epoch: int) -> None:
    warmup_epoch = FIXED_RECIPE["warmup_epoch"]
    max_epoch = FIXED_RECIPE["epochs"]
    for group in optimizer.param_groups:
        initial_lr = group["initial_lr"]
        if 0 <= epoch <= warmup_epoch:
            learning_rate = initial_lr * 2.718 ** (
                10 * (float(epoch) / float(warmup_epoch) - 1.0)
            )
            if epoch == warmup_epoch:
                learning_rate = initial_lr
        else:
            learning_rate = initial_lr * (1 - epoch / max_epoch) ** 0.9
        group["lr"] = learning_rate


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    per_f1 = 100.0 * f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_recall = 100.0 * recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "acc": 100.0 * accuracy_score(y_true, y_pred),
        "bmac": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "macro_f1": 100.0
        * f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": 100.0
        * f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "per_class_f1": {
            name: float(value) for name, value in zip(CLASS_NAMES, per_f1)
        },
        "per_class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, per_recall)
        },
    }


def evaluate(model, dataloader, criterion):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, label, _concept_label in tqdm(dataloader, desc="Evaluate", leave=False):
            data = data.float().cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            logits, _concept_logits, sparse_loss = model(data)
            losses.append((criterion(logits, label) + sparse_loss).item())
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    metrics = calculate_metrics(truth, prediction)
    metrics["loss"] = float(np.mean(losses))
    return metrics, truth, prediction


def format_metrics(metrics: dict) -> str:
    return (
        f"ACC={metrics['acc']:.2f}% | BMAC={metrics['bmac']:.2f}% | "
        f"Macro-F1={metrics['macro_f1']:.2f}% | "
        f"Weighted-F1={metrics['weighted_f1']:.2f}%"
    )


def construct_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline trainer requires a CUDA-capable PyTorch setup")
    source = model_source_path()
    source_hash = sha256_file(source)
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            f"Baseline model SHA-256 mismatch: expected {args.expected_model_sha256}, "
            f"found {source_hash}"
        )
    args.dataset = "isic2018"
    args.num_class = len(CLASS_NAMES)
    set_seed(FIXED_RECIPE["model_seed"])
    model = BaselineMVPCBM(concept_list=CONCEPTS, model_name="biomedclip", config=args)
    vit = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=len(CLASS_NAMES)
    )
    vit.head = nn.Identity()
    model.model.visual.trunk.load_state_dict(vit.state_dict())
    model.cuda()
    set_seed(FIXED_RECIPE["model_seed"])
    return model


def save_artifacts(
    output_dir: Path,
    args,
    model,
    manifest_metadata: dict,
    fold_audit: dict,
    optimizer_summary: dict,
    historical: dict,
    best_epoch: int,
    best_val_bmac: float,
    best_val_metrics: dict,
    test_metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_frame: pd.DataFrame,
    history: list[dict],
):
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    history_path = output_dir / "history.csv"
    write_csv(history_path, history)
    predictions = []
    for row, true_id, pred_id in zip(sample_frame.itertuples(), y_true, y_pred):
        predictions.append(
            {
                "fold": args.fold,
                "image": row.image,
                "lesion_id": row.lesion_id,
                "lesion_seen_in_train": bool(row.lesion_seen_in_train),
                "true_id": int(true_id),
                "true_class": CLASS_NAMES[int(true_id)],
                "pred_id": int(pred_id),
                "pred_class": CLASS_NAMES[int(pred_id)],
            }
        )
    predictions_path = output_dir / "predictions.csv"
    write_csv(predictions_path, predictions)
    matrix_rows = [
        {"true\\pred": name, **dict(zip(CLASS_NAMES, row.tolist()))}
        for name, row in zip(CLASS_NAMES, matrix)
    ]
    write_csv(output_dir / "confusion_matrix.csv", matrix_rows)
    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    trainer_path = Path(__file__).resolve()
    model_path = model_source_path()
    manifest_path = Path(fold_audit["manifest"])
    result = {
        "variant": "baseline",
        "protocol": "baseline_cv10_imagelevel_shared_val_test_diagnostic",
        "result_scope": "checkpoint-selected shared holdout; not an independent test estimate",
        "fold": args.fold,
        "metric_unit": "percent",
        "acc": test_metrics["acc"],
        "bmac": test_metrics["bmac"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "per_class_f1": test_metrics["per_class_f1"],
        "per_class_recall": test_metrics["per_class_recall"],
        "loss": test_metrics["loss"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_val_bmac": best_val_bmac,
        "validation_metrics_at_best_checkpoint": best_val_metrics,
        "shared_val_test_metrics_verified": all(
            abs(float(best_val_metrics[key]) - float(test_metrics[key])) < 1e-10
            for key in ("acc", "bmac", "macro_f1", "weighted_f1", "loss")
        ),
        "train_size": fold_audit["train_size"],
        "val_size": fold_audit["val_size"],
        "test_size": fold_audit["test_size"],
        "val_test_overlap": fold_audit["val_test_overlap"],
        "holdout_class_counts": fold_audit["holdout_class_counts"],
        "train_class_counts": fold_audit["train_class_counts"],
        "holdout_lesion_seen_in_train_count": fold_audit[
            "holdout_lesion_seen_in_train_count"
        ],
        "holdout_lesion_seen_in_train_ratio": fold_audit[
            "holdout_lesion_seen_in_train_ratio"
        ],
        "hyperparameters": FIXED_RECIPE,
        "runtime_config": {
            "gpu_argument": args.gpu,
            "train_workers": args.train_workers,
            "eval_workers": args.eval_workers,
            "data_path": str(Path(args.data_path).resolve()),
            "manifest_dir": str(Path(args.manifest_dir).resolve()),
            "tensorboard_dir": str(Path(args.tensorboard_dir).resolve()),
        },
        "optimizer_groups": optimizer_summary,
        "model_source": str(model_path),
        "model_source_sha256": sha256_file(model_path),
        "trainer_source": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_metadata": str((Path(args.manifest_dir) / "split_metadata.json").resolve()),
        "assignments_manifest_sha256": manifest_metadata[
            "assignments_manifest_sha256"
        ],
        "source_ground_truth_sha256": manifest_metadata[
            "source_ground_truth_sha256"
        ],
        "official_lesion_grouping_sha256": manifest_metadata[
            "official_lesion_grouping_sha256"
        ],
        "historical_reference_metrics": str(Path(args.historical_metrics).resolve()),
        "historical_reference_metrics_sha256": sha256_file(args.historical_metrics),
        "historical_reference_result": {
            key: historical[key]
            for key in ("acc", "bmac", "macro_f1", "weighted_f1")
        },
        "loss_history": str(history_path.resolve()),
        "loss_history_sha256": sha256_file(history_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": "warn_only",
            "tf32_enabled": False,
        },
        "interpretation_limits": manifest_metadata["interpretation_limits"],
    }
    if not result["shared_val_test_metrics_verified"]:
        raise RuntimeError("Validation and test metrics differ despite identical holdouts")
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    flat = {
        key: value
        for key, value in result.items()
        if not isinstance(value, (dict, list))
    }
    for name in CLASS_NAMES:
        flat[f"f1_{name}"] = result["per_class_f1"][name]
        flat[f"recall_{name}"] = result["per_class_recall"][name]
    write_csv(output_dir / "metrics.csv", [flat])
    return result, report_text


def train(model, args, fold_frame, manifest_metadata, fold_audit, historical):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(Path(args.tensorboard_dir).resolve() / f"fold_{args.fold:02d}"))
    report_path = output_dir / "training_report.txt"
    train_loader, val_loader, test_loader = make_dataloaders(model, args, fold_frame)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    optimizer, optimizer_summary = build_optimizer(model)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "warning": "val=test; checkpoint-selected scores are optimistic",
            "fold": args.fold,
            "recipe": FIXED_RECIPE,
            "optimizer_groups": optimizer_summary,
            "fold_audit": fold_audit,
        }, indent=2, ensure_ascii=False))
        handle.write("\n")

    best_val_bmac = float("-inf")
    best_epoch = -1
    best_path = output_dir / "best_bmac.pth"
    history = []
    for epoch in range(FIXED_RECIPE["epochs"]):
        model.train()
        schedule_group_lrs(optimizer, epoch)
        classification_losses = []
        concept_losses = []
        total_losses = []
        progress = tqdm(
            train_loader,
            desc=f"baseline fold {args.fold:02d} epoch {epoch+1}/100",
            leave=False,
        )
        for data, label, concept_label in progress:
            data = data.float().cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            concept_label = concept_label.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, concept_logits, sparse_loss = model(data)
            concept_loss = torch.zeros((), device=data.device)
            for concept_index, key in enumerate(model.concept_token_dict.keys()):
                concept_loss = concept_loss + F.cross_entropy(
                    concept_logits[key], concept_label[:, concept_index]
                )
            average_concept_loss = concept_loss / len(model.concept_token_dict)
            classification_loss = criterion(logits, label)
            total_loss = (
                classification_loss
                + FIXED_RECIPE["lambda_cpt"] * average_concept_loss
                + sparse_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at fold {args.fold}, epoch {epoch+1}"
                )
            total_loss.backward()
            optimizer.step()
            classification_losses.append(classification_loss.item())
            concept_losses.append(concept_loss.item())
            total_losses.append(total_loss.item())
            progress.set_postfix(
                cls=f"{classification_loss.item():.4f}",
                concept=f"{concept_loss.item():.4f}",
            )

        val_metrics, _, _ = evaluate(model, val_loader, criterion)
        row = {
            "epoch": epoch + 1,
            "train_cls_loss": float(np.mean(classification_losses)),
            "train_concept_loss_sum": float(np.mean(concept_losses)),
            "train_total_loss": float(np.mean(total_losses)),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_bmac": val_metrics["bmac"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            **{
                f"lr_{group['name']}": float(group["lr"])
                for group in optimizer.param_groups
            },
        }
        history.append(row)
        print(f"Fold {args.fold:02d} epoch {epoch+1}: {format_metrics(val_metrics)}")
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch + 1)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(f"Epoch {epoch+1}: {format_metrics(val_metrics)}\n")
        if val_metrics["bmac"] >= best_val_bmac:
            best_val_bmac = val_metrics["bmac"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint: epoch={best_epoch}, BMAC={best_val_bmac:.2f}%")

    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.cuda()
    best_val_metrics, val_true, val_pred = evaluate(model, val_loader, criterion)
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion)
    if not np.array_equal(val_true, y_true) or not np.array_equal(val_pred, y_pred):
        raise RuntimeError("Shared validation/test predictions unexpectedly differ")
    sample_frame = test_loader.dataset.df
    result, class_report = save_artifacts(
        output_dir,
        args,
        model,
        manifest_metadata,
        fold_audit,
        optimizer_summary,
        historical,
        best_epoch,
        best_val_bmac,
        best_val_metrics,
        test_metrics,
        y_true,
        y_pred,
        sample_frame,
        history,
    )
    print(f"Fold {args.fold:02d} final: {format_metrics(result)}")
    print(class_report)
    writer.close()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = model_source_path()
    source_hash = sha256_file(source)
    if source_hash != args.expected_model_sha256:
        raise RuntimeError(
            f"Baseline model SHA-256 mismatch: expected {args.expected_model_sha256}, "
            f"found {source_hash}"
        )
    historical = validate_historical_recipe(Path(args.historical_metrics).resolve())
    fold_frame, manifest_metadata, fold_audit = load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    output_dir = Path(args.output_dir).resolve()
    if not args.validate_config_only and output_dir.exists():
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")
    static_audit = {
        "validation_only": args.validate_config_only,
        "validation_scope": (
            "static recipe, import, hashes, historical reference, and manifest"
            if args.validate_config_only
            else "static checks plus constructed model and optimizer parameter coverage"
        ),
        "fold": args.fold,
        "model_source": str(source),
        "model_source_sha256": source_hash,
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(__file__),
        "recipe": FIXED_RECIPE,
        "optimizer_groups_expected": {
            "backbone": {
                "source": "model.get_backbone_params()",
                "lr": FIXED_RECIPE["backbone_lr"],
                "weight_decay": FIXED_RECIPE["weight_decay"],
            },
            "bridge": {
                "source": "model.get_bridge_params() excluding any backbone overlap",
                "lr": FIXED_RECIPE["bridge_lr"],
                "weight_decay": FIXED_RECIPE["weight_decay"],
            },
        },
        "manifest_sha256": fold_audit["manifest_sha256"],
        "historical_reference": str(Path(args.historical_metrics).resolve()),
        "historical_reference_acc": historical["acc"],
        "historical_reference_bmac": historical["bmac"],
    }
    if args.validate_config_only:
        print(json.dumps(static_audit, indent=2, ensure_ascii=False))
        print("Configuration validation completed; no training or output directory created.")
        return
    model = construct_model(args)
    optimizer, optimizer_summary = build_optimizer(model)
    static_audit["optimizer_groups"] = optimizer_summary
    print(json.dumps(static_audit, indent=2, ensure_ascii=False))
    del optimizer
    train(model, args, fold_frame, manifest_metadata, fold_audit, historical)


if __name__ == "__main__":
    main()
