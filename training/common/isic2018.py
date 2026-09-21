"""Small shared helpers for audited ISIC2018 training entry points."""

from __future__ import annotations

import json
from pathlib import Path

import train_isic2018_baseline_cv10 as baseline


TRAINING_RECIPE_KEYS = (
    "epochs",
    "batch_size",
    "warmup_epoch",
    "optimizer",
    "backbone_lr",
    "bridge_lr",
    "weight_decay",
    "lambda_cpt",
    "model_seed",
    "checkpoint_selection",
    "post_model_seed_reset",
    "lr_schedule",
    "drop_last_train",
    "model_name",
    "visual_trunk_initialization",
    "criterion",
    "class_weights",
    "train_augmentation",
    "eval_transform",
    "dataloader_generator_seeds",
    "deterministic_algorithms",
    "tf32",
)


def load_baseline_reference(path: Path, fold: int, fold_audit: dict) -> dict:
    """Load and verify the same-fold diagnostic baseline reference."""

    if not path.is_file():
        raise FileNotFoundError(f"Fold baseline reference is missing: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "fold",
        "acc",
        "bmac",
        "macro_f1",
        "weighted_f1",
        "best_epoch",
        "model_source_sha256",
        "manifest_sha256",
    }
    missing = required - set(metrics)
    if missing:
        raise RuntimeError(f"Baseline reference lacks fields: {sorted(missing)}")
    if metrics["fold"] != fold:
        raise RuntimeError(
            f"Baseline reference is fold {metrics['fold']}, requested fold is {fold}"
        )
    if metrics["manifest_sha256"] != fold_audit["manifest_sha256"]:
        raise RuntimeError(
            "Baseline reference and experiment use different fold manifests"
        )
    source_hash = baseline.sha256_file(baseline.model_source_path())
    if metrics["model_source_sha256"] != source_hash:
        raise RuntimeError(
            "Baseline reference does not match the current refined baseline source"
        )
    return metrics


__all__ = ["TRAINING_RECIPE_KEYS", "load_baseline_reference"]
