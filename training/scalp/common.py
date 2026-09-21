#!/usr/bin/env python3
"""Shared engine for controlled scalp MVP-CBM experiments.

Public experiment entry points are intentionally small.  This module owns the
audited data contract, deterministic recipe, checkpoint retention, metrics,
and smoke-test assertions so the three variants differ only in model logic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import random
from dataclasses import dataclass
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import utils
from .dataset import ScalpDataset
from model.mvpcbm_attribute_wavelet_last_v3 import (
    mvpcbm as AuditedAttributeWaveletLaSTV3,
)
from model.mvpcbm_attribute_wavelet_last_v3_configurable import (
    ConfigurableAttributeWaveletLaSTV3,
)
from model.mvpcbm_attribute_wavelet_last_new_v5_configurable import (
    ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5,
)
from model.mlaw_cbm_configurable import (
    ConfigurableMLAWCBM,
    ConfigurableMLAWCBMEnergy,
)
from model.mvpcbm_refined import mvpcbm as BaselineMVPCBM
from .config import (
    ATTRIBUTE_PROMPTS,
    CHECKPOINT_ALIASES,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    CONCEPT_LABEL_MAP,
    CONCEPTS,
    DATALOADER_SEEDS,
    DATA_ROOT,
    EXPECTED_SPLIT_SIZES,
    FIXED_RECIPE,
    HIDDEN_DIM,
    NEW_V5_RECIPE,
    PATCH_COUNT,
    PAPER_GLOBAL_ATTRIBUTE_PROMPTS,
    PROTOCOL,
    TRADEOFF_WEIGHTS,
    V3_RECIPE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILENAME = "dataset.py"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
AUDITED_SOURCE_HASHES = {
    "baseline": "29e0d80dd12830cd40d89f64cb9995b3bf5f26196577b44c8fcec9d4200cdc55",
    "isic_v3": "6a2c98eee9f489766f21cfa20748012a965bba7f7d5090ea62f98c111eff9348",
    "isic_new_v5": "647263b17bbc111c0f85f9794ab03e70ab3b9b5da32c1494d98b4ea2df491786",
}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    variant_name: str
    model_class: type[nn.Module]
    model_filename: str
    output_name: str
    uses_attribute_selector: bool
    counterfactual: bool = False
    paper_global_attribute_query: bool = False


VARIANTS = {
    "baseline": VariantSpec(
        key="baseline",
        variant_name="controlled_original_mvpcbm_baseline",
        model_class=BaselineMVPCBM,
        model_filename="mvpcbm_refined.py",
        output_name="scalp_baseline_fold04_recipe_seed43",
        uses_attribute_selector=False,
    ),
    "v3": VariantSpec(
        key="v3",
        variant_name="attribute_wavelet_last_v3",
        model_class=ConfigurableAttributeWaveletLaSTV3,
        model_filename="mvpcbm_attribute_wavelet_last_v3_configurable.py",
        output_name="scalp_attribute_wavelet_last_v3_k98_fold04_recipe_seed43",
        uses_attribute_selector=True,
    ),
    "new_v5": VariantSpec(
        key="new_v5",
        variant_name="counterfactual_band_selected_attribute_wavelet_last_new_v5",
        model_class=ConfigurableCounterfactualBandSelectedWaveletLaSTNewV5,
        model_filename="mvpcbm_attribute_wavelet_last_new_v5_configurable.py",
        output_name=(
            "scalp_attribute_wavelet_last_new_v5_k98_"
            "fold04_recipe_seed43"
        ),
        uses_attribute_selector=True,
        counterfactual=True,
    ),
    "mlaw": VariantSpec(
        key="mlaw",
        variant_name="mlaw_cbm",
        model_class=ConfigurableMLAWCBM,
        model_filename="mlaw_cbm_configurable.py",
        output_name="scalp_mlaw_cbm_k98_fold04_recipe_seed43",
        uses_attribute_selector=True,
        counterfactual=True,
    ),
    "mlaw_energy": VariantSpec(
        key="mlaw_energy",
        variant_name="mlaw_cbm_energy",
        model_class=ConfigurableMLAWCBMEnergy,
        model_filename="mlaw_cbm_configurable.py",
        output_name="scalp_mlaw_cbm_energy_k98_fold04_recipe_seed43",
        uses_attribute_selector=True,
        counterfactual=True,
    ),
}


def get_variant(key: str) -> VariantSpec:
    try:
        return VARIANTS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown scalp experiment variant: {key!r}") from exc


def parse_args(spec: VariantSpec) -> argparse.Namespace:
    result_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description=f"Controlled scalp MVP-CBM experiment: {spec.variant_name}"
    )
    parser.add_argument("--data-path", default=DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        default=str(result_root / "output" / "scalp" / spec.output_name),
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=str(result_root / "log" / "scalp" / spec.output_name),
    )
    parser.add_argument(
        "--expected-model-sha256",
        required=True,
        help="Required guard against running a different model source.",
    )
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
        char not in "0123456789abcdefABCDEF"
        for char in args.expected_model_sha256
    ):
        parser.error("--expected-model-sha256 must be a 64-character hex digest")
    return args


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)!r}")
    return state


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


def model_source_path(spec: VariantSpec) -> Path:
    imported = Path(inspect.getfile(spec.model_class)).resolve()
    expected = (PROJECT_ROOT / "model" / spec.model_filename).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected {spec.key} model import: {imported}; expected {expected}"
        )
    return imported


def dataset_source_path() -> Path:
    imported = Path(inspect.getfile(ScalpDataset)).resolve()
    expected = (PROJECT_ROOT / "training" / "scalp" / DATASET_FILENAME).resolve()
    if imported != expected:
        raise RuntimeError(
            f"Unexpected scalp dataset import: {imported}; expected {expected}"
        )
    return imported


def dependency_sources(spec: VariantSpec) -> list[Path]:
    paths = [PROJECT_ROOT / "model" / "mvpcbm_refined.py"]
    if spec.uses_attribute_selector:
        paths.extend(
            [
                PROJECT_ROOT / "model" / "mvpcbm_attribute_wavelet_last_v3.py",
                PROJECT_ROOT
                / "model"
                / "mvpcbm_attribute_wavelet_last_v3_configurable.py",
            ]
        )
    if spec.key in {"new_v5", "mlaw", "mlaw_energy"}:
        paths.extend(
            [
                PROJECT_ROOT
                / "model"
                / "mvpcbm_attribute_wavelet_last_new_v5.py",
                PROJECT_ROOT
                / "model"
                / "mvpcbm_attribute_wavelet_last_new_v5_configurable.py",
            ]
        )
        if spec.key in {"mlaw", "mlaw_energy"}:
            paths.extend(
                [
                    PROJECT_ROOT
                    / "model"
                    / "mvpcbm_attribute_wavelet_last_v5_4.py",
                    PROJECT_ROOT
                    / "model"
                    / "mlaw_cbm_configurable.py",
                ]
            )
        if spec.key == "mlaw_energy":
            paths.extend(
                [
                    PROJECT_ROOT
                    / "model"
                    / "mvpcbm_attribute_wavelet_last_v5_7.py",
                ]
            )
    return list(dict.fromkeys(path.resolve() for path in paths))


def validate_preserved_sources() -> dict:
    current_baseline = PROJECT_ROOT / "model" / "mvpcbm_refined.py"
    isic_v3 = PROJECT_ROOT / "model" / "mvpcbm_attribute_wavelet_last_v3.py"
    isic_new_v5 = (
        PROJECT_ROOT / "model" / "mvpcbm_attribute_wavelet_last_new_v5.py"
    )
    paths = {
        "working_baseline": current_baseline,
        "preserved_isic_v3": isic_v3,
        "preserved_isic_new_v5": isic_new_v5,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required preserved model sources missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in (
        ("working_baseline", AUDITED_SOURCE_HASHES["baseline"]),
        ("preserved_isic_v3", AUDITED_SOURCE_HASHES["isic_v3"]),
        ("preserved_isic_new_v5", AUDITED_SOURCE_HASHES["isic_new_v5"]),
    ):
        if hashes[name] != expected:
            raise RuntimeError(
                f"Preserved source hash mismatch for {name}: "
                f"expected {expected}, found {hashes[name]}"
            )
    return {
        name: {"path": str(paths[name].resolve()), "sha256": digest}
        for name, digest in hashes.items()
    }


def validate_static_model_contract(spec: VariantSpec) -> dict:
    if tuple(ATTRIBUTE_PROMPTS.keys()) != tuple(CONCEPTS.keys()):
        raise RuntimeError("Scalp Attribute prompt order differs from concept order")
    if len(CONCEPTS) != 6:
        raise RuntimeError(f"Scalp must have six Attributes, found {len(CONCEPTS)}")
    if tuple(PAPER_GLOBAL_ATTRIBUTE_PROMPTS) != tuple(CONCEPTS):
        raise RuntimeError(
            "Paper-global Attribute prompt order differs from concept order"
        )
    for attribute, states in CONCEPTS.items():
        expected_prompt = (
            f"this is a scalp dermoscopic image, the {attribute} of the scalp "
            f"condition is {' '.join(states)}"
        )
        if PAPER_GLOBAL_ATTRIBUTE_PROMPTS[attribute] != expected_prompt:
            raise RuntimeError(
                f"Paper-global prompt no longer matches Eq. (6) construction: "
                f"{attribute}"
            )
    for class_id, targets in CONCEPT_LABEL_MAP.items():
        if len(targets) != len(CONCEPTS):
            raise RuntimeError(
                f"Class {class_id} concept target length is {len(targets)}, expected 6"
            )
        for concept_index, target in enumerate(targets):
            states = tuple(CONCEPTS.values())[concept_index]
            if not 0 <= target < len(states):
                raise RuntimeError(
                    f"Invalid concept target class={class_id}, "
                    f"concept={concept_index}, target={target}"
                )
    result = {
        "attribute_count": len(CONCEPTS),
        "attribute_order": list(CONCEPTS),
        "attribute_prompt_order": list(ATTRIBUTE_PROMPTS),
        "paper_global_attribute_prompt_order": list(
            PAPER_GLOBAL_ATTRIBUTE_PROMPTS
        ),
        "selector_query_source": (
            "paper-global attribute name plus all concept states"
            if spec.paper_global_attribute_query
            else "attribute-only prompt"
            if spec.uses_attribute_selector
            else None
        ),
        "selector_query_includes_concept_states": (
            spec.paper_global_attribute_query
            if spec.uses_attribute_selector
            else None
        ),
        "original_cls_used_by_selector": None,
        "downstream_forward_reused_from_audited_v3": None,
    }
    if spec.uses_attribute_selector:
        if not issubclass(spec.model_class, AuditedAttributeWaveletLaSTV3):
            raise RuntimeError("Configurable selector model must reuse audited V3 forward")
        reuses_audited_forward = (
            spec.model_class.forward is AuditedAttributeWaveletLaSTV3.forward
        )
        if spec.key not in {"mlaw", "mlaw_energy"} and not reuses_audited_forward:
            raise RuntimeError("Configurable model unexpectedly overrides audited V3 forward")
        forward_source = inspect.getsource(spec.model_class.forward)
        if "patch_tokens = block_tokens[:, 1:, :]" not in forward_source:
            raise RuntimeError("Audited V3 forward no longer excludes the original CLS")
        if spec.key in {"mlaw", "mlaw_energy"}:
            required_fragments = (
                "return_details=True",
                'selector_details["filtered_high_tokens"]',
            )
            missing = [
                fragment for fragment in required_fragments
                if fragment not in forward_source
            ]
            if missing:
                raise RuntimeError(
                    f"{spec.key} forward lost Attribute-guided concept pooling: "
                    f"missing={missing}"
                )
        result.update(
            {
                "original_cls_used_by_selector": False,
                "downstream_forward_reused_from_audited_v3": (
                    reuses_audited_forward
                ),
                "all_vit_layers_expected": 12,
                "top_k_axis": "patch dimension (196) independently per channel (768)",
                "selected_value_source": "original ViT patch tokens",
                "concept_pooling": (
                    "AP(X) plus separate wavelet RMS energy"
                    if spec.key == "mlaw_energy"
                    else "AP(X + scaled counterfactual wavelet residual)"
                    if spec.key == "mlaw"
                    else "baseline AP"
                ),
                "concept_pooling_attention_detached": None,
                "concept_pooling_gradient_scale": None,
            }
        )
    return result


def resolved_recipe(spec: VariantSpec) -> dict:
    recipe = copy.deepcopy(FIXED_RECIPE)
    recipe.update(
        {
            "variant": spec.variant_name,
            "protocol": f"{PROTOCOL}_{spec.key}",
            "fold04_meaning": (
                "ISIC2018 Fold04 hyperparameter recipe only; scalp has no Fold04 split"
            ),
            "base_model_source": "model/mvpcbm_refined.py",
            "variant_model_source": f"model/{spec.model_filename}",
            "result_repository": "MLAW-CBM",
            "concept_target_source": "scalp class-derived prototype labels",
            "dynamic_concepts_used": False,
            "checkpoint_selection_policy": (
                "standard report uses maximum validation BMAC; additionally retain "
                "maximum ACC, maximum Macro-F1, and weighted tradeoff checkpoints; "
                "later epoch wins exact ties"
            ),
            "standard_report_checkpoint": "best_bmac.pth",
            "retained_checkpoint_aliases": dict(CHECKPOINT_ALIASES),
            "checkpoint_tradeoff_weights": dict(TRADEOFF_WEIGHTS),
        }
    )
    if spec.key == "baseline":
        recipe.update(
            {
                "architecture": "original MVP-CBM Baseline",
                "attribute_selector_prompts_used": False,
            }
        )
    elif spec.key == "v3":
        recipe.update(copy.deepcopy(V3_RECIPE))
        recipe.update(
            {
                "architecture": "Attribute-Wavelet LaST V3",
                "selector_attribute_names": list(ATTRIBUTE_PROMPTS),
                "selector_attribute_prompts": dict(ATTRIBUTE_PROMPTS),
                "wavelet_transform": "fixed one-level orthonormal 2-D Haar",
                "wavelet_detail_handling": "LH/HL/HH high-only residual",
                "attribute_union": "maximum across six spatial-softmax maps",
                "frequency_score": "channel-wise normalized high-frequency magnitude",
                "top_k_axis": "196 patch positions independently for each of 768 channels",
                "selected_value_source": "original ViT patch values",
                "new_cls_application": "all 12 layers; preference branch only",
                "old_cls_used_for_selection": False,
                "downstream_patch_to_concept_path_changed": False,
            }
        )
    elif spec.key in {"new_v5", "mlaw", "mlaw_energy"}:
        recipe.update(copy.deepcopy(NEW_V5_RECIPE))
        recipe.update(
            {
                "architecture": (
                    "MLAW-CBM-Energy"
                    if spec.key == "mlaw_energy"
                    else "MLAW-CBM"
                    if spec.key == "mlaw"
                    else "Counterfactual Band-Selected Attribute-Wavelet LaST new V5"
                ),
                "selector_attribute_names": list(ATTRIBUTE_PROMPTS),
                "selector_attribute_prompts": dict(ATTRIBUTE_PROMPTS),
                "selector_query_source": "configured attribute-only prompt",
                "paper_global_attribute_query_used": False,
                "selector_query_includes_concept_states": False,
                "wavelet_transform": "fixed one-level orthonormal 2-D Haar",
                "wavelet_detail_handling": (
                    "direct-IDWT LH/HL/HH ablation and independent exact "
                    "band-only residuals"
                ),
                "counterfactual_delta": (
                    "original attribute similarity minus direct single-band-"
                    "removed similarity"
                ),
                "counterfactual_routes": ["none", "lh", "hl", "hh"],
                "counterfactual_routing": "Sparsemax",
                "negative_delta_handling": (
                    "exclude non-positive band routes from wavelet enhancement"
                ),
                "none_route_handling": (
                    "consumes Sparsemax mass and contributes zero residual"
                ),
                "attribute_route_fusion": (
                    "raw-X spatial-attention weighted mean across six Attributes"
                ),
                "filtered_high_residual": (
                    "H*=g_lh*R_lh+g_hl*R_hl+g_hh*R_hh"
                ),
                "semantic_score": "V3 U computed from filtered wavelet patch",
                "frequency_score": (
                    "V3 channel-wise spatial z-score of absolute filtered H*"
                ),
                "selection_score": "S=U*(1+softplus(w_l)*E)",
                "top_k_axis": (
                    "196 patch positions independently for each of 768 channels"
                ),
                "selected_value_source": "original ViT patch values",
                "new_cls_application": "all 12 layers; preference branch only",
                "raw_high_frequency_magnitude_used": True,
                "ll_guided_conv_gate_used": False,
                "high_frequency_residual_injected": True,
                "v3_semantic_and_frequency_scoring_retained": True,
                "individual_concept_state_selector_used": False,
                "old_cls_used_for_selection": False,
                "downstream_patch_to_concept_path_changed": (
                    spec.key in {"mlaw", "mlaw_energy"}
                ),
                "concept_pooling": (
                    "AP(X) plus separately projected counterfactual wavelet RMS energy"
                    if spec.key == "mlaw_energy"
                    else "AP(X + scaled counterfactual wavelet residual)"
                    if spec.key == "mlaw"
                    else "baseline fixed-bin AdaptiveAvgPool1d"
                ),
                "concept_pooling_attention_source": None,
                "concept_pooling_value_source": (
                    "original X and wavelet RMS energy"
                    if spec.key == "mlaw_energy"
                    else "X plus selected wavelet residual"
                    if spec.key == "mlaw"
                    else None
                ),
                "concept_pooling_attention_detached": None,
                "concept_pooling_gradient_scale": None,
                "baseline_adaptive_average_pool_bypassed": False,
                "concept_visual_shape": (
                    "batch x 6 attributes x 768 channels"
                    if spec.key in {"mlaw", "mlaw_energy"}
                    else None
                ),
                "concept_projected_shape": (
                    "batch x 6 attributes x 512 channels"
                    if spec.key in {"mlaw", "mlaw_energy"}
                    else None
                ),
                "concept_projector": (
                    "inherited content projector plus a separate energy projector"
                    if spec.key == "mlaw_energy"
                    else "reuse inherited AvgPoolProjector unchanged"
                    if spec.key == "mlaw"
                    else None
                ),
                "concept_similarity": (
                    "retain inherited logit-scaled dot product; no added normalize"
                    if spec.key in {"mlaw", "mlaw_energy"}
                    else None
                ),
            }
        )
    else:
        raise ValueError(f"No recipe is defined for variant {spec.key!r}")
    return recipe


def validate_data_layout(data_root: Path) -> dict:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Scalp data root does not exist: {data_root}")
    counts: dict[str, dict[str, int]] = {}
    inventory: list[dict] = []
    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        counts[split] = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise RuntimeError(f"No images found in {class_dir}")
            counts[split][class_name] = len(paths)
            inventory.extend(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "relative_path": str(path.relative_to(data_root)),
                    "size_bytes": path.stat().st_size,
                }
                for path in paths
            )
    sizes = {
        split: sum(class_counts.values())
        for split, class_counts in counts.items()
    }
    if sizes != EXPECTED_SPLIT_SIZES:
        raise RuntimeError(
            f"Controlled scalp split sizes changed: expected "
            f"{EXPECTED_SPLIT_SIZES}, found {sizes}"
        )
    digest = hashlib.sha256()
    for row in inventory:
        digest.update(
            (
                f"{row['split']}\t{row['class_id']}\t{row['relative_path']}\t"
                f"{row['size_bytes']}\n"
            ).encode("utf-8")
        )
    return {
        "counts": counts,
        "train_size": sizes["train"],
        "val_size": sizes["test"],
        "test_size": sizes["test"],
        "inventory": inventory,
        "inventory_sha256": digest.hexdigest(),
    }


def sort_dataset(dataset: ScalpDataset) -> None:
    paired = sorted(zip(dataset.image_paths, dataset.labels), key=lambda item: item[0])
    dataset.image_paths = [path for path, _label in paired]
    dataset.labels = [int(label) for _path, label in paired]


def validate_dataset_contract(data_root: Path) -> dict:
    datasets = {
        mode: ScalpDataset(
            str(data_root),
            mode=mode,
            transforms=None,
            flag=2,
            config=argparse.Namespace(),
            return_concept_label=True,
        )
        for mode in ("train", "val", "test")
    }
    for dataset in datasets.values():
        sort_dataset(dataset)
        if tuple(dataset.class_names) != tuple(CLASS_NAMES):
            raise RuntimeError(
                f"Dataset class order mismatch: {dataset.class_names} != {CLASS_NAMES}"
            )
        found_map = {
            int(key): tuple(int(value) for value in values)
            for key, values in dataset.concept_label_map.items()
        }
        if found_map != CONCEPT_LABEL_MAP:
            raise RuntimeError(
                f"Dataset concept label map mismatch: {found_map} != {CONCEPT_LABEL_MAP}"
            )
    if datasets["val"].image_paths != datasets["test"].image_paths:
        raise RuntimeError("Validation and test paths are expected to overlap 100%")
    if datasets["val"].labels != datasets["test"].labels:
        raise RuntimeError("Validation and test labels unexpectedly differ")
    return {
        "class_order_verified": True,
        "concept_label_map_verified": True,
        "val_test_overlap_percent": 100.0,
        "val_test_same_order": True,
        "dataset_sizes": {
            mode: len(dataset) for mode, dataset in datasets.items()
        },
    }


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


def make_dataloaders(model, args: argparse.Namespace, smoke_test: bool = False):
    train_transform, eval_transform = make_transforms(model)
    data_root = str(Path(args.data_path).resolve())
    train_set = ScalpDataset(
        data_root,
        mode="train",
        transforms=train_transform,
        flag=2,
        config=args,
        return_concept_label=True,
    )
    val_set = ScalpDataset(
        data_root,
        mode="val",
        transforms=eval_transform,
        flag=2,
        config=args,
        return_concept_label=True,
    )
    test_set = ScalpDataset(
        data_root,
        mode="test",
        transforms=copy.deepcopy(eval_transform),
        flag=2,
        config=args,
        return_concept_label=True,
    )
    for dataset in (train_set, val_set, test_set):
        sort_dataset(dataset)
    if val_set.image_paths != test_set.image_paths:
        raise RuntimeError("Validation/test path identity was lost")
    batch_size = 2 if smoke_test else FIXED_RECIPE["batch_size"]
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if smoke_test else args.train_workers,
        drop_last=FIXED_RECIPE["drop_last_train"] and not smoke_test,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["train"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if smoke_test else args.eval_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["val"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if smoke_test else args.eval_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(DATALOADER_SEEDS["test"]),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def apply_variant_config(args: argparse.Namespace, spec: VariantSpec) -> None:
    args.dataset = "scalp"
    args.num_class = len(CLASS_NAMES)
    if spec.key == "v3":
        for name, value in V3_RECIPE.items():
            setattr(args, name, value)
    elif spec.key in {"new_v5", "mlaw", "mlaw_energy"}:
        for name, value in NEW_V5_RECIPE.items():
            setattr(args, name, value)


def construct_model(spec: VariantSpec, args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The controlled scalp trainer requires a CUDA-capable PyTorch setup"
        )
    apply_variant_config(args, spec)
    set_seed(FIXED_RECIPE["model_seed"])
    kwargs = {
        "model_name": FIXED_RECIPE["model_name"],
        "config": args,
    }
    if spec.uses_attribute_selector:
        kwargs["attribute_prompts"] = ATTRIBUTE_PROMPTS
    model = spec.model_class(CONCEPTS, **kwargs)
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=len(CLASS_NAMES),
    )
    vit.head = nn.Identity()
    model.model.visual.trunk.load_state_dict(vit.state_dict())
    model.cuda()
    set_seed(FIXED_RECIPE["model_seed"])
    return model


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
        raise RuntimeError("Fold04 optimizer requires nonempty backbone and bridge groups")
    selected = backbone_ids | {id(parameter) for parameter in bridge}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if selected != trainable:
        raise RuntimeError(
            "Optimizer does not cover trainable parameters: "
            f"missing={len(trainable - selected)}, extra={len(selected - trainable)}"
        )
    optimizer = optim.AdamW(groups)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    summary = {}
    for group in groups:
        elements = 0
        uninitialized = 0
        for parameter in group["params"]:
            try:
                elements += parameter.numel()
            except ValueError:
                uninitialized += 1
        summary[group["name"]] = {
            "lr": group["initial_lr"],
            "weight_decay": group["weight_decay"],
            "parameter_tensors": len(group["params"]),
            "initialized_parameter_elements": int(elements),
            "uninitialized_parameter_tensors": uninitialized,
            "parameter_names": sorted(names[id(parameter)] for parameter in group["params"]),
        }
    return optimizer, summary


def schedule_group_lrs(optimizer, epoch_index: int) -> None:
    for group in optimizer.param_groups:
        initial_lr = group["initial_lr"]
        if 0 <= epoch_index <= FIXED_RECIPE["warmup_epoch"]:
            learning_rate = initial_lr * 2.718 ** (
                10
                * (
                    float(epoch_index) / float(FIXED_RECIPE["warmup_epoch"])
                    - 1.0
                )
            )
            if epoch_index == FIXED_RECIPE["warmup_epoch"]:
                learning_rate = initial_lr
        else:
            learning_rate = initial_lr * (
                1 - epoch_index / FIXED_RECIPE["epochs"]
            ) ** 0.9
        group["lr"] = learning_rate


def compute_losses(model, data, labels, concept_targets, criterion):
    logits, concept_logits, sparse_loss = model(data)
    concept_loss_sum = torch.zeros((), device=data.device)
    for concept_index, key in enumerate(model.concept_token_dict):
        concept_loss_sum = concept_loss_sum + F.cross_entropy(
            concept_logits[key], concept_targets[:, concept_index]
        )
    concept_loss = concept_loss_sum / len(model.concept_token_dict)
    classification_loss = criterion(logits, labels)
    total_loss = (
        classification_loss
        + FIXED_RECIPE["lambda_cpt"] * concept_loss
        + sparse_loss
    )
    return logits, concept_logits, classification_loss, concept_loss, sparse_loss, total_loss


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": 100.0 * float(precision[index]),
            "recall": 100.0 * float(recall[index]),
            "f1": 100.0 * float(per_f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "acc": 100.0 * accuracy_score(y_true, y_pred),
        "bmac": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "macro_f1": 100.0
        * f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": 100.0
        * f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "per_class": per_class,
        "per_class_precision": {
            name: values["precision"] for name, values in per_class.items()
        },
        "per_class_recall": {
            name: values["recall"] for name, values in per_class.items()
        },
        "per_class_f1": {name: values["f1"] for name, values in per_class.items()},
    }


def evaluate(model, dataloader, criterion):
    model.eval()
    totals = {
        "classification_loss": 0.0,
        "concept_loss": 0.0,
        "sparse_loss": 0.0,
        "total_loss": 0.0,
    }
    y_true, y_pred, probabilities = [], [], []
    sample_count = 0
    with torch.no_grad():
        for data, labels, concept_targets in tqdm(
            dataloader, desc="Evaluate", leave=False
        ):
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            (
                logits,
                _concept_logits,
                cls_loss,
                concept_loss,
                sparse_loss,
                total_loss,
            ) = compute_losses(model, data, labels, concept_targets, criterion)
            batch_size = labels.size(0)
            sample_count += batch_size
            for key, value in (
                ("classification_loss", cls_loss),
                ("concept_loss", concept_loss),
                ("sparse_loss", sparse_loss),
                ("total_loss", total_loss),
            ):
                totals[key] += float(value.item()) * batch_size
            probs = logits.softmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    if sample_count == 0:
        raise RuntimeError("Evaluation loader produced no samples")
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    metrics = calculate_metrics(truth, prediction)
    metrics.update({key: value / sample_count for key, value in totals.items()})
    metrics["sample_count"] = sample_count
    return metrics, truth, prediction, np.asarray(probabilities, dtype=np.float32)


def format_metrics(metrics: dict) -> str:
    return (
        f"ACC={metrics['acc']:.2f}% | BMAC={metrics['bmac']:.2f}% | "
        f"Macro-F1={metrics['macro_f1']:.2f}% | "
        f"Weighted-F1={metrics['weighted_f1']:.2f}%"
    )


def _install_selector_smoke_audit(model):
    calls: list[dict] = []
    value_source_verified = {"value": False}
    query_source_verified = {"value": False}
    original_build = model.build_attribute_wavelet_cls

    def audited_build(patch_tokens, layer_index: int, return_details: bool = False):
        if tuple(patch_tokens.shape[1:]) != (PATCH_COUNT, HIDDEN_DIM):
            raise RuntimeError(
                "Selector must receive only [B,196,768] patch tokens; found "
                f"{tuple(patch_tokens.shape)}"
            )
        result = original_build(
            patch_tokens,
            layer_index=layer_index,
            return_details=return_details,
        )
        new_cls = result[0] if return_details else result
        if tuple(new_cls.shape) != (patch_tokens.shape[0], HIDDEN_DIM):
            raise RuntimeError(f"Unexpected generated CLS shape: {tuple(new_cls.shape)}")
        calls.append(
            {
                "layer_index": int(layer_index),
                "patch_shape": list(patch_tokens.shape),
                "new_cls_shape": list(new_cls.shape),
            }
        )
        if layer_index == 0:
            if return_details:
                audit_cls, details = result
            else:
                audit_cls, details = original_build(
                    patch_tokens,
                    layer_index=layer_index,
                    return_details=True,
                )
            indices = details["topk_indices"]
            weights = details["aggregation_weights"]
            expected_index_shape = (
                patch_tokens.shape[0],
                model.attribute_wavelet_top_k,
                HIDDEN_DIM,
            )
            if tuple(indices.shape) != expected_index_shape:
                raise RuntimeError(
                    f"Top-K must be [B,K,768], found {tuple(indices.shape)}"
                )
            if int(indices.min()) < 0 or int(indices.max()) >= PATCH_COUNT:
                raise RuntimeError("Top-K indices are outside the 196 patch positions")
            selected_original_values = torch.gather(
                patch_tokens, dim=1, index=indices
            )
            recomputed_cls = model.attribute_wavelet_aggregator.output_norm(
                (weights * selected_original_values).sum(dim=1)
            )
            if not torch.allclose(audit_cls, recomputed_cls, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Generated CLS is not aggregating original patch values")
            if not torch.allclose(new_cls, audit_cls, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Selector audit changed the generated CLS")
            selector_embeddings = (
                model.global_attribute_text_embeddings
                if getattr(model, "attribute_selector_uses_concept_states", False)
                else model.attribute_text_embeddings
            )
            expected_queries = F.normalize(
                selector_embeddings
                + model.attribute_wavelet_aggregator.layer_attribute_residuals[
                    layer_index
                ],
                p=2,
                dim=-1,
                eps=model.attribute_wavelet_eps,
            )
            if not torch.allclose(
                details["attribute_queries"],
                expected_queries,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "Selector did not use the declared Attribute query source"
                )
            value_source_verified["value"] = True
            query_source_verified["value"] = True
        return result

    model.build_attribute_wavelet_cls = audited_build
    return calls, value_source_verified, query_source_verified


def run_smoke_test(
    model,
    spec: VariantSpec,
    args: argparse.Namespace,
    output_existed_before: bool,
) -> None:
    train_loader, _val_loader, _test_loader = make_dataloaders(
        model, args, smoke_test=True
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    selector_calls = []
    value_source_verified = {"value": False}
    query_source_verified = {"value": False}
    if spec.uses_attribute_selector:
        (
            selector_calls,
            value_source_verified,
            query_source_verified,
        ) = _install_selector_smoke_audit(model)

    data, labels, concept_targets = next(iter(train_loader))
    data = data.float().cuda(non_blocking=True)
    labels = labels.long().cuda(non_blocking=True)
    concept_targets = concept_targets.long().cuda(non_blocking=True)
    (
        logits,
        concept_logits,
        cls_loss,
        concept_loss,
        sparse_loss,
        total_loss,
    ) = compute_losses(model, data, labels, concept_targets, criterion)
    expected_logits = (len(labels), len(CLASS_NAMES))
    if tuple(logits.shape) != expected_logits:
        raise RuntimeError(
            f"Classification logits shape {tuple(logits.shape)} != {expected_logits}"
        )
    for key, states in CONCEPTS.items():
        expected = (len(labels), len(states))
        if key not in concept_logits or tuple(concept_logits[key].shape) != expected:
            found = None if key not in concept_logits else tuple(concept_logits[key].shape)
            raise RuntimeError(
                f"Concept logits shape mismatch for {key}: {found} != {expected}"
            )
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Smoke-test total loss is non-finite")
    total_loss.backward()

    gradient_audit = {}
    if spec.key in {"v3", "new_v5", "mlaw", "mlaw_energy"}:
        selector = model.attribute_wavelet_aggregator
        required = {
            "layer_attribute_residuals": selector.layer_attribute_residuals.grad,
            "raw_high_feature_scales": selector.raw_high_feature_scales.grad,
            "raw_high_score_weights": selector.raw_high_score_weights.grad,
        }
        gradient_audit = _validate_required_gradients(required)

    selector_audit = None
    if spec.uses_attribute_selector:
        layer_indices = [call["layer_index"] for call in selector_calls]
        if layer_indices != list(range(12)):
            raise RuntimeError(
                f"Expected one generated CLS for each of 12 layers, got {layer_indices}"
            )
        if tuple(model.attribute_names) != tuple(CONCEPTS):
            raise RuntimeError(
                f"Selector Attribute order mismatch: {model.attribute_names}"
            )
        if not value_source_verified["value"]:
            raise RuntimeError("Original patch-value aggregation was not verified")
        if not query_source_verified["value"]:
            raise RuntimeError("Attribute query source was not verified")
        if model.attribute_wavelet_old_cls_used:
            raise RuntimeError("Selector unexpectedly reports use of original CLS")
        if bool(model.attribute_selector_uses_concept_states) != bool(
            spec.paper_global_attribute_query
        ):
            raise RuntimeError(
                "Runtime selector concept-state flag differs from VariantSpec"
            )
        if spec.paper_global_attribute_query:
            expected_global_embeddings = torch.cat(
                [model.global_attr_concepts[name] for name in model.attribute_names],
                dim=0,
            )
            if not torch.equal(
                model.global_attribute_text_embeddings,
                expected_global_embeddings,
            ):
                raise RuntimeError(
                    "V6 selector buffer differs from refined MVP-CBM global Attributes"
                )
        selector_audit = {
            "attribute_count": len(model.attribute_names),
            "attribute_order": list(model.attribute_names),
            "generated_cls_layer_indices": layer_indices,
            "generated_cls_count": len(layer_indices),
            "selector_input_shape_each_layer": "[B,196,768]",
            "top_k": model.attribute_wavelet_top_k,
            "top_k_axis_verified": True,
            "original_patch_values_aggregated": True,
            "attribute_query_source_verified": True,
            "paper_global_attribute_query_used": (
                spec.paper_global_attribute_query
            ),
            "selector_query_includes_concept_states": (
                spec.paper_global_attribute_query
            ),
            "original_cls_used_by_selector": False,
            "downstream_forward_reused_from_audited_v3": (
                spec.model_class.forward is AuditedAttributeWaveletLaSTV3.forward
            ),
            "new_parameter_gradients": gradient_audit,
        }
    output_exists_after = Path(args.output_dir).exists()
    if output_exists_after != output_existed_before:
        raise RuntimeError("Smoke test changed the planned formal output directory")
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "variant": spec.variant_name,
                "batch_size": len(labels),
                "forward_shapes": {
                    "classification_logits": list(logits.shape),
                    "concept_logits": {
                        key: list(value.shape) for key, value in concept_logits.items()
                    },
                    "sparse_loss": list(sparse_loss.shape),
                },
                "losses": {
                    "classification_loss": float(cls_loss.item()),
                    "mean_attribute_concept_loss": float(concept_loss.item()),
                    "lambda_cpt": FIXED_RECIPE["lambda_cpt"],
                    "sparse_loss": float(sparse_loss.item()),
                    "total_loss": float(total_loss.item()),
                },
                "backward_verified": True,
                "selector_audit": selector_audit,
                "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "formal_output_directory_created": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _validate_required_gradients(required: dict[str, torch.Tensor | None]) -> dict:
    missing = [name for name, value in required.items() if value is None]
    nonfinite = [
        name
        for name, value in required.items()
        if value is not None and not torch.isfinite(value).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            f"Selector gradient failure: missing={missing}, nonfinite={nonfinite}"
        )
    return {
        name: {
            "present": True,
            "finite": True,
            "shape": list(value.shape),
        }
        for name, value in required.items()
        if value is not None
    }


def tradeoff_score(metrics: dict) -> float:
    return sum(
        TRADEOFF_WEIGHTS[key] * float(metrics[key]) for key in TRADEOFF_WEIGHTS
    )


def policy_rank(policy: str, metrics: dict, epoch: int) -> tuple[float, ...]:
    if policy == "best_bmac":
        return float(metrics["bmac"]), float(epoch)
    if policy == "best_acc":
        return float(metrics["acc"]), float(epoch)
    if policy == "best_macro_f1":
        return float(metrics["macro_f1"]), float(epoch)
    if policy == "best_tradeoff":
        return float(tradeoff_score(metrics)), float(epoch)
    raise KeyError(policy)


def atomic_relative_symlink(target: Path, alias: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(target, start=alias.parent))
    os.replace(temporary, alias)


def checkpoint_index_payload(
    records: dict,
    last_record: dict | None,
    output_dir: Path,
) -> dict:
    return {
        "warning": (
            "Validation and final test use the same scalp test split; checkpoint "
            "metrics are diagnostic, not an independent test estimate."
        ),
        "standard_report_checkpoint": "best_bmac.pth",
        "tradeoff_formula": "0.50 * BMAC + 0.25 * ACC + 0.25 * Macro-F1",
        "tradeoff_weights": dict(TRADEOFF_WEIGHTS),
        "checkpoint_storage": (
            "one canonical epoch file per unique retained epoch; best aliases are "
            "relative symbolic links; last.pth is the final epoch state"
        ),
        "policies": records,
        "last": last_record,
        "unique_retained_epochs": sorted(
            {int(record["epoch"]) for record in records.values()}
        ),
        "output_dir": str(output_dir),
    }


def write_checkpoint_index(
    records: dict,
    last_record: dict | None,
    output_dir: Path,
) -> None:
    payload = checkpoint_index_payload(records, last_record, output_dir)
    (output_dir / "checkpoint_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def checkpoint_record(
    checkpoint_type: str,
    epoch: int,
    metrics: dict,
    alias: Path,
    actual: Path,
) -> dict:
    return {
        "checkpoint_type": checkpoint_type,
        "epoch": int(epoch),
        "acc": float(metrics["acc"]),
        "bmac": float(metrics["bmac"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "classification_loss": float(metrics["classification_loss"]),
        "concept_loss": float(metrics["concept_loss"]),
        "sparse_loss": float(metrics["sparse_loss"]),
        "total_loss": float(metrics["total_loss"]),
        "tradeoff_score": float(tradeoff_score(metrics)),
        "alias_path": str(alias),
        "actual_checkpoint_path": str(actual),
    }


def update_retained_checkpoints(
    model,
    metrics: dict,
    epoch: int,
    best_ranks: dict,
    records: dict,
    output_dir: Path,
) -> list[str]:
    improved = []
    ranks = {}
    for policy in CHECKPOINT_ALIASES:
        rank = policy_rank(policy, metrics, epoch)
        ranks[policy] = rank
        if policy not in best_ranks or rank > best_ranks[policy]:
            improved.append(policy)
    if not improved:
        return improved

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    canonical = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    if not canonical.exists():
        torch.save(model.state_dict(), canonical)

    for policy in improved:
        best_ranks[policy] = ranks[policy]
        alias = output_dir / CHECKPOINT_ALIASES[policy]
        atomic_relative_symlink(canonical, alias)
        records[policy] = checkpoint_record(
            policy, epoch, metrics, alias, canonical
        )
        records[policy]["rank"] = [float(value) for value in ranks[policy]]

    referenced = {
        Path(record["actual_checkpoint_path"]).resolve()
        for record in records.values()
    }
    for candidate in checkpoint_dir.glob("epoch_*.pth"):
        if candidate.resolve() not in referenced:
            candidate.unlink()
    return improved


def save_final_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    spec: VariantSpec,
    recipe: dict,
    data_audit: dict,
    dataset_contract: dict,
    source_audit: dict,
    static_model_contract: dict,
    optimizer_summary: dict,
    model,
    best_record: dict,
    best_val_metrics: dict,
    test_metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    test_paths: list[str],
    history: list[dict],
) -> None:
    write_csv(output_dir / "history.csv", history)
    write_csv(output_dir / "data_manifest.csv", data_audit["inventory"])
    predictions = []
    data_root = Path(args.data_path).resolve()
    for path, true_id, pred_id, probs in zip(
        test_paths, y_true, y_pred, probabilities
    ):
        row = {
            "image": str(Path(path).resolve().relative_to(data_root)),
            "true_id": int(true_id),
            "true_class": CLASS_NAMES[int(true_id)],
            "pred_id": int(pred_id),
            "pred_class": CLASS_NAMES[int(pred_id)],
        }
        row.update(
            {
                f"prob_{class_name}": float(probs[class_id])
                for class_id, class_name in enumerate(CLASS_NAMES)
            }
        )
        predictions.append(row)
    write_csv(output_dir / "predictions.csv", predictions)

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    write_csv(
        output_dir / "confusion_matrix.csv",
        [
            {"true\\pred": name, **dict(zip(CLASS_NAMES, row.tolist()))}
            for name, row in zip(CLASS_NAMES, matrix)
        ],
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    model_path = model_source_path(spec)
    dataset_path = dataset_source_path()
    checkpoint_index_path = output_dir / "checkpoint_index.json"
    checkpoint_index = json.loads(
        checkpoint_index_path.read_text(encoding="utf-8")
    )
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
        )
    )
    if not shared_metrics_verified:
        raise RuntimeError("Validation/test metrics differ despite identical files")

    metrics = {
        "variant": spec.variant_name,
        "protocol": recipe["protocol"],
        "result_scope": (
            "Validation and final test use the same scalp test split. Results are "
            "a diagnostic comparison, not an independent test estimate."
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
        "validation_metrics_at_best_checkpoint": best_val_metrics,
        "shared_val_test_metrics_verified": shared_metrics_verified,
        "data_path": str(data_root),
        "train_size": data_audit["train_size"],
        "val_size": data_audit["val_size"],
        "test_size": data_audit["test_size"],
        "val_test_overlap": "100%: ScalpDataset maps both val and test to test/",
        "class_order": list(CLASS_NAMES),
        "class_counts": data_audit["counts"],
        "class_weights": list(CLASS_WEIGHTS),
        "attribute_order": list(CONCEPTS),
        "attribute_prompts": (
            dict(PAPER_GLOBAL_ATTRIBUTE_PROMPTS)
            if spec.paper_global_attribute_query
            else dict(ATTRIBUTE_PROMPTS)
            if spec.uses_attribute_selector
            else None
        ),
        "attribute_prompts_used_by_selector": spec.uses_attribute_selector,
        "attribute_only_prompts": (
            dict(ATTRIBUTE_PROMPTS) if spec.uses_attribute_selector else None
        ),
        "paper_global_attribute_prompts": (
            dict(PAPER_GLOBAL_ATTRIBUTE_PROMPTS)
            if spec.paper_global_attribute_query
            else None
        ),
        "selector_query_includes_concept_states": (
            spec.paper_global_attribute_query
            if spec.uses_attribute_selector
            else None
        ),
        "concept_states": {key: list(values) for key, values in CONCEPTS.items()},
        "concept_label_map": {
            str(key): list(values) for key, values in CONCEPT_LABEL_MAP.items()
        },
        "concept_target_source": "scalp class-derived prototype labels",
        "dynamic_concepts_used": False,
        "hyperparameters": recipe,
        "seed": FIXED_RECIPE["model_seed"],
        "dataloader_seeds": dict(DATALOADER_SEEDS),
        "checkpoint_selection_policy": recipe["checkpoint_selection_policy"],
        "standard_report_checkpoint": "best_bmac.pth",
        "checkpoint_index": checkpoint_index,
        "checkpoint_index_path": str(checkpoint_index_path),
        "checkpoint_index_sha256": sha256_file(checkpoint_index_path),
        "model_source": str(model_path),
        "model_source_sha256": sha256_file(model_path),
        "model_dependency_sources": {
            str(path): sha256_file(path) for path in dependency_sources(spec)
        },
        "dataset_source": str(dataset_path),
        "dataset_source_sha256": sha256_file(dataset_path),
        "trainer_source": str(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(__file__),
        "entry_source": str(Path(args.entry_source).resolve()),
        "entry_sha256": sha256_file(args.entry_source),
        "data_inventory_sha256": data_audit["inventory_sha256"],
        "dataset_contract": dataset_contract,
        "static_model_contract": static_model_contract,
        "preserved_source_audit": source_audit,
        "optimizer_groups": optimizer_summary,
        "runtime_config": {
            "gpu_argument": args.gpu,
            "train_workers": args.train_workers,
            "eval_workers": args.eval_workers,
            "output_dir": str(output_dir),
            "tensorboard_dir": str(Path(args.tensorboard_dir).resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": "warn_only",
            "tf32_enabled": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }
    for class_name, values in metrics["per_class_metrics"].items():
        flat[f"precision_{class_name}"] = values["precision"]
        flat[f"recall_{class_name}"] = values["recall"]
        flat[f"f1_{class_name}"] = values["f1"]
        flat[f"support_{class_name}"] = values["support"]
    write_csv(output_dir / "metrics.csv", [flat])


def train(
    model,
    spec: VariantSpec,
    args: argparse.Namespace,
    recipe: dict,
    data_audit: dict,
    dataset_contract: dict,
    source_audit: dict,
    static_model_contract: dict,
    optimizer_summary: dict,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_dir = Path(args.tensorboard_dir).resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tensorboard_dir))
    write_csv(output_dir / "data_manifest.csv", data_audit["inventory"])
    train_loader, val_loader, test_loader = make_dataloaders(model, args)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).cuda()
    )
    optimizer, _ = build_optimizer(model)
    history = []
    best_ranks: dict = {}
    checkpoint_records: dict = {}
    last_record = None
    report_path = output_dir / "training_report.txt"
    report_path.write_text(
        f"Controlled scalp MVP-CBM: {spec.variant_name}\n"
        f"Protocol: {recipe['protocol']}\n"
        "WARNING: Validation and final test both use data-path/test. Results "
        "are diagnostic, not an independent test estimate.\n\n",
        encoding="utf-8",
    )

    for epoch_index in range(FIXED_RECIPE["epochs"]):
        epoch = epoch_index + 1
        schedule_group_lrs(optimizer, epoch_index)
        model.train()
        accumulated = {
            "classification_loss": 0.0,
            "concept_loss": 0.0,
            "sparse_loss": 0.0,
            "total_loss": 0.0,
        }
        sample_count = 0
        progress = tqdm(
            train_loader,
            desc=f"{spec.key.upper()} epoch {epoch}/100 [Train]",
            leave=False,
        )
        for data, labels, concept_targets in progress:
            data = data.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)
            concept_targets = concept_targets.long().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            (
                _logits,
                _concept_logits,
                cls_loss,
                concept_loss,
                sparse_loss,
                total_loss,
            ) = compute_losses(model, data, labels, concept_targets, criterion)
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss for {spec.key} at epoch {epoch}"
                )
            total_loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            sample_count += batch_size
            for key, value in (
                ("classification_loss", cls_loss),
                ("concept_loss", concept_loss),
                ("sparse_loss", sparse_loss),
                ("total_loss", total_loss),
            ):
                accumulated[key] += float(value.item()) * batch_size
            progress.set_postfix(
                cls=f"{cls_loss.item():.4f}",
                concept=f"{concept_loss.item():.4f}",
                total=f"{total_loss.item():.4f}",
            )
        if sample_count == 0:
            raise RuntimeError("Training loader produced no samples")
        train_losses = {
            key: value / sample_count for key, value in accumulated.items()
        }
        val_metrics, _truth, _prediction, _probabilities = evaluate(
            model, val_loader, criterion
        )
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
                )
            },
            "val_tradeoff": tradeoff_score(val_metrics),
        }
        history.append(row)
        write_csv(output_dir / "history.csv", history)

        for key, value in train_losses.items():
            writer.add_scalar(f"Train/{key}", value, epoch)
        for key in (
            "classification_loss",
            "concept_loss",
            "sparse_loss",
            "total_loss",
            "acc",
            "bmac",
            "macro_f1",
            "weighted_f1",
        ):
            writer.add_scalar(f"Val/{key}", val_metrics[key], epoch)
        writer.add_scalar("Val/tradeoff", row["val_tradeoff"], epoch)
        writer.add_scalar("LR/backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/bridge", optimizer.param_groups[1]["lr"], epoch)

        improved = update_retained_checkpoints(
            model,
            val_metrics,
            epoch,
            best_ranks,
            checkpoint_records,
            output_dir,
        )
        last_path = output_dir / "last.pth"
        torch.save(model.state_dict(), last_path)
        last_record = checkpoint_record(
            "last", epoch, val_metrics, last_path, last_path
        )
        write_checkpoint_index(checkpoint_records, last_record, output_dir)

        message = (
            f"Epoch {epoch:03d} | {format_metrics(val_metrics)} | "
            f"retained={','.join(improved) if improved else 'none'} | "
            f"best BMAC={checkpoint_records['best_bmac']['bmac']:.2f}% "
            f"@ epoch {checkpoint_records['best_bmac']['epoch']}"
        )
        print(message)
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    writer.flush()
    writer.close()
    best_record = checkpoint_records["best_bmac"]
    best_path = output_dir / CHECKPOINT_ALIASES["best_bmac"]
    incompatible = model.load_state_dict(load_state(best_path), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict best-checkpoint reload failed: {incompatible}")
    model.cuda()
    best_val_metrics, val_true, val_pred, val_probabilities = evaluate(
        model, val_loader, criterion
    )
    test_metrics, y_true, y_pred, probabilities = evaluate(
        model, test_loader, criterion
    )
    if not (
        np.array_equal(val_true, y_true)
        and np.array_equal(val_pred, y_pred)
        and np.array_equal(val_probabilities, probabilities)
    ):
        raise RuntimeError("Shared validation/test predictions unexpectedly differ")

    save_final_artifacts(
        output_dir,
        args,
        spec,
        recipe,
        data_audit,
        dataset_contract,
        source_audit,
        static_model_contract,
        optimizer_summary,
        model,
        best_record,
        best_val_metrics,
        test_metrics,
        y_true,
        y_pred,
        probabilities,
        test_loader.dataset.image_paths,
        history,
    )
    final_message = (
        f"Completed {spec.variant_name}; best epoch {best_record['epoch']}: "
        f"{format_metrics(test_metrics)}\n"
        f"Standard checkpoint: {best_path}"
    )
    print(final_message)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + final_message + "\n")


def main_for_variant(variant_key: str, entry_source: str) -> None:
    spec = get_variant(variant_key)
    args = parse_args(spec)
    args.entry_source = str(Path(entry_source).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args.tensorboard_dir = str(Path(args.tensorboard_dir).expanduser().resolve())

    source = model_source_path(spec)
    source_hash = sha256_file(source)
    if source_hash.lower() != args.expected_model_sha256.lower():
        raise RuntimeError(
            f"{spec.key} model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, found {source_hash}"
        )
    data_root = Path(args.data_path)
    data_audit = validate_data_layout(data_root)
    dataset_contract = validate_dataset_contract(data_root)
    source_audit = validate_preserved_sources()
    static_model_contract = validate_static_model_contract(spec)
    recipe = resolved_recipe(spec)
    output_dir = Path(args.output_dir)
    output_existed_before = output_dir.exists()
    if not (args.validate_config_only or args.smoke_test) and output_existed_before:
        raise FileExistsError(f"Output collision; refusing to overwrite: {output_dir}")

    audit = {
        "validation_only": args.validate_config_only,
        "smoke_test": args.smoke_test,
        "variant": spec.variant_name,
        "protocol": recipe["protocol"],
        "fold04_meaning": recipe["fold04_meaning"],
        "model_source": str(source),
        "model_source_sha256": source_hash,
        "model_dependency_sources": {
            str(path): sha256_file(path) for path in dependency_sources(spec)
        },
        "dataset_source": str(dataset_source_path()),
        "dataset_source_sha256": sha256_file(dataset_source_path()),
        "entry_source": args.entry_source,
        "trainer_source": str(Path(__file__).resolve()),
        "data_path": str(data_root),
        "data_inventory_sha256": data_audit["inventory_sha256"],
        "class_counts": data_audit["counts"],
        "train_size": data_audit["train_size"],
        "val_size": data_audit["val_size"],
        "test_size": data_audit["test_size"],
        "val_test_overlap": "100%: both map to data-path/test",
        "dataset_contract": dataset_contract,
        "static_model_contract": static_model_contract,
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
            raise RuntimeError("Configuration validation changed the output directory")
        print("Configuration validation completed; no model/output created.")
        return

    model = construct_model(spec, args)
    optimizer, optimizer_summary = build_optimizer(model)
    print(json.dumps({"optimizer_groups": optimizer_summary}, indent=2))
    del optimizer
    if args.smoke_test:
        run_smoke_test(model, spec, args, output_existed_before)
        return
    train(
        model,
        spec,
        args,
        recipe,
        data_audit,
        dataset_contract,
        source_audit,
        static_model_contract,
        optimizer_summary,
    )
