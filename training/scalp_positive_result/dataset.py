"""Audited image-to-VLM-label join for scalp joint concept training.

The VLM export anonymises image names as sequential IDs within each split and
disease class.  The IDs were produced from lexicographically sorted source
filenames.  This module reconstructs that mapping and refuses to continue if
any class count, ID sequence, diagnosis, or label invariant has changed.
"""

from __future__ import annotations

import ast
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from training.scalp.config import CLASS_NAMES, CONCEPT_LABEL_MAP, CONCEPTS


NOTHING_LABEL = 0
SUPPORTED_LABEL_COLUMNS = ("inclusive_output_list", "hard_output_list")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
CSV_FILENAMES = {"train": "training.csv", "test": "test.csv"}


@dataclass(frozen=True)
class PseudoLabelRecord:
    image_id: str
    diagnosis: str
    disease_name: str
    original_zero_based: tuple[int, ...]
    shifted: tuple[int, ...]
    hard: tuple[int, ...]
    inclusive: tuple[int, ...]
    target: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(value: str, field: str, image_id: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid {field} for {image_id}: {value!r}") from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(CONCEPTS):
        raise ValueError(
            f"{field} for {image_id} must contain {len(CONCEPTS)} integers"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError(f"{field} for {image_id} contains a non-integer")
    return tuple(int(item) for item in parsed)


def _diagnosis_token(class_name: str) -> str:
    return class_name.upper().replace("-", "_").replace(" ", "_")


def _expected_image_id(split: str, class_name: str, index: int) -> str:
    return f"SCALP_{split.upper()}_{_diagnosis_token(class_name)}_{index:05d}"


def audit_concept_vocabulary(path: Path) -> dict:
    """Verify the VLM vocabulary against the original scalp architecture."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing scalp concept vocabulary: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"concept_id", "group_id", "group_name", "concept"}
    if not rows or required - set(rows[0]):
        raise RuntimeError("Scalp concept vocabulary is empty or malformed")
    grouped: dict[str, list[str]] = {}
    group_names: dict[str, str] = {}
    for row in rows:
        group_id = row["group_id"].strip()
        grouped.setdefault(group_id, []).append(row["concept"].strip())
        group_names[group_id] = row["group_name"].strip()
    expected_group_ids = [f"A{index}" for index in range(1, len(CONCEPTS) + 1)]
    if list(grouped) != expected_group_ids:
        raise RuntimeError(
            f"VLM vocabulary group order changed: {list(grouped)}"
        )

    missing_original_states = {}
    for attribute_index, ((attribute_name, states), group_id) in enumerate(
        zip(CONCEPTS.items(), expected_group_ids)
    ):
        csv_states = tuple(grouped[group_id])
        if csv_states != tuple(states[: len(csv_states)]):
            raise RuntimeError(
                f"VLM vocabulary states differ from original {attribute_name}: "
                f"{csv_states} is not a prefix of {tuple(states)}"
            )
        missing = tuple(states[len(csv_states) :])
        if missing:
            used_targets = {
                int(targets[attribute_index])
                for targets in CONCEPT_LABEL_MAP.values()
            }
            missing_indices = set(range(len(csv_states), len(states)))
            if used_targets & missing_indices:
                raise RuntimeError(
                    f"VLM vocabulary omits used original states for {attribute_name}"
                )
            missing_original_states[attribute_name] = list(missing)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "group_ids": expected_group_ids,
        "group_names": group_names,
        "vlm_state_counts": {
            attribute_name: len(grouped[group_id])
            for attribute_name, group_id in zip(CONCEPTS, expected_group_ids)
        },
        "original_model_state_counts": {
            name: len(states) for name, states in CONCEPTS.items()
        },
        "unused_original_states_absent_from_vlm_vocabulary": (
            missing_original_states
        ),
        "missing_states_verified_unused_by_class_map": True,
    }


def load_pseudo_label_csv(
    path: Path,
    split: str,
    label_column: str,
) -> tuple[dict[str, PseudoLabelRecord], dict]:
    """Load one VLM CSV and exhaustively verify its label semantics."""

    if split not in CSV_FILENAMES:
        raise ValueError(f"Unsupported split: {split}")
    if label_column not in SUPPORTED_LABEL_COLUMNS:
        raise ValueError(f"Unsupported concept label column: {label_column}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing scalp pseudo-label CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "diagnosis",
        "disease_name",
        "original_zero_based_list",
        "shifted_input_list",
        "hard_output_list",
        "inclusive_output_list",
        "num_strong",
        "num_weak_visible",
        "num_not_visibly_supported",
        "num_inclusive",
    }
    if not rows:
        raise RuntimeError(f"Pseudo-label CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Pseudo-label CSV lacks columns: {sorted(missing)}")

    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    head_sizes = tuple(len(states) + 1 for states in CONCEPTS.values())
    records: dict[str, PseudoLabelRecord] = {}
    class_counts: Counter[str] = Counter()
    label_counts = [Counter() for _ in CONCEPTS]
    strong_total = 0
    weak_total = 0
    inclusive_total = 0
    all_nothing = 0

    for row_index, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        disease_name = row["disease_name"].strip()
        diagnosis = row["diagnosis"].strip()
        if not image_id or image_id in records:
            raise RuntimeError(
                f"Pseudo-label image_id is blank or duplicated at row {row_index}"
            )
        if disease_name not in class_to_id:
            raise RuntimeError(
                f"Unknown disease_name {disease_name!r} for {image_id}"
            )
        expected_diagnosis = _diagnosis_token(disease_name)
        if diagnosis != expected_diagnosis:
            raise RuntimeError(
                f"Diagnosis token mismatch for {image_id}: "
                f"{diagnosis!r} != {expected_diagnosis!r}"
            )

        original = _parse_int_list(
            row["original_zero_based_list"],
            "original_zero_based_list",
            image_id,
        )
        shifted = _parse_int_list(
            row["shifted_input_list"], "shifted_input_list", image_id
        )
        hard = _parse_int_list(
            row["hard_output_list"], "hard_output_list", image_id
        )
        inclusive = _parse_int_list(
            row["inclusive_output_list"], "inclusive_output_list", image_id
        )
        expected_original = tuple(CONCEPT_LABEL_MAP[class_to_id[disease_name]])
        if original != expected_original:
            raise RuntimeError(
                f"Original concept labels disagree with {disease_name} for "
                f"{image_id}: {original} != {expected_original}"
            )
        expected_shifted = tuple(value + 1 for value in original)
        if shifted != expected_shifted:
            raise RuntimeError(f"Shifted labels are invalid for {image_id}")
        for attribute_index, expected_value in enumerate(expected_shifted):
            if hard[attribute_index] not in (NOTHING_LABEL, expected_value):
                raise RuntimeError(f"Invalid hard label for {image_id}")
            if inclusive[attribute_index] not in (NOTHING_LABEL, expected_value):
                raise RuntimeError(f"Invalid inclusive label for {image_id}")
            if hard[attribute_index] and hard[attribute_index] != inclusive[
                attribute_index
            ]:
                raise RuntimeError(
                    f"Hard labels are not a subset of inclusive labels for {image_id}"
                )

        num_strong = sum(value != NOTHING_LABEL for value in hard)
        num_inclusive = sum(value != NOTHING_LABEL for value in inclusive)
        num_weak = num_inclusive - num_strong
        if num_strong != int(row["num_strong"]):
            raise RuntimeError(f"num_strong mismatch for {image_id}")
        if num_weak != int(row["num_weak_visible"]):
            raise RuntimeError(f"num_weak_visible mismatch for {image_id}")
        if num_inclusive != int(row["num_inclusive"]):
            raise RuntimeError(f"num_inclusive mismatch for {image_id}")
        if len(CONCEPTS) - num_inclusive != int(
            row["num_not_visibly_supported"]
        ):
            raise RuntimeError(
                f"num_not_visibly_supported mismatch for {image_id}"
            )

        target = inclusive if label_column == "inclusive_output_list" else hard
        for attribute_index, value in enumerate(target):
            if not 0 <= value < head_sizes[attribute_index]:
                raise RuntimeError(
                    f"Target {value} lies outside head {attribute_index} range "
                    f"[0,{head_sizes[attribute_index]}) for {image_id}"
                )
            label_counts[attribute_index][value] += 1
        records[image_id] = PseudoLabelRecord(
            image_id=image_id,
            diagnosis=diagnosis,
            disease_name=disease_name,
            original_zero_based=original,
            shifted=shifted,
            hard=hard,
            inclusive=inclusive,
            target=target,
        )
        class_counts[disease_name] += 1
        strong_total += num_strong
        weak_total += num_weak
        inclusive_total += num_inclusive
        all_nothing += int(all(value == NOTHING_LABEL for value in target))

    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "encoding": "utf-8-sig",
        "split": split,
        "row_count": len(rows),
        "unique_image_ids": len(records),
        "label_column": label_column,
        "attribute_order": list(CONCEPTS),
        "head_sizes": dict(zip(CONCEPTS, head_sizes)),
        "label_encoding": "0=nothing; nonzero=original_zero_based+1",
        "class_counts": dict(class_counts),
        "target_counts": {
            name: {
                str(label): int(count)
                for label, count in sorted(label_counts[index].items())
            }
            for index, name in enumerate(CONCEPTS)
        },
        "all_nothing_rows": all_nothing,
        "strong_positive_total": strong_total,
        "weak_positive_total": weak_total,
        "inclusive_positive_total": inclusive_total,
        "all_rows_passed_label_invariants": True,
    }
    return records, audit


def build_joined_manifests(
    data_root: Path,
    pseudo_label_dir: Path,
    label_column: str,
) -> tuple[dict[str, list[dict]], dict]:
    """Pair sequential VLM IDs with sorted images, per split and class."""

    data_root = data_root.resolve()
    pseudo_label_dir = pseudo_label_dir.resolve()
    vocabulary_audit = audit_concept_vocabulary(
        pseudo_label_dir / "scalp_concept_vocabulary.csv"
    )
    manifests: dict[str, list[dict]] = {}
    csv_audits = {}
    mapping_digest = hashlib.sha256()
    mapping_examples = {}

    for split, csv_filename in CSV_FILENAMES.items():
        records, csv_audit = load_pseudo_label_csv(
            pseudo_label_dir / csv_filename,
            split,
            label_column,
        )
        csv_audits[split] = csv_audit
        entries = []
        used_ids = set()
        examples = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = data_root / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing scalp class directory: {class_dir}")
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            expected_count = csv_audit["class_counts"].get(class_name, 0)
            if len(paths) != expected_count:
                raise RuntimeError(
                    f"{split}/{class_name} image/CSV count mismatch: "
                    f"{len(paths)} != {expected_count}"
                )
            for one_based_index, image_path in enumerate(paths, start=1):
                image_id = _expected_image_id(split, class_name, one_based_index)
                record = records.get(image_id)
                if record is None:
                    raise RuntimeError(
                        f"Missing expected sequential VLM ID {image_id}"
                    )
                if record.disease_name != class_name:
                    raise RuntimeError(
                        f"Disease mismatch for {image_id}: "
                        f"{record.disease_name} != {class_name}"
                    )
                relative_path = image_path.relative_to(data_root)
                entry = {
                    "split": split,
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "relative_path": str(relative_path),
                    "class_id": class_id,
                    "class_name": class_name,
                    "concept_target": record.target,
                    "original_zero_based": record.original_zero_based,
                    "shifted": record.shifted,
                    "hard": record.hard,
                    "inclusive": record.inclusive,
                }
                entries.append(entry)
                used_ids.add(image_id)
                mapping_digest.update(
                    (
                        f"{split}\t{image_id}\t{relative_path}\t{class_id}\t"
                        f"{','.join(map(str, record.target))}\n"
                    ).encode("utf-8")
                )
            if paths:
                examples.extend(
                    [
                        {
                            "image_id": _expected_image_id(split, class_name, 1),
                            "relative_path": str(paths[0].relative_to(data_root)),
                        },
                        {
                            "image_id": _expected_image_id(
                                split, class_name, len(paths)
                            ),
                            "relative_path": str(paths[-1].relative_to(data_root)),
                        },
                    ]
                )
        unused = set(records) - used_ids
        if unused:
            raise RuntimeError(
                f"{split} CSV contains {len(unused)} unmapped IDs; "
                f"examples={sorted(unused)[:3]}"
            )
        manifests[split] = entries
        mapping_examples[split] = examples

    audit = {
        "mapping_strategy": (
            "within each split and disease class, VLM sequential IDs map to "
            "lexicographically sorted source image filenames"
        ),
        "mapping_key_format": (
            "SCALP_{SPLIT}_{UPPERCASE_DISEASE_WITH_UNDERSCORES}_{1_BASED:05d}"
        ),
        "label_column": label_column,
        "split_sizes": {key: len(value) for key, value in manifests.items()},
        "csv_audits": csv_audits,
        "concept_vocabulary_audit": vocabulary_audit,
        "mapping_sha256": mapping_digest.hexdigest(),
        "mapping_examples_first_last_per_class": mapping_examples,
        "count_class_id_and_label_invariants_verified": True,
    }
    return manifests, audit


class ScalpPositiveResultDataset(Dataset):
    """Manifest-backed scalp dataset returning per-image concept targets."""

    def __init__(self, entries: list[dict], transforms=None):
        if not entries:
            raise RuntimeError("Scalp positive-result dataset is empty")
        self.entries = list(entries)
        self.transforms = transforms
        self.image_paths = [Path(entry["image_path"]) for entry in entries]
        self.labels = [int(entry["class_id"]) for entry in entries]
        self.image_ids = [str(entry["image_id"]) for entry in entries]
        self.concept_targets = [tuple(entry["concept_target"]) for entry in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        with Image.open(entry["image_path"]) as handle:
            image = handle.convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return (
            image,
            int(entry["class_id"]),
            np.asarray(entry["concept_target"], dtype=np.int64),
        )


__all__ = [
    "NOTHING_LABEL",
    "SUPPORTED_LABEL_COLUMNS",
    "ScalpPositiveResultDataset",
    "audit_concept_vocabulary",
    "build_joined_manifests",
    "load_pseudo_label_csv",
]
