"""Folder-based scalp dataset used by the controlled training protocol.

Expected layout::

    DATA_ROOT/
      train/<class name>/*.{jpg,jpeg,png,bmp,tiff}
      test/<class name>/*.{jpg,jpeg,png,bmp,tiff}

Validation intentionally reuses ``test`` to reproduce the audited diagnostic
protocol.  It is therefore not an independent test estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = (
    "Xerosis",
    "Normal",
    "Oily-Dandruff",
    "Folliculitis",
    "Seborrheic-dermatitis",
    "Dry-Dandruff",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
CONCEPT_LABEL_MAP = {
    0: (3, 1, 0, 1, 1, 1),
    1: (0, 0, 0, 0, 0, 0),
    2: (2, 0, 0, 2, 2, 0),
    3: (0, 3, 3, 2, 2, 0),
    4: (3, 2, 0, 2, 2, 1),
    5: (1, 0, 0, 1, 1, 1),
}


class ScalpDataset(Dataset):
    """Deterministic folder-indexed six-class scalp dataset."""

    def __init__(
        self,
        dataset_dir,
        mode="train",
        transforms=None,
        flag=0,
        debug=False,
        config=None,
        return_concept_label=False,
    ):
        del flag, config
        root = Path(dataset_dir)
        if debug:
            root = root / "debug"
        split = "test" if mode in {"val", "test"} else "train"
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing scalp split directory: {split_root}")

        self.mode = mode
        self.dataset_dir = root
        self.transforms = transforms
        self.return_concept_label = return_concept_label
        self.class_names = list(CLASS_NAMES)
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(CLASS_NAMES)
        }
        self.concept_label_map = {
            key: list(value) for key, value in CONCEPT_LABEL_MAP.items()
        }
        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        for class_name in CLASS_NAMES:
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing scalp class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(image_path)
                    self.labels.append(self.class_to_idx[class_name])

        if not self.image_paths:
            raise RuntimeError(f"No images found below {split_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        label = int(self.labels[index])
        if not self.return_concept_label:
            return image, label
        concept_labels = np.asarray(self.concept_label_map[label], dtype=np.int64)
        return image, label, concept_labels


__all__ = ["CLASS_NAMES", "CONCEPT_LABEL_MAP", "ScalpDataset"]
