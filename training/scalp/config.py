"""Single audited configuration for controlled scalp MVP-CBM experiments.

The concept vocabulary mirrors the existing scalp trainers, while the
class-derived targets mirror ``dataset/dataset_scalp.py``.  Training code must
validate the live dataset object against these values before constructing a
model.
"""

from __future__ import annotations


PROTOCOL = "scalp_fold04_recipe_shared_val_test_diagnostic_v1"
DATA_ROOT = "./dataset/scalp"
PATCH_COUNT = 196
HIDDEN_DIM = 768

CLASS_NAMES = (
    "Xerosis",
    "Normal",
    "Oily-Dandruff",
    "Folliculitis",
    "Seborrheic-dermatitis",
    "Dry-Dandruff",
)

CLASS_WEIGHTS = (1.8893, 0.3204, 0.4162, 1.7720, 1.0504, 3.5518)

CONCEPTS = {
    "Flakes_and_Scales": (
        "absent",
        "fine powdery",
        "yellowish greasy",
        "thick waxy plaques",
    ),
    "Erythema_and_Inflammation": (
        "no redness",
        "mild pinkish",
        "diffuse red",
        "perifollicular red",
    ),
    "Follicular_Lesions": (
        "no bumps",
        "keratin plugs",
        "red papules",
        "pus-filled pustules",
    ),
    "Sebum_and_Moisture": (
        "supple sheen",
        "dry dull",
        "excessively oily",
        "other",
    ),
    "Hair_Root_Condition": (
        "clear peripilar spaces",
        "dry loose scales",
        "greasy sebum coating",
    ),
    "Skin_Texture": (
        "smooth intact",
        "rough scaly",
        "scratched damaged",
    ),
}

CONCEPT_LABEL_MAP = {
    0: (3, 1, 0, 1, 1, 1),
    1: (0, 0, 0, 0, 0, 0),
    2: (2, 0, 0, 2, 2, 0),
    3: (0, 3, 3, 2, 2, 0),
    4: (3, 2, 0, 2, 2, 1),
    5: (1, 0, 0, 1, 1, 1),
}

ATTRIBUTE_PROMPTS = {
    "Flakes_and_Scales": (
        "this is a clinical scalp image; focus on the flakes and scales attribute"
    ),
    "Erythema_and_Inflammation": (
        "this is a clinical scalp image; focus on the erythema and inflammation "
        "attribute"
    ),
    "Follicular_Lesions": (
        "this is a clinical scalp image; focus on the follicular lesions attribute"
    ),
    "Sebum_and_Moisture": (
        "this is a clinical scalp image; focus on the sebum and moisture attribute"
    ),
    "Hair_Root_Condition": (
        "this is a clinical scalp image; focus on the hair root condition attribute"
    ),
    "Skin_Texture": (
        "this is a clinical scalp image; focus on the skin texture attribute"
    ),
}

# Equation (6) in MVP-CBM: concatenate every concept state belonging to an
# attribute into one text input, then encode it as a global attribute feature.
PAPER_GLOBAL_ATTRIBUTE_PROMPTS = {
    attribute: (
        f"this is a scalp dermoscopic image, the {attribute} of the scalp "
        f"condition is {' '.join(states)}"
    )
    for attribute, states in CONCEPTS.items()
}

EXPECTED_SPLIT_SIZES = {"train": 7645, "test": 3568}
DATALOADER_SEEDS = {"train": 43, "val": 44, "test": 45}

FIXED_RECIPE = {
    "epochs": 100,
    "batch_size": 64,
    "warmup_epoch": 5,
    "optimizer": "AdamW",
    "backbone_lr": 1e-5,
    "bridge_lr": 1e-4,
    "weight_decay": 0.01,
    "lambda_cpt": 2.5,
    "model_seed": 43,
    "drop_last_train": True,
    "model_name": "biomedclip",
    "criterion": "class-weighted cross entropy",
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
    "evaluation_preprocessing": "BiomedCLIP pretrained evaluation preprocessing",
    "dataloader_seeds": DATALOADER_SEEDS,
    "torch_deterministic": True,
    "deterministic_algorithms": "warn_only",
    "cudnn_benchmark": False,
    "tf32": False,
}

V3_RECIPE = {
    "attribute_wavelet_top_k": 98,
    "attribute_temperature": 0.07,
    "initial_high_feature_scale": 0.1,
    "initial_high_score_weight": 1.0,
    "attribute_wavelet_eps": 1e-6,
}

# new V5 changes only how the V3 high-frequency residual is constructed.  Its
# final semantic/high-frequency score keeps the same V3 scale parameters.
NEW_V5_RECIPE = {
    **V3_RECIPE,
    "counterfactual_route_temperature": 0.05,
}

TRADEOFF_WEIGHTS = {"bmac": 0.50, "acc": 0.25, "macro_f1": 0.25}
CHECKPOINT_ALIASES = {
    "best_bmac": "best_bmac.pth",
    "best_acc": "best_acc.pth",
    "best_macro_f1": "best_macro_f1.pth",
    "best_tradeoff": "best_tradeoff.pth",
}
