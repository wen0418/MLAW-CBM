"""Retained-checkpoint policies shared by MLAW-CBM experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


TRADEOFF_WEIGHTS = {"bmac": 0.50, "acc": 0.25, "macro_f1": 0.25}
CHECKPOINT_ALIASES = {
    "best_bmac": "best_bmac.pth",
    "best_acc": "best_acc.pth",
    "best_macro_f1": "best_macro_f1.pth",
    "best_tradeoff": "best_tradeoff.pth",
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
        return tradeoff_score(metrics), float(epoch)
    raise KeyError(policy)


def atomic_relative_symlink(target: Path, alias: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(target, start=alias.parent))
    os.replace(temporary, alias)


def checkpoint_index_payload(records: dict, output_dir: Path) -> dict:
    return {
        "warning": "validation and test are the same Fold04 holdout",
        "standard_report_checkpoint": "best_bmac.pth",
        "tradeoff_formula": "0.50 * BMAC + 0.25 * ACC + 0.25 * Macro-F1",
        "tradeoff_weights": dict(TRADEOFF_WEIGHTS),
        "checkpoint_storage": (
            "one canonical epoch file per unique retained epoch; root aliases "
            "are relative symbolic links; unreferenced superseded files are pruned"
        ),
        "policies": records,
        "unique_retained_epochs": sorted(
            {int(record["epoch"]) for record in records.values()}
        ),
        "output_dir": str(output_dir),
    }


def write_checkpoint_index(records: dict, output_dir: Path) -> None:
    payload = checkpoint_index_payload(records, output_dir)
    (output_dir / "checkpoint_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_retained_checkpoints(
    model,
    metrics: dict,
    epoch: int,
    best_ranks: dict,
    records: dict,
    output_dir: Path,
) -> list[str]:
    ranks = {
        policy: policy_rank(policy, metrics, epoch)
        for policy in CHECKPOINT_ALIASES
    }
    improved = [
        policy
        for policy, rank in ranks.items()
        if policy not in best_ranks or rank > best_ranks[policy]
    ]
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
        records[policy] = {
            "epoch": int(epoch),
            "alias": str(alias),
            "canonical_checkpoint": str(canonical),
            "acc": float(metrics["acc"]),
            "bmac": float(metrics["bmac"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "val_loss": float(metrics["loss"]),
            "tradeoff_score": float(tradeoff_score(metrics)),
            "rank": [float(value) for value in ranks[policy]],
        }

    referenced = {
        Path(record["canonical_checkpoint"]).resolve()
        for record in records.values()
    }
    for candidate in checkpoint_dir.glob("epoch_*.pth"):
        if candidate.resolve() not in referenced:
            candidate.unlink()
    write_checkpoint_index(records, output_dir)
    return improved


__all__ = [
    "CHECKPOINT_ALIASES",
    "TRADEOFF_WEIGHTS",
    "checkpoint_index_payload",
    "tradeoff_score",
    "update_retained_checkpoints",
    "write_checkpoint_index",
]
