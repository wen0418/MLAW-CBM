#!/usr/bin/env python3
"""Measure how much of V5-4's wavelet residual survives patch pooling.

For every ViT layer and Fold04 holdout image, this script extracts

    X : the 196 original patch tokens
    R : alpha * H*, V5-4's counterfactually selected wavelet residual
    Y : X + R

and compares three summaries of X and Y:

    AP(7) : the baseline AdaptiveAvgPool1d(7) stripe representation
    mean  : one signed global mean over all 196 patches
    std   : one global population standard deviation over all 196 patches

AP and mean are linear, so AP(Y)-AP(X)=AP(R) and
mean(Y)-mean(X)=mean(R) are the exact signed wavelet components that survive.
Standard deviation is nonlinear; std(Y)-std(X) is therefore reported as a
response to the residual, not as a retained residual.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_isic2018_baseline_cv10 as baseline
import train_isic2018_fold04_attribute_wavelet_last_v5_4 as v5_4_trainer
from model.mvpcbm_attribute_wavelet_last_new_v5 import (
    DEFAULT_ATTRIBUTE_TEMPERATURE,
    DEFAULT_EPS,
    DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
    DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
    DEFAULT_ROUTE_TEMPERATURE,
    DEFAULT_TOP_K,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "output/isic2018/"
    "baseline_cv10_attribute_wavelet_last_v5_4_k98_imglevel_fold04_seed43"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "best_bmac.pth"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "wavelet_pooling_analysis"
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset/ISIC2018"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "splits/isic2018_baseline_cv10_imagelevel"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--fold", type=int, choices=range(10), default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Analyze only the first N holdout images (useful for a smoke test).",
    )
    parser.add_argument(
        "--save-feature-samples",
        type=int,
        default=16,
        help="Save pooled tensors for the first N analyzed images.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.save_feature_samples < 0:
        parser.error("--save-feature-samples cannot be negative")
    return args


def sample_rms(value: torch.Tensor) -> torch.Tensor:
    """Return one RMS value per sample, reducing every other dimension."""

    return value.float().flatten(1).square().mean(dim=1).sqrt()


def sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float().flatten(1), right.float().flatten(1))


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.squared_sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().double().flatten().cpu()
        self.sums[name] += float(values.sum())
        self.squared_sums[name] += float(values.square().sum())
        self.counts[name] += int(values.numel())

    def summary(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for name in sorted(self.sums):
            count = self.counts[name]
            mean = self.sums[name] / count
            variance = max(self.squared_sums[name] / count - mean * mean, 0.0)
            result[name] = {
                "mean": mean,
                "std_across_samples": math.sqrt(variance),
                "count": count,
            }
        return result


def add_metrics(
    accumulator: MetricAccumulator,
    x: torch.Tensor,
    residual: torch.Tensor,
    enhanced: torch.Tensor,
    ap_x: torch.Tensor,
    ap_residual: torch.Tensor,
    ap_enhanced: torch.Tensor,
    mean_x: torch.Tensor,
    mean_residual: torch.Tensor,
    mean_enhanced: torch.Tensor,
    std_x: torch.Tensor,
    std_residual: torch.Tensor,
    std_enhanced: torch.Tensor,
) -> None:
    raw_rms = sample_rms(residual).clamp_min(EPS)
    x_rms = sample_rms(x).clamp_min(EPS)
    ap_x_rms = sample_rms(ap_x).clamp_min(EPS)
    mean_x_rms = sample_rms(mean_x).clamp_min(EPS)
    std_x_rms = sample_rms(std_x).clamp_min(EPS)
    std_change = std_enhanced - std_x
    mean_abs_residual = residual.abs().mean(dim=1)

    values = {
        "raw_wavelet_rms": raw_rms,
        "raw_original_rms": x_rms,
        "raw_wavelet_to_original_ratio": raw_rms / x_rms,
        "ap_wavelet_rms": sample_rms(ap_residual),
        "ap_signed_amplitude_retention": sample_rms(ap_residual) / raw_rms,
        "ap_wavelet_to_original_ratio": sample_rms(ap_residual) / ap_x_rms,
        "ap_original_enhanced_cosine": sample_cosine(ap_x, ap_enhanced),
        "mean_wavelet_rms": sample_rms(mean_residual),
        "mean_signed_amplitude_retention": sample_rms(mean_residual) / raw_rms,
        "mean_wavelet_to_original_ratio": sample_rms(mean_residual) / mean_x_rms,
        "mean_original_enhanced_cosine": sample_cosine(mean_x, mean_enhanced),
        "std_wavelet_rms": sample_rms(std_residual),
        "std_wavelet_amplitude_ratio": sample_rms(std_residual) / raw_rms,
        "std_response_rms": sample_rms(std_change),
        "std_response_amplitude_ratio": sample_rms(std_change) / raw_rms,
        "std_response_to_original_ratio": sample_rms(std_change) / std_x_rms,
        "std_original_enhanced_cosine": sample_cosine(std_x, std_enhanced),
        "mean_abs_wavelet_rms": sample_rms(mean_abs_residual),
        "mean_abs_wavelet_amplitude_ratio": (
            sample_rms(mean_abs_residual) / raw_rms
        ),
    }
    for name, tensor in values.items():
        accumulator.add(name, tensor)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    source_path = v5_4_trainer.model_source_path()
    config = SimpleNamespace(
        expected_model_sha256=baseline.sha256_file(source_path),
        attribute_wavelet_top_k=DEFAULT_TOP_K,
        attribute_temperature=DEFAULT_ATTRIBUTE_TEMPERATURE,
        counterfactual_route_temperature=DEFAULT_ROUTE_TEMPERATURE,
        initial_high_feature_scale=DEFAULT_INITIAL_HIGH_FEATURE_SCALE,
        initial_high_score_weight=DEFAULT_INITIAL_HIGH_SCORE_WEIGHT,
        attribute_wavelet_eps=DEFAULT_EPS,
        data_path=str(Path(args.data_path).resolve()),
        train_workers=args.workers,
        eval_workers=args.workers,
        gpu=args.gpu,
    )
    model = v5_4_trainer.construct_model(config)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    return model


def make_holdout_loader(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[DataLoader, object, dict, dict]:
    fold_frame, metadata, fold_audit = baseline.load_fold_audit(
        Path(args.manifest_dir).resolve(), args.fold
    )
    holdout_frame = fold_frame[fold_frame["role"] == "holdout"].reset_index(
        drop=True
    )
    if args.max_samples is not None:
        holdout_frame = holdout_frame.iloc[: args.max_samples].copy()
    _, eval_transform = baseline.make_transforms(model)
    dataset = baseline.ManifestDataset(
        Path(args.data_path).resolve(), holdout_frame, copy.deepcopy(eval_transform)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=baseline.seed_worker,
        generator=torch.Generator().manual_seed(
            baseline.FIXED_RECIPE["model_seed"] + 2
        ),
    )
    return loader, holdout_frame, metadata, fold_audit


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_layer_summary(
    layer: int, summary: dict[str, dict[str, float | int]]
) -> dict:
    row: dict[str, float | int] = {"layer": layer}
    for metric, values in summary.items():
        row[metric] = values["mean"]
        row[f"{metric}_sample_std"] = values["std_across_samples"]
    return row


def save_plot(layer_rows: list[dict], output_path: Path) -> None:
    layers = np.asarray([row["layer"] + 1 for row in layer_rows])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)

    axes[0].plot(
        layers,
        100 * np.asarray([row["ap_signed_amplitude_retention"] for row in layer_rows]),
        marker="o",
        label="AP(7): signed residual retained",
    )
    axes[0].plot(
        layers,
        100
        * np.asarray([row["mean_signed_amplitude_retention"] for row in layer_rows]),
        marker="o",
        label="Global mean: signed residual retained",
    )
    axes[0].plot(
        layers,
        100 * np.asarray([row["std_response_amplitude_ratio"] for row in layer_rows]),
        marker="o",
        label="Global std: response to residual",
    )
    axes[0].plot(
        layers,
        100
        * np.asarray(
            [row["mean_abs_wavelet_amplitude_ratio"] for row in layer_rows]
        ),
        marker="o",
        linestyle="--",
        label="Mean |R|: magnitude reference",
    )
    axes[0].set(
        xlabel="ViT block",
        ylabel="Amplitude relative to raw R (%)",
        title="How much wavelet signal remains after pooling?",
        xticks=layers,
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        layers,
        100 * np.asarray([row["ap_wavelet_to_original_ratio"] for row in layer_rows]),
        marker="o",
        label="AP(7)",
    )
    axes[1].plot(
        layers,
        100
        * np.asarray([row["mean_wavelet_to_original_ratio"] for row in layer_rows]),
        marker="o",
        label="Global mean",
    )
    axes[1].plot(
        layers,
        100
        * np.asarray([row["std_response_to_original_ratio"] for row in layer_rows]),
        marker="o",
        label="Global std response",
    )
    axes[1].set(
        xlabel="ViT block",
        ylabel="Change relative to pooled original (%)",
        title="Visibility of the wavelet change in each representation",
        xticks=layers,
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires CUDA, like the V5-4 trainer")

    baseline.set_seed(baseline.FIXED_RECIPE["model_seed"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    loader, holdout_frame, metadata, fold_audit = make_holdout_loader(model, args)
    num_layers = len(model.model.visual.trunk.blocks)

    per_layer = [MetricAccumulator() for _ in range(num_layers)]
    aggregate = MetricAccumulator()
    ap_slots = [MetricAccumulator() for _ in range(num_layers)]
    saved_batches: dict[str, list[torch.Tensor]] = defaultdict(list)
    saved_count = 0
    max_save = min(args.save_feature_samples, len(loader.dataset))
    exact_ap_max_error = 0.0
    exact_mean_max_error = 0.0

    with torch.inference_mode():
        for images, labels, concept_targets in tqdm(loader, desc="V5-4 pooling analysis"):
            images = images.float().cuda(non_blocking=True)
            model.visual_features.clear()
            model.model(images, None)
            if len(model.visual_features) != num_layers:
                raise RuntimeError(
                    f"Expected {num_layers} hooked layers, got "
                    f"{len(model.visual_features)}"
                )

            save_now = min(max_save - saved_count, images.shape[0])
            if save_now > 0:
                saved_batches["labels"].append(labels[:save_now].clone())
                saved_batches["concept_targets"].append(
                    concept_targets[:save_now].clone()
                )
            batch_layer_features: dict[str, list[torch.Tensor]] = defaultdict(list)

            for layer_index, block_tokens in enumerate(model.visual_features):
                x = block_tokens[:, 1:, :].float()
                _, details = model.build_attribute_wavelet_cls(
                    x, layer_index=layer_index, return_details=True
                )
                residual = (
                    details["high_feature_scale"]
                    * details["filtered_high_tokens"].float()
                )
                enhanced = x + residual

                ap_x = F.adaptive_avg_pool1d(x.transpose(1, 2), 7).transpose(1, 2)
                ap_residual = F.adaptive_avg_pool1d(
                    residual.transpose(1, 2), 7
                ).transpose(1, 2)
                ap_enhanced = F.adaptive_avg_pool1d(
                    enhanced.transpose(1, 2), 7
                ).transpose(1, 2)
                mean_x = x.mean(dim=1)
                mean_residual = residual.mean(dim=1)
                mean_enhanced = enhanced.mean(dim=1)
                std_x = x.std(dim=1, unbiased=False)
                std_residual = residual.std(dim=1, unbiased=False)
                std_enhanced = enhanced.std(dim=1, unbiased=False)

                ap_error = float((ap_enhanced - ap_x - ap_residual).abs().max())
                mean_error = float(
                    (mean_enhanced - mean_x - mean_residual).abs().max()
                )
                exact_ap_max_error = max(exact_ap_max_error, ap_error)
                exact_mean_max_error = max(exact_mean_max_error, mean_error)
                # Pooling X+R separately changes float32 accumulation order,
                # so an exact algebraic identity can differ by a few ulps.
                if ap_error > 1.0e-4 or mean_error > 1.0e-4:
                    raise RuntimeError(
                        "Linearity check failed: "
                        f"layer={layer_index}, AP error={ap_error}, "
                        f"mean error={mean_error}"
                    )

                metric_args = (
                    x,
                    residual,
                    enhanced,
                    ap_x,
                    ap_residual,
                    ap_enhanced,
                    mean_x,
                    mean_residual,
                    mean_enhanced,
                    std_x,
                    std_residual,
                    std_enhanced,
                )
                add_metrics(per_layer[layer_index], *metric_args)
                add_metrics(aggregate, *metric_args)

                raw_rms = sample_rms(residual).clamp_min(EPS)
                for slot in range(ap_residual.shape[1]):
                    ap_slots[layer_index].add(
                        f"slot_{slot + 1}_signed_amplitude_retention",
                        sample_rms(ap_residual[:, slot, :]) / raw_rms,
                    )

                if save_now > 0:
                    feature_values = {
                        "ap_original": ap_x,
                        "ap_enhanced": ap_enhanced,
                        "ap_wavelet": ap_residual,
                        "mean_original": mean_x,
                        "mean_enhanced": mean_enhanced,
                        "mean_wavelet": mean_residual,
                        "std_original": std_x,
                        "std_enhanced": std_enhanced,
                        "std_wavelet": std_residual,
                        "std_change": std_enhanced - std_x,
                        "mean_abs_wavelet": residual.abs().mean(dim=1),
                    }
                    for name, value in feature_values.items():
                        batch_layer_features[name].append(
                            value[:save_now].detach().cpu().to(torch.float16)
                        )

                del details

            if save_now > 0:
                for name, layer_values in batch_layer_features.items():
                    saved_batches[name].append(torch.stack(layer_values, dim=1))
                saved_count += save_now
            model.visual_features.clear()

    layer_summaries = [item.summary() for item in per_layer]
    layer_rows = [
        flatten_layer_summary(index, summary)
        for index, summary in enumerate(layer_summaries)
    ]
    slot_rows = []
    for layer_index, slot_accumulator in enumerate(ap_slots):
        row: dict[str, float | int] = {"layer": layer_index}
        for name, values in slot_accumulator.summary().items():
            row[name] = values["mean"]
            row[f"{name}_sample_std"] = values["std_across_samples"]
        slot_rows.append(row)

    aggregate_summary = aggregate.summary()
    payload = {
        "experiment": "V5-4 wavelet pooling cancellation analysis",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": baseline.sha256_file(Path(args.checkpoint).resolve()),
        "model_source": str(v5_4_trainer.model_source_path()),
        "model_source_sha256": baseline.sha256_file(
            v5_4_trainer.model_source_path()
        ),
        "fold": args.fold,
        "sample_count": len(loader.dataset),
        "full_holdout_count": int(fold_audit["holdout_size"]),
        "manifest_sha256": fold_audit["manifest_sha256"],
        "split_metadata_sha256": baseline.sha256_file(
            Path(args.manifest_dir).resolve() / "split_metadata.json"
        ),
        "num_layers": num_layers,
        "num_patches": 196,
        "feature_dimension": 768,
        "ap_output_slots": 7,
        "residual_definition": "R = softplus(s_l) * H-star",
        "enhanced_definition": "Y = X + R (pre-LayerNorm V5-4 concept input)",
        "definitions": {
            "signed_amplitude_retention": (
                "RMS of the exact linearly pooled residual divided by RMS of "
                "the raw patch residual R; lower means more signed cancellation"
            ),
            "wavelet_to_original_ratio": (
                "RMS of the pooled residual divided by RMS of the corresponding "
                "pooled original X"
            ),
            "std_response_amplitude_ratio": (
                "RMS(std(X+R)-std(X)) / RMS(R); std is nonlinear, so this is "
                "a response rather than exact retained R"
            ),
            "mean_abs_wavelet_amplitude_ratio": (
                "RMS(mean(|R|)) / RMS(R), a sign-free magnitude reference"
            ),
        },
        "linearity_checks": {
            "ap_max_abs_error": exact_ap_max_error,
            "mean_max_abs_error": exact_mean_max_error,
        },
        "aggregate_over_all_images_and_layers": aggregate_summary,
        "per_layer": layer_summaries,
        "data_provenance": {
            "data_path": str(Path(args.data_path).resolve()),
            "manifest_dir": str(Path(args.manifest_dir).resolve()),
            "fold_audit": fold_audit,
            "metadata_seed": metadata.get("seed"),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_csv(output_dir / "per_layer.csv", layer_rows)
    save_csv(output_dir / "ap_slot_retention.csv", slot_rows)
    save_plot(layer_rows, output_dir / "retention_plot.png")

    if saved_count:
        tensor_payload = {
            "description": (
                "First N holdout samples. Tensor layout is [sample, layer, ...]. "
                "AP tensors are [N,L,7,768]; mean/std tensors are [N,L,768]."
            ),
            "sample_indices": torch.arange(saved_count),
            "image_ids": holdout_frame.iloc[:saved_count]["image"].astype(str).tolist(),
            "labels": torch.cat(saved_batches.pop("labels"), dim=0),
            "concept_targets": torch.cat(
                saved_batches.pop("concept_targets"), dim=0
            ),
            **{
                name: torch.cat(chunks, dim=0)
                for name, chunks in saved_batches.items()
            },
        }
        torch.save(tensor_payload, output_dir / "sample_pooled_features.pt")

    concise = {
        key: round(float(values["mean"]), 8)
        for key, values in aggregate_summary.items()
        if key
        in {
            "raw_wavelet_to_original_ratio",
            "ap_signed_amplitude_retention",
            "mean_signed_amplitude_retention",
            "std_response_amplitude_ratio",
            "mean_abs_wavelet_amplitude_ratio",
            "ap_wavelet_to_original_ratio",
            "mean_wavelet_to_original_ratio",
            "std_response_to_original_ratio",
        }
    }
    print(json.dumps({"output_dir": str(output_dir), **concise}, indent=2))


if __name__ == "__main__":
    main()
