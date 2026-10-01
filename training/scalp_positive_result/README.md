# Formal residual-bank scalp MLAW-CBM + per-image VLM concepts

繁體中文完整交接流程（包含另外交付的 VLM label 包、SHA-256、資料對應規則與
執行檢查表）請見 [HANDOFF_ZH_TW.md](HANDOFF_ZH_TW.md)。

This folder reruns the scalp `positive_result` experiment on the formal
residual-bank MLAW-CBM. It keeps the controlled scalp recipe, split, seed, and
disease path unchanged, while replacing class-fixed concept targets with the
same per-image VLM labels used by the earlier experiment. The model contract
explicitly verifies that counterfactuals are constructed as `X - R_band`.

Default target: `inclusive_output_list`:

- `0`: nothing visibly supported;
- `1..S`: the original zero-based state shifted by one;
- six head sizes: `5, 5, 5, 5, 4, 4`.

The disease classifier still consumes the original 22 state activations. The
six learnable `nothing` references are used only by concept cross-entropy.
The original `Sebum_and_Moisture=other` state is retained for architecture and
comparison compatibility even though neither the class map nor VLM CSV selects
it as a target.

## Dataset join

The VLM CSV uses anonymised sequential IDs. Within each split and disease
class, IDs are joined to lexicographically sorted source filenames. The loader
checks every class count, sequential ID, diagnosis, original/shifted target,
hard/inclusive relationship, and declared visibility count before constructing
the model.

Defaults:

```text
images: /home/wen/Desktop/CBM/scalp_dataset/New_scalp
labels: /home/wen/Desktop/CBM/filtered data (VLM check concept label)/scalp_concept
output: output/scalp/scalp_mlaw_cbm_residual_bank_joint_inclusive_k98_fold04_recipe_seed43
```

## Run

From the repository root, with `mamba activate cbm`:

```bash
# CPU/data validation; creates no output directory
bash training/scalp_positive_result/run_mlaw_cbm.sh --validate-config-only

# One real GPU forward/backward batch; creates no formal output directory
bash training/scalp_positive_result/run_mlaw_cbm.sh --smoke-test --gpu 0

# Full 100-epoch joint training
bash training/scalp_positive_result/run_mlaw_cbm.sh --gpu 0
```

The full run refuses to overwrite an existing output directory. Use
`--output-dir` and `--tensorboard-dir` to name another run.

The result directory contains checkpoints, disease metrics, per-Attribute
concept metrics, predictions, runtime, and the exact image-to-VLM-label
manifest used for training and evaluation.
