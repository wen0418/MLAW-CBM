# Scalp V5-4 quick start

This is the shortest supported path for training the historical **V5-4** model
on the six-class scalp dataset. In the public API, V5-4 is named
**MLAW-CBM**.

Run every command from the repository root.

## 1. Create the environment

```bash
mamba env create -f environment.yaml
mamba activate cbm
```

The first model construction needs access to the BiomedCLIP and pretrained
ViT weights, either from the local Hugging Face/timm cache or from the network.
An `HF_TOKEN` is optional but avoids anonymous Hub rate limits.

## 2. Arrange the dataset

```text
dataset/scalp/
  train/
    Xerosis/
    Normal/
    Oily-Dandruff/
    Folliculitis/
    Seborrheic-dermatitis/
    Dry-Dandruff/
  test/
    Xerosis/
    Normal/
    Oily-Dandruff/
    Folliculitis/
    Seborrheic-dermatitis/
    Dry-Dandruff/
```

Folder names are case-sensitive. The audited dataset contains 7,645 training
and 3,568 test images. The current controlled protocol intentionally uses
`test/` for both validation and final reporting.

If the dataset is stored elsewhere, pass its absolute path with `--data-path`.

## 3. Validate before training

```bash
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /absolute/path/to/scalp \
  --validate-config-only
```

This checks class order, image counts, concept targets, source hashes, and the
resolved training recipe without constructing a GPU model or creating output.

Then run one real forward/backward batch:

```bash
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /absolute/path/to/scalp \
  --smoke-test
```

The smoke test does not create the formal training directory.

## 4. Start training

```bash
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /absolute/path/to/scalp \
  --gpu 0 \
  --output-dir ./output/scalp/mlaw_v5_4_experiment_001 \
  --tensorboard-dir ./log/scalp/mlaw_v5_4_experiment_001
```

The trainer refuses to overwrite an existing output directory. Use a new name
for every experiment.

## 5. Tune the recipe

The controlled defaults are centralized in
`training/scalp/config.py`:

- `FIXED_RECIPE`: epochs, batch size, learning rates, weight decay,
  `lambda_cpt`, seed, and augmentation;
- `V3_RECIPE`: Top-K, Attribute temperature, initial wavelet scales, and
  numerical epsilon;
- `NEW_V5_RECIPE`: counterfactual route temperature used by V5-4.

Change one factor at a time and use a unique output name. The default model is
the configurable V5-4 class in `model/mlaw_cbm_configurable.py`; no model file
needs to be renamed.

## Verified configuration

The maintained smoke test was run on an RTX 4090 with batch size 2 and passed:

- six-class logits: `[2, 6]`;
- all six concept-head shapes verified;
- all 12 ViT layers generated the guided CLS representation;
- forward, total loss, backward, and selector gradients verified;
- peak allocated CUDA memory: approximately 2.17 GiB.

The normal training batch size remains 64 and requires substantially more
memory than the smoke test.
