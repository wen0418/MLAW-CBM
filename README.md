# MLAW-CBM

**Multi-Layer Attribute–Wavelet Guided Concept Bottleneck Model** for skin
lesion and scalp image analysis.

MLAW-CBM combines multi-layer ViT patch representations, wavelet-enhanced
patch evidence, Attribute-guided high-frequency CLS aggregation, and an
interpretable concept bottleneck. The formal model keeps the V5-4 architecture
while constructing its counterfactuals from a vectorized residual bank as
`X - R_band`. The former direct-IDWT implementation is preserved as
**MLAWCBM_res**, and **V5-7** remains the **MLAW-CBM-Energy** ablation.

> Research status: this repository currently reports controlled diagnostic
> experiments. The ISIC2018 Fold04 holdout and the scalp test split are reused
> for checkpoint selection and final reporting, so the numbers below are not
> independent test estimates.

## Method at a glance

For every ViT layer, MLAW-CBM:

1. decomposes patch features with a fixed one-level Haar transform;
2. reconstructs an LH/HL/HH residual bank and uses Attribute queries to score
   the counterfactuals `X - R_band`;
3. aggregates the selected patches into a wavelet-guided CLS representation;
4. sends the pre-LayerNorm wavelet-enhanced patches through the original
   fixed-bin concept pooling path; and
5. aggregates concept evidence across layers before disease classification.

The fixed-bin adaptive pooling operation is inherited from the MVP-CBM
baseline. It is not presented as the contribution; the contribution under
study is the multi-layer Attribute/wavelet guidance and how wavelet patch
evidence is preserved for both global preference and concept scoring.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the exact data flow and
the Energy ablation.

## Repository layout

```text
model/
  mlaw_cbm.py                  formal residual-bank MLAW-CBM API
  mlaw_cbm_res.py              preserved direct-IDWT MLAW-CBM
  mlaw_cbm_energy.py           stable public API for the Energy ablation (V5-7)
  mlaw_cbm_configurable.py     dataset-configurable skin/scalp variants
training/
  isic2018/                    audited Fold04 launchers
  scalp/                       controlled six-class scalp protocol
  two_stage/                   concept-first sequential training
  concept_supervision/         per-image concept-supervision experiments
splits/                        versioned ISIC2018 split manifests
reference_results/             lightweight baseline references; no weights
results/                       compact result tables
docs/                          architecture notes
```

## Environment

```bash
mamba env create -f environment.yaml
mamba activate cbm
```

The training code expects NVIDIA CUDA. Datasets and pretrained/checkpoint
weights are intentionally not included in Git.

## ISIC2018 Fold04

Place ISIC2018 under `dataset/ISIC2018`, or override `--data-path`.

```bash
# Validate paths, hashes, split, and recipe without starting training
bash training/isic2018/run_mlaw_cbm_fold04.sh --validate-config-only

# Main paper model
bash training/isic2018/run_mlaw_cbm_fold04.sh

# Wavelet-energy ablation
bash training/isic2018/run_mlaw_cbm_energy_fold04.sh
```

The launchers preserve the controlled Fold04 recipe: seed 43, 100 epochs,
batch size 64, AdamW, `lambda_cpt=2.5`, and Top-K 98.

## Scalp experiments

Arrange the private scalp dataset as:

```text
dataset/scalp/
  train/<class name>/*
  test/<class name>/*
```

Then run:

```bash
# Validate paths and configuration without creating output
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --validate-config-only

# One-batch forward/backward check
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --gpu 0 \
  --smoke-test

# Train the formal residual-bank MLAW-CBM
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --gpu 0

bash training/scalp/run_mlaw_cbm_energy.sh
```

See [training/scalp/README.md](training/scalp/README.md) for the six required
class names and protocol warning. A new researcher who only needs the formal
Scalp model can follow
[training/scalp/QUICKSTART_V5_4.md](training/scalp/QUICKSTART_V5_4.md).

The public imports are:

```python
from model.mlaw_cbm import MLAWCBM              # formal X - R_band model
from model.mlaw_cbm_res import MLAWCBM_res      # preserved direct-IDWT model
```

## Diagnostic results

ISIC2018 Fold04, checkpoint-selected shared holdout, percent:

| Model | Historical ID | ACC | BMAC | Macro-F1 |
|---|---:|---:|---:|---:|
| MVP-CBM baseline | baseline | 92.016 | 87.303 | 87.027 |
| Attribute–Wavelet | V3 | 92.814 | 89.778 | 89.114 |
| Counterfactual selector | newV5 | 92.116 | 89.611 | 88.890 |
| **MLAW-CBM** | **V5-4** | **92.615** | **90.310** | **90.126** |
| MLAW-CBM-Energy | V5-7 | 92.216 | 90.041 | 89.966 |

These are method-development results, not a claim of final generalization.
Machine-readable values and protocol notes are in
[results/isic2018/fold04_summary.csv](results/isic2018/fold04_summary.csv).

## Additional training modes

- [training/two_stage/README.md](training/two_stage/README.md): train concepts
  first, then freeze the concept branch and train the disease classifier.
- [training/concept_supervision/README.md](training/concept_supervision/README.md):
  train the seven categorical concept heads with per-image pseudo labels.

## Reproducibility and limitations

- The versioned ISIC2018 split contains 9,013 training images and a 1,002-image
  shared validation/test holdout.
- The historical image-level split has lesion overlap between training and
  holdout (445 holdout images). Use a lesion-grouped independent test protocol
  before making publication-level generalization claims.
- The scalp protocol also uses the same test directory for validation and
  final reporting.
- `output/`, datasets, logs, and weights are ignored. Preserve the exact source
  commit, manifest hashes, metrics, and prediction CSVs for every reported run.
- A literature review is still required before using an academic priority
  claim such as “the first” in a paper.

## License

Released under the [MIT License](LICENSE).
