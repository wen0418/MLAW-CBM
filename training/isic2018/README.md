# ISIC2018 training

The public entry points preserve the audited Fold04 recipe (seed 43, batch 64,
100 epochs, AdamW, `lambda_cpt=2.5`, Top-K 98).  Place ISIC2018 below
`dataset/ISIC2018` or override `--data-path`.

```bash
# Configuration validation only
bash training/isic2018/run_mlaw_cbm_fold04.sh --validate-config-only

# Main model
bash training/isic2018/run_mlaw_cbm_fold04.sh

# Wavelet-energy ablation
bash training/isic2018/run_mlaw_cbm_energy_fold04.sh
```

Method-evolution runs:

```bash
bash training/isic2018/run_v3_fold04.sh
bash training/isic2018/run_counterfactual_wavelet_fold04.sh
```

The Fold04 holdout is used both for checkpoint selection and final reporting.
These metrics are diagnostic and optimistically biased, not an independent
test estimate.
