# Scalp training

This package contains the controlled six-class scalp protocol used by
MLAW-CBM.  Images are not included.  Arrange the dataset as documented in
`training/scalp/dataset.py`, then pass its root with `--data-path`.

`run_mlaw_cbm.sh` uses the formal residual-bank implementation. It builds the
LH/HL/HH residual bank with one batched IDWT and forms each counterfactual as
`X - R_band`. The former direct-IDWT model remains importable as
`model.mlaw_cbm_res.MLAWCBM_res` for controlled comparisons.

For a handoff-ready V5-4 walkthrough, start with
[`QUICKSTART_V5_4.md`](QUICKSTART_V5_4.md).

The audited protocol reuses the test split for validation and final reporting.
Reported scores are therefore diagnostic and not an independent test estimate.

```bash
# Configuration only (does not create an output directory)
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --validate-config-only

# One-batch forward/backward test (creates no output directory)
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --gpu 0 \
  --smoke-test

# Formal residual-bank MLAW-CBM
bash training/scalp/run_mlaw_cbm.sh \
  --data-path /path/to/scalp \
  --gpu 0

# MLAW-CBM-Energy ablation
bash training/scalp/run_mlaw_cbm_energy.sh --data-path /path/to/scalp
```

The trainer refuses to overwrite an existing output directory. Preserve the
source commit, generated `metrics.json`, manifest hash, and checkpoint index
for every reported run. Datasets, logs, outputs, and model weights are ignored
by Git and must be transferred separately when needed.

Historical V3 and counterfactual-selector entry points remain available as
Python modules for method-evolution experiments:

```bash
python -m training.scalp.train_v3 --help
python -m training.scalp.train_counterfactual_wavelet --help
```
