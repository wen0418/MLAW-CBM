# Training entry points

Run all commands from the repository root.

| Directory | Purpose |
|---|---|
| `isic2018/` | Main MLAW-CBM, Energy ablation, V3, and newV5 Fold04 runs |
| `scalp/` | Configurable six-class scalp experiments |
| `two_stage/` | Concept-first sequential training |
| `concept_supervision/` | Per-image categorical concept supervision |
| `common/` | Shared checkpoint and ISIC2018 recipe helpers |

Each formal trainer refuses to overwrite its planned output directory. Use
`--validate-config-only` before handing a run to another researcher, and give
every rerun a new `--output-dir`.
