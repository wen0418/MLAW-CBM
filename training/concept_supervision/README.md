# Per-image concept-supervision training

這個資料夾只管理 ISIC2018 Fold04 的 per-image `positive_result` joint
訓練。兩階段訓練位於 `training/two_stage/`。

Pseudo-label CSV 不會上傳到公開 Git。執行前請將授權的檔案放在
`dataset/pseudo_labels/training.csv`；`--validate-config-only` 也會讀取並驗證
此檔案，因此檔案不存在時會明確失敗。CPU contract tests 仍會執行模型與
recipe 測試，但會跳過需要私人 CSV 的 data-contract cases。

> 專案沿用的「multilabel」名稱，技術上是七個互斥的 categorical
> Attribute heads，每個 head 多一個 `nothing=0`，不是 34 個獨立 sigmoid/BCE
> labels。

## 檔案

- `train_fold04_new_v5_positive_result.py`：既有 NewV5 trainer 的相容入口。
- `train_fold04_v5_4_positive_result.py`：V5-4 positive-result joint trainer。
- `run_fold04_new_v5_positive_result.sh`：既有 NewV5 完整 Fold04 設定。
- `run_fold04_v5_4_positive_result.sh`：V5-4 的相同 Fold04 設定。
- `test_new_v5_positive_result.py`：既有 NewV5 CPU contract tests 入口。
- `test_v5_4_positive_result.py`：V5-4 label/head/recipe CPU contract tests。

相關 model 保持在標準 `model/` package：

- `model/mvpcbm_attribute_wavelet_last_new_v5_positive_result.py`
- `model/mvpcbm_attribute_wavelet_last_v5_4_positive_result.py`

## 控制條件

V5-4 positive-result run 保留原 V5-4 的：

- Fold04 image-level manifest（train 9,013；shared val/test 1,002）
- seed 43、100 epochs、batch size 64
- AdamW、warmup/LR schedule、class weights、augmentations
- `lambda_cpt=2.5` 與 MCSAF sparse loss
- best-BMAC / best-ACC / best-Macro-F1 / best-tradeoff checkpoints
- pre-LayerNorm `X + softplus(s_l) * H*` concept AP input
- 原 34 維 disease-classifier input

唯一的 supervision 改動是七個 concept heads 加入 learnable `nothing=0`，並以
`positive_result` 訓練。`nothing` logits 不會送進 disease classifier。

## 建議執行順序

以下命令都從專案根目錄執行。環境名稱是 `cbm`。

```bash
cd /path/to/MLAW-CBM
mamba run -n cbm python -m unittest -v \
  training.concept_supervision.test_v5_4_positive_result
```

只驗證資料、recipe、SHA 與輸出規劃，不載入 GPU model、不建立 output：

```bash
mamba run -n cbm bash training/concept_supervision/run_fold04_v5_4_positive_result.sh \
  --validate-config-only
```

GPU smoke test（不建立正式 output）：

```bash
mamba run -n cbm bash training/concept_supervision/run_fold04_v5_4_positive_result.sh \
  --smoke-test
```

正式訓練：

```bash
mamba run -n cbm bash training/concept_supervision/run_fold04_v5_4_positive_result.sh
```

預設輸出：

`output/isic2018/baseline_cv10_attribute_wavelet_last_v5_4_positive_result_k98_imglevel_fold04_seed43`

trainer 會拒絕覆寫已存在的 output directory。若要重跑，請透過額外參數指定
新的 `--output-dir`，不要覆蓋舊實驗。
