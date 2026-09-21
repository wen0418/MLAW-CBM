# Two-stage training

這個資料夾只管理 concept-first 的兩階段訓練，與
`training/concept_supervision/` 中的 `positive_result` joint 訓練分開。

## 目前實驗

`v5_4_baseline_sequential` 使用 V5-4 model 與原始 MVP-CBM concept labels：

- Label source：`baseline.CONCEPT_LABEL_MAP`，七個 zero-based categorical labels。
- 不讀 `training.csv`，不使用 `positive_result`、`hard_result` 或 `nothing`。
- Stage 1：凍結 disease classifier，只訓練 concept branch。
- Stage 2：載入最佳 concept checkpoint，凍結 concept branch，只訓練 linear
  disease classifier。
- Stage 2 input：與原模型相同的 34 維 raw MCSAF concept activations。

## 檔案

- `train_fold04_v5_4_baseline_sequential.py`：兩階段 trainer。
- `run_fold04_v5_4_baseline_sequential.sh`：完整 Fold04 設定入口。
- `test_v5_4_baseline_sequential.py`：label、Raw34、metric 與凍結邊界測試。

## 執行

所有命令都從專案根目錄執行：

```bash
cd /path/to/MLAW-CBM

mamba run -n cbm python -m unittest -v \
  training.two_stage.test_v5_4_baseline_sequential

mamba run -n cbm bash \
  training/two_stage/run_fold04_v5_4_baseline_sequential.sh \
  --validate-config-only

mamba run -n cbm bash \
  training/two_stage/run_fold04_v5_4_baseline_sequential.sh \
  --smoke-test

mamba run -n cbm bash \
  training/two_stage/run_fold04_v5_4_baseline_sequential.sh
```

## 輸出

預設輸出目錄：

`output/isic2018/baseline_cv10_attribute_wavelet_last_v5_4_baseline_concept_sequential_raw34_k98_imglevel_fold04_seed43`

重要結果：

- `stage1_concept/metrics.json`：Stage 1 concept 指標與 runtime。
- `stage2_classifier/overall_metrics.csv`：整體 ACC、BMAC、Macro/Weighted
  Precision、Recall、F1。
- `stage2_classifier/per_class_metrics.csv`：七個疾病各自的 one-vs-rest
  ACC、Balanced Accuracy、Precision、Recall、Specificity、F1 與 support。
- `stage2_classifier/metrics.json`：Stage 2 完整指標與 runtime。
- `runtime.json`：Stage 1、Stage 2、每個 epoch、最終評估及總 runtime。
- `comparison.json`：Sequential 與既有 V5-4 joint 結果比較。

每類 Balanced Accuracy 是 one-vs-rest 的
`(recall + specificity) / 2`；整體 BMAC 是七類 recall 的平均。

trainer 會拒絕覆寫既有 output directory；重跑時請指定新的
`--output-dir`。
