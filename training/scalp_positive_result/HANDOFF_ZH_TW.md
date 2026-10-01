# Scalp VLM image-grounded labels 實驗交接

這份文件用於交接「正式 residual-bank MLAW-CBM + 逐圖 VLM concept labels」的
scalp 訓練。程式碼由 GitHub repository 提供；VLM label CSV 由交接人另外以私人資料包
提供。影像資料、label CSV、checkpoint 與輸出結果都不應直接放進公開 GitHub。

## 1. 這個實驗在測什麼

正式模型維持 residual-bank ACFS、GAB/newCLS、concept bottleneck 與疾病分類路徑不變，
只把原本「每個疾病類別共用一組 concept label」換成每張圖片各自的 VLM label。

正式設定使用 `inclusive_output_list`：

- `0`：圖片沒有足夠的可見證據支持該 Attribute；
- `1..S`：原本 zero-based concept state 加一；
- 六個 concept head 大小依序為 `5, 5, 5, 5, 4, 4`；
- 每個 head 新增的 `nothing` reference 只用於 concept cross-entropy；
- 疾病分類器仍只接收原本 22 個 concept-state activations，不接收六個
  `nothing` logits。

主要程式：

```text
model/mlaw_cbm_scalp_positive_result.py
training/scalp_positive_result/dataset.py
training/scalp_positive_result/train_mlaw_cbm.py
training/scalp_positive_result/run_mlaw_cbm.sh
training/scalp_positive_result/test_contract.py
```

## 2. 交接人需要提供的兩部分

### A. GitHub 程式碼

接收者應取得包含 `training/scalp_positive_result/` 的確切 branch/commit。開始前先記錄：

```bash
git status --short --branch
git rev-parse HEAD
```

不要只用 branch 名稱記錄實驗；最後報告必須保存 commit SHA。

### B. 私下提供的 VLM label 包

解壓後必須保留以下結構與檔名：

```text
scalp_concept/
├── scalp_concept_vocabulary.csv
├── training.csv
└── test.csv
```

這三個檔案不是影像，也不會取代 scalp dataset。它們提供逐圖 concept target 與
稽核資訊；影像仍由 `New_scalp/` 讀取。

目前正式 label 包的 SHA-256：

```text
5ec712bee003fb1b49bfebd61e91016f4d9e539e64b0788583cadf040ed052d9  scalp_concept_vocabulary.csv
483a54b79dee1485a4688ac7f2d61e5f36dc02146ad7e42a7096ad1d2cc7dcd6  training.csv
4c26e2a90eab1f3e09b72f361397626fa974ea1027360bb5e495dbd957840e5f  test.csv
```

預期 CSV 行數（包含 header）：

```text
scalp_concept_vocabulary.csv: 22
training.csv:                 7646
test.csv:                     3569
```

收到資料包後，在 `scalp_concept/` 目錄執行：

```bash
sha256sum scalp_concept_vocabulary.csv training.csv test.csv
wc -l scalp_concept_vocabulary.csv training.csv test.csv
```

任一 hash 或行數不同時先停止，不要直接訓練。這通常代表拿錯版本、傳輸損壞，或
CSV 曾被試算表軟體重新儲存。

## 3. 影像資料結構

`--data-path` 必須指向下列結構的根目錄：

```text
New_scalp/
├── train/
│   ├── Xerosis/
│   ├── Normal/
│   ├── Oily-Dandruff/
│   ├── Folliculitis/
│   ├── Seborrheic-dermatitis/
│   └── Dry-Dandruff/
└── test/
    ├── Xerosis/
    ├── Normal/
    ├── Oily-Dandruff/
    ├── Folliculitis/
    ├── Seborrheic-dermatitis/
    └── Dry-Dandruff/
```

正式資料量：

```text
train: 7645 images
test:  3568 images
```

### 重要：不要任意更名、增刪或搬動影像

VLM CSV 使用匿名流水號。程式會在每個 split 與疾病類別內，將「依字典序排序的
影像檔名」對到一號起算的 VLM ID。例如：

```text
SCALP_TEST_FOLLICULITIS_00001
SCALP_TEST_FOLLICULITIS_00002
...
```

只要檔名、數量或 class folder 改變，就可能讓 label 對錯圖片。loader 會檢查數量、
ID、診斷、label 範圍與 hard/inclusive 關係，但它無法判斷一張被重新命名的圖片是否
仍對到原本 ID。因此交接時應使用完全相同的影像資料版本。

## 4. 環境準備

從 repository 根目錄執行：

```bash
mamba env create -f environment.yaml
mamba activate cbm
```

若環境已建立，只需：

```bash
mamba activate cbm
```

本實驗需要 NVIDIA GPU。正式訓練沿用 100 epochs、batch size 64、Top-K 98、
`lambda_cpt=2.5`、seed 43 與相同的 class weights/augmentation。

## 5. 執行順序

程式內的預設資料路徑是原作者電腦的路徑。交接者在其他電腦執行時，務必明確傳入
`--data-path` 與 `--pseudo-label-dir`。

以下以實際路徑代換 `/path/to/...`。

### Step 1：CPU/資料設定驗證

```bash
cd /path/to/MLAW-CBM

mamba run -n cbm bash training/scalp_positive_result/run_mlaw_cbm.sh \
  --data-path /path/to/New_scalp \
  --pseudo-label-dir /path/to/scalp_concept \
  --validate-config-only
```

這一步不會建立正式 output directory。成功時應確認輸出包含：

```text
train_size: 7645
val_size:   3568
test_size:  3568
concept_label_column: inclusive_output_list
counterfactual_construction: X_minus_R_band
formal_residual_bank_acfs_verified: true
```

如果出現 count、ID、diagnosis、hash、source 或 mapping mismatch，先處理錯誤，不要跳過
驗證直接訓練。

### Step 2：單 batch GPU smoke test

```bash
mamba run -n cbm bash training/scalp_positive_result/run_mlaw_cbm.sh \
  --data-path /path/to/New_scalp \
  --pseudo-label-dir /path/to/scalp_concept \
  --gpu 0 \
  --smoke-test
```

這一步驗證真正的 GPU forward、loss、backward、ACFS selector gradient、concept heads
與 disease classifier gradient。smoke-test 數字不能當正式實驗結果。

### Step 3：正式 100-epoch 訓練

```bash
mamba run -n cbm bash training/scalp_positive_result/run_mlaw_cbm.sh \
  --data-path /path/to/New_scalp \
  --pseudo-label-dir /path/to/scalp_concept \
  --output-dir /path/to/outputs/scalp_mlaw_vlm_seed43 \
  --tensorboard-dir /path/to/logs/scalp_mlaw_vlm_seed43 \
  --gpu 0
```

正式 target 預設就是 `inclusive_output_list`。除非正在做另一個明確命名的消融，請勿
加入 `--concept-label-column hard_output_list`。

程式會拒絕覆寫已存在的 `--output-dir`。不要刪除舊結果後重用相同名稱；每次 rerun
都建立新的 output 名稱並記錄原因。目前 trainer 沒有正式的 `--resume` 介面，中斷後
不要假設 `last.pth` 可以直接接續完整訓練。

## 6. 正式輸出

完成後 output directory 應至少包含：

```text
README.md
checkpoint_index.json
classification_report.txt
concept_metrics.csv
configuration_audit.json
confusion_matrix.csv
history.csv
image_label_manifest.csv
last.pth
metrics.csv
metrics.json
predictions.csv
training_report.txt
```

標準報告 checkpoint 是 `checkpoint_index.json` 指向的 best-BMAC checkpoint。不要只看
最後一個 epoch，也不要手動改用 test ACC 最高的 epoch 後再報 BMAC。

原機器 seed-43 參考結果如下，僅供確認結果量級，不是要求不同 GPU/軟體版本逐位元
一致：

```text
best epoch: 12
ACC:        86.1267
BMAC:       87.7121
Macro-F1:   80.5191
```

主要指標是 BMAC，但必須一起查看 Macro-F1、ACC、每類 precision/recall/F1，以及
`concept_metrics.csv`。目前資料嚴重不平衡，只看 BMAC 可能忽略少數類 precision
下降。

## 7. 目前 protocol 的限制

目前 scalp loader 將 `val` 與 `test` 都映射到 `New_scalp/test/`，兩者 100% 重疊。
因此 best-BMAC checkpoint 也是在同一批影像上選出，現有結果只能稱為
`shared-val/test diagnostic result`，不能描述成完全獨立的 test estimate。

若要產生論文最終結果，應另行建立固定 validation split，並把原本 test 保留到模型與
checkpoint 決策全部完成後再評估。不要把新 protocol 的結果直接與目前 shared-val/test
數字混成同一張表。

## 8. 隱私與檔案分享

- VLM CSV 使用匿名 image ID，但疾病名稱與 VLM reasoning 仍屬研究資料。
- `overall_reasoning` 欄位供稽核，目前不會進入模型或 loss。
- 原始影像檔名可能含可識別資訊。
- `image_label_manifest.csv`、`predictions.csv` 與其他輸出會保存影像相對路徑。
- 未完成去識別化與資料授權確認前，不要把影像、label 包、manifest、prediction 或
  checkpoint 推到公開 GitHub。

## 9. 交接完成檢查表

交接人：

- [ ] 將 scalp VLM 程式 commit/push，提供 branch 與 commit SHA。
- [ ] 私下提供完整 `scalp_concept/` 資料夾。
- [ ] 提供完全相同版本的 `New_scalp/` 或確認接收者已有相同版本。
- [ ] 與接收者共同核對三個 CSV hash。
- [ ] 共同跑過 `--validate-config-only`。

接收者：

- [ ] 記錄 `git rev-parse HEAD`。
- [ ] 核對 label hashes 與行數。
- [ ] 不更名、不增刪影像。
- [ ] 先跑 config validation，再跑 GPU smoke test。
- [ ] 正式輸出使用全新的 output/tensorboard 目錄。
- [ ] 保存 `configuration_audit.json`、`metrics.json`、predictions 與 checkpoint index。
- [ ] 報告中註明 shared validation/test 的限制。
