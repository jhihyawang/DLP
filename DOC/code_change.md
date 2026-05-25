# Code Change Log

## 2026-05-20

### 新增資料夾

- `download/`：保留原始下載檔或 setup scripts。
- `data/`：若需要小型解壓資料可放置於此；大型 dataset 不應放進提交。
- `configs/`：保留實驗設定。
- `scripts/`：保留輔助 scripts。
- `logs/`：保留環境建立、training、evaluation logs。
- `checkpoints/`：保留 checkpoint 或 checkpoint symlink/說明。
- `results/curves/`：保留 training/metric curves。
- `outputs/`：保留最終輸出或提交檔案。
- `data/hf_cache_smoke/`：framework import smoke test 使用的空 Hugging Face cache placeholder，避免 import 時要求預設 `/media/zia/...` mount。

### 新增檔案

- `DOC/rule.md`
  - 整理專案規則、proposal/paper 重點、starter code 摘要、輸入/輸出格式、暫定驗收標準。
- `DOC/implementation_plan.md`
  - 整理 Strategy A/B/C 的優缺點、初始選擇與工作順序。
- `DOC/codex_log.md`
  - 記錄初始盤點、讀檔、環境檢查 command 與摘要。
- `DOC/code_change.md`
  - 本檔，記錄新增/修改檔案與原因。
- `DOC/experiment_log.md`
  - 記錄初始實驗策略與待執行項目。
- `logs/env_create_DL_Final.log`
  - `DL_Final` conda 環境建立完整輸出。

### 修改檔案

- 無 starter/framework source code 修改。
- 執行 `py_compile` 與 framework import smoke test 時，Python 產生/更新了 `framework/sourcecode/**/__pycache__/*.pyc`。這些是 generated cache artifacts，不是 source code 變更。
- 產生 `logs/env_create_DL_Final.log`，內容為 `DL_Final` 環境建立完整輸出。
- 路徑修正：初始 markdown 文件曾誤放在外層 `/home/chrceu/Desktop/master/1-2/DL/lab/DOC/`；確認內容全為本次 Codex 產物後，已移動到 `final_project/DOC/`，並移除外層誤建資料夾。
- `DOC/codex_log.md`：補上 2026-05-20 重新閱讀 `proposal.md` 與 `Group14_Proposal.pptx.pdf` 的 command、摘要與結論。
- `DOC/code_change.md`：補上本次文件維護紀錄。
- `DOC/codex_log.md`：補上 PIPE / PIPE_Masks 資料來源、大小、欄位、sample 與載入建議。
- `DOC/experiment_log.md`：補上 streaming 讀取 sample 的風險註記。
- `scripts/download_pipe_data.py`：新增 PIPE / PIPE_Masks 下載腳本，會設定 Hugging Face cache、下載指定 split，並輸出 summary JSON。
- `scripts/download_pipe_data.py`：補上 `HF_XET_CACHE`，避免 Hugging Face Xet backend 把大型 blob 放到預設 `~/.cache/huggingface/xet`。
- `scripts/download_pipe_data.py`：將 `datasets` / `huggingface_hub` imports 移到 cache env 設定之後，確保 `HF_HOME`、`HF_DATASETS_CACHE`、`HF_HUB_CACHE`、`HF_XET_CACHE` 生效。
- `data/hf_cache/`：下載並建立完整 PIPE / PIPE_Masks Hugging Face dataset cache，供後續 training 使用；此為大型資料產物，不應放進最終提交。
- `data/hf_cache/download_pipe_data_summary.json`：下載腳本輸出的資料來源、split 筆數、欄位與 sample metadata summary。
- `logs/download_pipe_data.log`：完整下載與 cache 生成 log。
- `DOC/codex_log.md`：補上完整下載 command、監控過程、完成結果、驗證 command 與問題處理。
- `DOC/experiment_log.md`：將 `smoke_dataset_pipe_bbox` 更新為 completed，補上資料筆數、sample tensor shape 與 bbox 驗證結果。

### 是否改到 starter / official / TA-provided files

- Source code 否。`framework/` 下 `.py` 檔目前只讀不改。
- 注意：`__pycache__` 是驗證時由 Python 自動產生/更新的 cache。

## 2026-05-21

### 新增檔案與產物

- `data/model_cache/`
  - `paint-by-inpaint/add-base` model cache for smoke training；此為大型 cache，後續提交時應排除。
- `logs/smoke_train_b.log`
  - Strategy B 1-step smoke training log。
- `logs/smoke_train_b_metrics.csv`
  - Strategy B smoke run 的 step/loss/learning-rate CSV。
- `checkpoints/smoke_train_b/`
  - Strategy B epoch/final LoRA smoke checkpoints。

### 修改檔案

- `framework/sourcecode/model/modelA.py`
  - 讓 model cache 可由 `PIPE_MODEL_CACHE_DIR` 指定，避免本機沒有 `/media/zia/...` mount 時無法載入 pretrained model。
- `framework/sourcecode/model/modelB.py`
  - 同上，供 Strategy B smoke training 與後續 baseline training 使用。
- `framework/sourcecode/model/modelC.py`
  - 同上，供主要 Strategy C 後續 training 使用。
- `framework/sourcecode/train_strategy_b.py`
  - 新增 `--metrics-csv`。
  - 每個 training step 寫出 `step`, `epoch`, `loss`, `metric`, `learning_rate` 欄位；目前 smoke training 尚未有 evaluation metric，所以 `metric` 欄先留空。
- `framework/sourcecode/train_strategy_c.py`
  - 新增 `--metrics-csv`，讓 ControlNet training 也逐 step 留下 loss / learning-rate CSV。
- `framework/sourcecode/infer_strategy_a.py`
  - 讓 inference dataset cache 可由 `PIPE_HF_CACHE_DIR` 指定，避免本機沒有 `/media/zia/...` mount 時無法生圖。
- `framework/sourcecode/infer_strategy_b.py`
  - 同上，讓 Strategy B inference 可使用 project 內 PIPE cache。
- `DOC/codex_log.md`
  - 補上進度查核、hard-coded model cache 問題、Strategy B smoke training command 與結果。
- `DOC/experiment_log.md`
  - 將 `smoke_train_b` 更新為 completed，記錄 config、loss、CSV、checkpoint 與尚未跑 inference/evaluation。

### 是否改到 starter / official / TA-provided files

- 有改到組員提供的 `framework/sourcecode/` Python source。
- 修改原因是讓本機 project cache 可用，並補上 smoke training 的 metrics CSV；未修改論文、proposal PDF 或環境檔。
- `checkpoints/strategy_b/`
  - Strategy B 256-sample / 10-epoch run 的 epoch 與 final LoRA checkpoints。
- `logs/strategy_b_train.log`
  - Strategy B full run training log。
- `logs/strategy_b_metrics.csv`
  - Strategy B full run metrics CSV。
- `checkpoints/smoke_train_c_fp32_512/`
  - Strategy C 512px FP32 one-step smoke ControlNet checkpoint。
- `logs/smoke_train_c_fp32_512.log`
  - Strategy C smoke training log。
- `logs/smoke_train_c_fp32_512_metrics.csv`
  - Strategy C smoke metrics CSV。
- `logs/strategy_c_train.log`
  - Running Strategy C full run training log。
- `logs/strategy_c_metrics.csv`
  - Running Strategy C full run metrics CSV。
- `results/strategy_a_infer/`
  - Strategy A final checkpoint 對 test index 0 的 source/target/bbox overlay/baseline/strategy/grid images。
- `results/strategy_b_infer/`
  - Strategy B final checkpoint 對 test index 0 的 source/target/bbox overlay/baseline/strategy/grid images。
- `logs/infer_ab_index0.log`
  - Strategy A/B test index 0 inference log。
- `scripts/infer_ab_batch.sh`
  - 批次產生 Strategy A/B multi-test 與 bbox-sweep inference images。
- `results/strategy_a_multi_test/`
  - Strategy A final checkpoint 對 test indices 1-4 的 inference artifacts。
- `results/strategy_b_multi_test/`
  - Strategy B final checkpoint 對 test indices 1-4 的 inference artifacts。
- `results/strategy_a_bbox_sweep/`
  - Strategy A final checkpoint 對 test indices 0-2 的 bbox sweep artifacts。
- `results/strategy_b_bbox_sweep/`
  - Strategy B final checkpoint 對 test indices 0-2 的 bbox sweep artifacts。
- `logs/infer_ab_batch.log`
  - A/B batch inference log。

## 2026-05-22 Strategy C batch inference

### 新增檔案與產物

- `scripts/infer_c_batch.sh`
  - 批次產生 Strategy C multi-test 與 bbox-sweep inference images，測資 index 與 A/B batch 對齊。
- `logs/infer_c_batch.log`
  - Strategy C batch inference log；由 tmux run 透過 `tee` 寫入。
- `results/strategy_c_multi_test/`
  - Strategy C final checkpoint 對 test indices 1-4 的 inference artifacts。
- `results/strategy_c_bbox_sweep/`
  - Strategy C final checkpoint 對 test indices 0-2 的 bbox sweep artifacts。

### 修改檔案

- `framework/sourcecode/infer_strategy_c.py`
  - 讓 inference dataset cache 可由 `PIPE_HF_CACHE_DIR` 指定，避免本機沒有舊 `/media/zia/...` mount 時無法使用 project 內 PIPE cache。
- `DOC/codex_log.md`
  - 補上 Strategy C batch inference 的啟動與驗證 command。
- `DOC/experiment_log.md`
  - 補上 Strategy C batch inference 的 config、輸出路徑與完成結果。

### 是否改到 starter / official / TA-provided files

- 有改到組員提供的 `framework/sourcecode/infer_strategy_c.py`。
- 修改範圍只在 cache path 設定，讓已下載到 project 的 PIPE cache 可被 Strategy C inference 重用。

## 2026-05-22 Large-offset bbox comparison

### 新增檔案與產物

- `scripts/infer_bbox_corner_compare.sh`
  - 對指定 test indices 跑 A/B/C corner bbox sweep；不傳 index 時預設跑 bbox 較小的 indices 3, 4, 5。
- `scripts/stack_bbox_sweep_grids.py`
  - 將同一 test index 的 A/B/C bbox sweep grid 疊成一張 comparison image；可指定 input suffix 與 output prefix。
- `logs/infer_bbox_corner_compare.log`
  - Corner bbox compare batch log；由 tmux run 透過 `tee` 寫入。
- `logs/infer_bbox_corner_compare_12.log`
  - Test indices 1, 2 的 corner bbox compare batch log。
- `results/strategy_a_bbox_corner_sweep/`
- `results/strategy_b_bbox_corner_sweep/`
- `results/strategy_c_bbox_corner_sweep/`
  - A/B/C corner bbox sweep 的 per-strategy artifacts。
- `results/bbox_corner_compare/`
  - indices 3, 4, 5 的 A/B/C stacked comparison images。

### 修改檔案

- `framework/sourcecode/infer_strategy_a.py`
- `framework/sourcecode/infer_strategy_b.py`
- `framework/sourcecode/infer_strategy_c.py`
  - 將 bbox sweep 從同一水平線的 left/center/right 改成 top-left/top-right/center/bottom-left/bottom-right/original，讓位置控制差異更容易觀察。
  - 新增 `small-corners` 與 `vertical` sweep modes，供 spoon test 0 以縮小 bbox 四角移動、原 bbox 上中下移動觀察。
- `DOC/codex_log.md`
- `DOC/experiment_log.md`
  - 記錄選樣理由、batch command 與 comparison image 位置。

### 是否改到 starter / official / TA-provided files

- 有改到組員提供的 Strategy A/B/C inference source。
- 修改只影響 bbox sweep 測試產圖，不改 training checkpoint。

## 2026-05-22 Spoon test 0 bbox comparisons

### 新增檔案與產物

- `scripts/infer_spoon_bbox_compare.sh`
  - 產生 spoon test 0 的 small custom bbox 與 original-size vertical bbox A/B/C comparison。
- `logs/infer_spoon_bbox_compare.log`
  - Spoon bbox compare batch log。
- `results/bbox_spoon_compare/`
  - Spoon A/B/C stacked comparison images。

### 修改檔案

- `scripts/stack_bbox_sweep_grids.py`
  - 支援 spoon 專用 strategy output suffix 與 comparison output prefix。
- `framework/sourcecode/infer_strategy_a.py`
- `framework/sourcecode/infer_strategy_b.py`
- `framework/sourcecode/infer_strategy_c.py`
  - 支援 `--bbox-sweep-mode small-corners` 與 `--bbox-sweep-mode vertical`。

### 是否改到 starter / official / TA-provided files

- 有改到組員提供的 Strategy A/B/C inference source。
- 修改只新增 inference 測試方式，不改 training checkpoint。

## 2026-05-25 GitHub packaging preparation

### 新增檔案與產物

- `.gitignore`
  - 排除 dataset cache、model cache、training checkpoints、full result tree、runtime logs、Python cache。
  - 保留可供報告/隊友檢視的 compact comparison figures。
- `README.md`
  - GitHub 首頁用的 setup / dataset / checkpoint / result 說明。
- `DOC/github_packaging.md`
  - 說明哪些檔案應該推 GitHub、哪些大型產物不該推、如何另外分享 checkpoints/data。

### 修改檔案

- `DOC/code_change.md`
  - 補上 GitHub packaging 準備紀錄。

### 是否改到 starter / official / TA-provided files

- 否。只新增 packaging 文件與 `.gitignore`。
