# Final Project Rule Summary

更新時間：2026-05-20

## 專案範圍

- 專案根目錄：`/home/chrceu/Desktop/master/1-2/DL/lab/final_project`
- 主要 framework：`framework/sourcecode/`
- 環境檔：`framework/environment.yml`
- 任務文件：`DOC/2404.18212v3.pdf`、`DOC/Group14_Proposal.pptx.pdf`、`DOC/proposal.md`
- `DOC/proposal.md` 有簡短中文 proposal note；完整 proposal 內容主要仍在 `Group14_Proposal.pptx.pdf`。

## 已盤點資料夾結構

```text
final_project/
├── DOC/
│   ├── 2404.18212v3.pdf
│   ├── Group14_Proposal.pptx.pdf
│   └── proposal.md
├── framework/
│   ├── environment.yml
│   └── sourcecode/
│       ├── dataset.py
│       ├── evaluate.py
│       ├── export_pipe_samples.py
│       ├── infer_strategy_a.py
│       ├── infer_strategy_b.py
│       ├── infer_strategy_c.py
│       ├── pipe_bbox_dataset.py
│       ├── run_paint_by_inpaint.py
│       ├── test.py
│       ├── train.py
│       ├── train_strategy_b.py
│       ├── train_strategy_c.py
│       └── model/
│           ├── modelA.py
│           ├── modelB.py
│           └── modelC.py
```

Codex 也建立了以下工作資料夾：`download/`、`data/`、`configs/`、`scripts/`、`logs/`、`checkpoints/`、`results/curves/`、`outputs/`。

## 必做事項

- 根據 Paint by Inpaint 論文與 Group14 proposal，完成「可用 bounding box 控制物件新增位置/尺度」的 text-guided image editing final project。
- 使用專用 conda 環境 `DL_Final`，由 `framework/environment.yml` 建立。
- 維護 `DOC/codex_log.md`、`DOC/code_change.md`、`DOC/experiment_log.md`。
- 所有 training/evaluation 都要保留 run name、config、log、checkpoint/output、metric。
- 需要能產生 training/metric curve，建議輸出到 `results/curves/`，metrics CSV 放在 `logs/`。
- 實作後需做基本驗證，例如 compile、smoke test、small run 或 evaluation。

## 禁止事項

- 不改動與 final project 無關的檔案。
- 不 revert、不覆蓋既有使用者修改。
- 不任意修改 official/TA/third-party/starter code；若必須修改，要先說明理由。
- 不把大型 dataset 放進最後提交，除非規則要求。
- 長時間任務需用 tmux，且 command 需 tee 到 `logs/`。

## 論文重點

論文：`Paint by Inpaint: Learning to Add Image Objects by Removing Them First`。

- 核心想法：object addition 可視為 object removal/inpainting 的反向流程。
- PIPE dataset：source 是移除物件後的圖，target 是原圖，並搭配 object addition instruction。
- 訓練方法：以 Stable Diffusion / InstructPix2Pix 類架構，condition on text instruction 與 source image。
- 論文訓練細節摘要：SD 1.5 初始化、training resolution 256、conditioning dropout 約 5%、Adam、learning rate `5e-5`、gradient accumulation、max grad norm 1。
- 論文 evaluation：PIPE test、OPA、MagicBrush；metrics 包含 L1/L2、CLIP-I、DINO、CLIP-T、CMMD。
- 限制：資料生成不是完全無錯，較難處理遠距離 shadow/reflection 或複雜 object-object interaction。

## Proposal 目標

- Motivation：現有 diffusion image editing 可新增物件，但 placement 通常由模型自動決定。
- Goal：加入 bounding-box guidance，讓使用者明確控制新增物件的位置與尺度。
- `DOC/proposal.md` note：主軸延伸原文 Paint；原文使用 conditional diffusion model，以輸入影像和文字 prompt 作條件；本專案多加 bbox 作為條件，並提出三種 bbox conditioning 方法；dataset 使用原文 PIPE。
- Proposed strategies：
  - Strategy A：Input-level conditioning
  - Strategy B：Attention-level conditioning
  - Strategy C：Feature-level conditioning
- Dataset：
  - Training/validation：PIPE Dataset
  - Benchmark/generalization：MagicBrush Dataset
- Expected metrics：
  - Semantic Alignment：CLIP Score 越高越好
  - Spatial Control：Edit-region IoU 越高越好
  - Background Preservation：Outside LPIPS 越低越好
  - Object Quality：Inside LPIPS 越低越好

## Starter Code 摘要

- `dataset.py`：載入 `paint-by-inpaint/PIPE`，需要 `PIPE_HF_CACHE_DIR` 或預設 3TB disk mount。
- `pipe_bbox_dataset.py`：載入 PIPE 與 PIPE_Masks，產生 `source_img`、`target_img`、`object_mask`、`bbox_mask`、`bbox`、`instruction`。
- `model/modelA.py`：Strategy A，擴充 U-Net `conv_in` 到 9 channels，訓練 bbox input 與 LoRA。
- `model/modelB.py`：Strategy B，原 InstructPix2Pix input，訓練 U-Net LoRA。
- `model/modelC.py`：Strategy C，ControlNet-style bbox branch，frozen editing U-Net + trainable ControlNet。
- `train.py`：Strategy A training。
- `train_strategy_b.py`：Strategy B training。
- `train_strategy_c.py`：Strategy C training，包含 bbox random shift pseudo-target。
- `infer_strategy_a.py` / `infer_strategy_b.py` / `infer_strategy_c.py`：各策略 inference 與 bbox sweep。
- `run_paint_by_inpaint.py`：直接使用 Paint-by-Inpaint pretrained model。
- `export_pipe_samples.py`：輸出 PIPE sample、mask、bbox overlay。
- `evaluate.py`、`test.py` 目前為空。

## 輸入資料格式

PIPE sample 主要欄位：

- `source_img`：移除物件後的 source image
- `target_img`：含物件的 target/original image
- `Instruction_VLM-LLM`
- `Instruction_Class`
- `Instruction_Ref_Dataset`
- `object_location`
- `img_id`
- `ann_id`

PIPE_Masks sample 主要欄位：

- `mask`
- `img_id`
- `ann_id`

`PIPEBBoxDataset` 會額外產生：

- `object_mask`
- `bbox_mask`
- `bbox`，格式為 `(x1, y1, x2, y2)`

## 輸出格式

建議與 starter code 對齊：

- Checkpoints：`checkpoints/` 或明確 run-specific checkpoint path。
- Training logs：`logs/{RUN_NAME}.log`
- Metrics CSV：`logs/{RUN_NAME}_metrics.csv`
- Curves：`results/curves/`
- Inference outputs：`outputs/` 或 `results/`
- Final report draft：`DOC/`

## 評分/驗收標準

目前文件未看到助教明確 rubric；暫定驗收重點：

- 是否能用 bbox 控制物件新增的位置/尺度。
- 生成結果是否符合文字 instruction。
- bbox 指定區域內是否成功新增目標物。
- bbox 外背景是否盡量保持。
- 是否能重現環境、訓練、evaluation 與結果。

## Deadline / 提交格式

目前已讀文件沒有明確 deadline 或提交格式。後續若取得課程 spec，需補到本檔。
