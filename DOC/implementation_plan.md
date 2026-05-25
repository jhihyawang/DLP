# Implementation Plan

更新時間：2026-05-20

## 目標

完成 bounding-box-guided object addition：輸入 source image、文字 instruction、bounding box，輸出在指定區域新增物件且盡量保留 bbox 外背景的 edited image。

## 可行方法

### Strategy A：Input-level Conditioning

- 做法：將 bbox mask downsample 後接到 U-Net input，擴充 `conv_in` 到 9 channels，同時訓練 LoRA。
- 優點：最直接，bbox 訊號進入 denoising input，容易做 bbox ablation。
- 缺點：需要修改 `conv_in`，checkpoint/inference 需要額外載入 conv weights；可能對 pretrained distribution 擾動較大。
- Starter：`model/modelA.py`、`train.py`、`infer_strategy_a.py`。

### Strategy B：Attention/Latent-level Conditioning

- 做法：保持 pretrained input，訓練 LoRA，inference 時用 bbox latent blending / pixel composite 限制改動區域。
- 優點：對 pretrained pipeline 侵入較小，訓練成本較低。
- 缺點：bbox 主要在 loss/inference 約束，模型本身對位置的可控性可能較弱。
- Starter：`model/modelB.py`、`train_strategy_b.py`、`infer_strategy_b.py`。

### Strategy C：Feature-level Conditioning

- 做法：以 ControlNet-style bbox branch 注入 multi-level residual features，凍結原 editing U-Net，訓練 bbox-conditioned ControlNet。
- 優點：較符合 spatial control 需求，bbox 訊號能在多層 feature 影響生成；主模型凍結，便於比較。
- 缺點：ControlNet 訓練參數較多，GPU/記憶體需求較高；starter 仍需補 evaluation 與 logging。
- Starter：`model/modelC.py`、`train_strategy_c.py`、`infer_strategy_c.py`。

## 初始選擇

先以 Strategy C 作為主要研究方向，Strategy A/B 作為 baseline 或 ablation。

理由：

- Proposal 的核心是明確 spatial control，Strategy C 的 feature-level conditioning 最直接支援 bbox control。
- ControlNet-style 設計有既有成功經驗，且凍結 base editing U-Net 可降低破壞 pretrained 能力的風險。
- Strategy B 可作為低成本 baseline；Strategy A 可驗證 input-level bbox 是否已足夠。

## 初始工作順序

1. 建立 `DL_Final` 環境並驗證 imports。
2. 修正或補齊可重現 logging：run name、metrics CSV、checkpoint path、config snapshot。
3. 做 dataset smoke test：確認 PIPE/PIPE_Masks 可下載或可從 cache 讀取。
4. 做小樣本 training smoke test：每個策略至少 1 至數個 step，確認 loss finite、checkpoint 可存。
5. 先完成 Strategy C 小實驗與 inference bbox sweep。
6. 補 evaluation script：至少輸出 bbox IoU/inside-outside LPIPS/CLIP-based metric 或可替代初版 metric。
7. 整理結果曲線、圖表、report draft。

## 風險

- Starter code 多處預設 `/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def`，本機若未 mount，dataset/model cache 會失敗。
- PIPE dataset 很大，不應直接下載到專案資料夾。
- RTX 3090 24GB 可做小 batch，但 full training 需調 batch/image size/gradient accumulation。
- `evaluate.py`、`test.py` 目前為空，正式驗收前需補。
