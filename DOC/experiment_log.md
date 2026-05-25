# Experiment Log

## 2026-05-20 初始策略

### 實驗策略

先建立可重現環境與資料/模型 smoke test，再比較 Strategy A/B/C。主要方向暫定為 Strategy C，Strategy A/B 作為 baseline 或 ablation。

### 為什麼選這個策略

- Final project 目標是 bounding-box-guided controllable object addition。
- Strategy C 使用 ControlNet-style feature-level bbox conditioning，理論上對 spatial control 最直接。
- Strategy B 可快速得到低成本 baseline；Strategy A 可測試 input-level bbox conditioning 是否足夠。

### 待跑實驗

| Run name | Strategy | Config | Training command | Evaluation command | Status |
| --- | --- | --- | --- | --- | --- |
| `env_create_DL_Final` | environment | `framework/environment.yml` | `conda env create -n DL_Final -f framework/environment.yml` | import/version check | completed |
| `smoke_dataset_pipe_bbox` | dataset | image size 512, split test | none | load 1 sample | completed |
| `smoke_train_a` | A | tiny samples, 1 epoch/steps | `train.py` | `infer_strategy_a.py` | pending |
| `smoke_train_b` | B | 1 sample, 1 step, image size 256, fp16 | `train_strategy_b.py` | not run yet | completed |
| `strategy_b_train_256x10` | B | 256 samples, 10 epochs, image size 512, fp32 | `train_strategy_b.py` | not run yet | completed |
| `smoke_train_c_fp32_512` | C | 1 sample, 1 step, image size 512, fp32 | `train_strategy_c.py` | not run yet | completed |
| `strategy_c_train_256x10` | C | 256 samples, 10 epochs, image size 512, fp32 | `train_strategy_c.py` | not run yet | completed |
| `smoke_train_c` | C | tiny samples, 1 epoch/steps | `train_strategy_c.py` | `infer_strategy_c.py` | pending |

### Metrics 計畫

- Training：step/epoch、loss、learning rate，輸出 `logs/{RUN_NAME}_metrics.csv`。
- Semantic：CLIP Score 或 CLIP-T。
- Spatial：edit-region IoU 或 bbox/mask overlap proxy。
- Background：Outside LPIPS，越低越好。
- Object quality：Inside LPIPS 或 CLIP/DINO similarity proxy。
- Curves：輸出到 `results/curves/`。

### 目前結果

- Environment run `env_create_DL_Final` completed.
- Python：`3.11.15`
- torch：`2.12.0+cu130`
- torchvision：`0.27.0+cu130`
- CUDA：available，device `NVIDIA GeForce RTX 3090`
- diffusers：`0.37.1`
- transformers：`5.8.1`
- accelerate：`1.13.0`
- peft：`0.19.1`
- safetensors：`0.7.0`
- datasets：`4.8.5`
- CUDA smoke：`cuda_matmul_ok (64, 64) torch.float32`
- Starter compile smoke：`py_compile_ok`
- Framework import smoke：`framework_import_ok`

尚未開始 training/evaluation；下一步會進入小樣本訓練。

### Dataset Smoke Test Result

- Run name：`smoke_dataset_pipe_bbox`
- Data cache：`data/hf_cache`
- Summary：`data/hf_cache/download_pipe_data_summary.json`
- PIPE train/test rows：888,230 / 752
- PIPE_Masks train/test rows：888,230 / 752
- First test instruction：`Add a metallic spoon`
- First test ids：`img_id=129113`, `ann_id=1529722`
- Framework sample tensor shapes：
  - `source_img`: `(3, 512, 512)`
  - `target_img`: `(3, 512, 512)`
  - `object_mask`: `(1, 512, 512)`
  - `bbox_mask`: `(1, 512, 512)`
- First bbox：`[0.0, 144.0, 504.0, 344.0]`
- Verification result：`offline_load_ok`, `framework_dataset_smoke_ok`

### Strategy B Training Smoke Test Result

- Run name：`smoke_train_b`
- Why Strategy B first：LoRA-only baseline is the lightest path to verify dataset, pretrained model loading, optimizer step, checkpoint save, and metrics CSV before moving to Strategy C.
- Model：`paint-by-inpaint/add-base`
- Model cache：`data/model_cache`
- Dataset cache：`data/hf_cache`
- Config：
  - `max_train_samples=1`
  - `epochs=1`
  - `batch_size=1`
  - `num_workers=0`
  - `image_size=256`
  - `fp16=true`
  - `lr=5e-5`
  - `bbox_loss_weight=8.0`
- Training command：

```bash
python framework/sourcecode/train_strategy_b.py \
  --output-dir checkpoints/smoke_train_b \
  --metrics-csv logs/smoke_train_b_metrics.csv \
  --max-train-samples 1 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --image-size 256 \
  --save-every 1 \
  --fp16
```

- Log：`logs/smoke_train_b.log`
- Metrics CSV：`logs/smoke_train_b_metrics.csv`
- Step 1 loss：`0.5391167402267456`
- Learning rate：`5e-05`
- Checkpoints：
  - `checkpoints/smoke_train_b/strategy_b_epoch_0001_step_000001_lora/`
  - `checkpoints/smoke_train_b/strategy_b_final_epoch_0001_step_000001_lora/`
- Result：checkpoint save and metrics CSV both succeeded.
- Evaluation / inference：not run yet.

### Strategy A FP16 Failure Check

- User run attempted Strategy A with 256 samples, 10 epochs, image size 512, `lr=1e-4`, and `--fp16`.
- Observed failure：epoch 1 failed after the first completed step with `noise_pred finite=False` and `FloatingPointError: Training loss became NaN/Inf.`.
- Interpretation：the current Strategy A loop loads fp16 weights and performs training without mixed-precision scaler / fp32 master-weight handling, so direct fp16 training is unstable on this path.
- Verification run：Strategy A FP32 smoke at 512px passed 2 steps.
- Verification command：

```bash
python framework/sourcecode/train.py \
  --output-dir checkpoints/smoke_train_a_fp32 \
  --max-train-samples 2 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --image-size 512 \
  --lr 1e-4 \
  --lora-rank 4 \
  --lora-alpha 4 \
  --bbox-loss-weight 8.0 \
  --save-every 1
```

- FP32 smoke losses：`0.8218`, `0.9837`.
- FP32 smoke log：`logs/smoke_train_a_fp32.log`.
- FP32 smoke checkpoint：`checkpoints/smoke_train_a_fp32/`.

### Strategy B 256-Sample Train Run

- Run name：`strategy_b_train_256x10`
- Started：2026-05-21
- tmux session：`final_project_train_b`
- Training command：

```bash
python framework/sourcecode/train_strategy_b.py \
  --output-dir checkpoints/strategy_b \
  --metrics-csv logs/strategy_b_metrics.csv \
  --max-train-samples 256 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 2 \
  --image-size 512 \
  --lr 5e-5 \
  --lora-rank 4 \
  --lora-alpha 4 \
  --bbox-loss-weight 8.0 \
  --save-every 1
```

- Log：`logs/strategy_b_train.log`
- Metrics CSV：`logs/strategy_b_metrics.csv`
- Checkpoint dir：`checkpoints/strategy_b`
- Result：completed with final checkpoint.
- Metrics rows：2560 training steps plus CSV header.
- Final metrics row：step `2560`, epoch `10`, loss `0.08246304839849472`, learning rate `5e-05`.
- Final checkpoint：`checkpoints/strategy_b/strategy_b_final_epoch_0010_step_002560_lora/`.
- FP32 was chosen for this full-size run to match the successful Strategy A path and avoid the direct fp16 instability already observed on Strategy A.

### Strategy C Training Smoke and Full Run

- Smoke run：`smoke_train_c_fp32_512`
- Smoke config：1 sample, 1 epoch, image size 512, FP32.
- Smoke step 1 loss：`0.6548`.
- Smoke log：`logs/smoke_train_c_fp32_512.log`.
- Smoke metrics：`logs/smoke_train_c_fp32_512_metrics.csv`.
- Smoke checkpoint：`checkpoints/smoke_train_c_fp32_512/`.
- Full run：`strategy_c_train_256x10`
- tmux session：`final_project_train_c`
- Full training command：

```bash
python framework/sourcecode/train_strategy_c.py \
  --output-dir checkpoints/strategy_c \
  --metrics-csv logs/strategy_c_metrics.csv \
  --max-train-samples 256 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 2 \
  --image-size 512 \
  --lr 1e-5 \
  --bbox-loss-weight 8.0 \
  --bbox-shift-prob 0.5 \
  --controlnet-conditioning-scale 1.0 \
  --save-every 1
```

- Log：`logs/strategy_c_train.log`.
- Metrics CSV：`logs/strategy_c_metrics.csv`.
- Checkpoint dir：`checkpoints/strategy_c`.
- Final checkpoint：`checkpoints/strategy_c/strategy_c_final_epoch_0010_step_002560_controlnet/`.
- Final metrics row：step `2560`, epoch `10`, loss `0.017148436978459358`, learning rate `1e-05`.
- GPU at initial progress check：RTX 3090 using about 12.4GB at 98% utilization.

### Strategy A/B Test Image Generation

- Run：`infer_ab_index0`
- Shared config：
  - `split=test`
  - `index=0`
  - `img_id=129113`
  - `ann_id=1529722`
  - prompt：`Add a metallic spoon`
  - bbox：`[0.0, 144.0, 504.0, 344.0]`
  - seed：`1234`
  - inference steps：`20`
- Strategy A grid：
  - `results/strategy_a_infer/strategy_a_final_epoch_0010_step_002560_test_0_seed1234_grid.png`
- Strategy B grid：
  - `results/strategy_b_infer/strategy_b_final_epoch_0010_step_002560_test_0_seed1234_grid.png`
- Inference log：`logs/infer_ab_index0.log`
- Note：A/B inference ran while Strategy C training continued; Strategy C metrics had reached epoch 3, step 612 at follow-up check.

### Strategy A/B Multi-Test and BBox Sweep Batch

- Batch script：`scripts/infer_ab_batch.sh`
- Batch log：`logs/infer_ab_batch.log`
- Shared config：
  - checkpoint tag：`final_epoch_0010_step_002560`
  - split：`test`
  - seed：`1234`
  - inference steps：`20`
- Multi-test indices：`1`, `2`, `3`, `4`
- Multi-test grids：
  - Strategy A：`results/strategy_a_multi_test/`
  - Strategy B：`results/strategy_b_multi_test/`
- BBox sweep indices：`0`, `1`, `2`
- BBox sweep grids：
  - Strategy A：`results/strategy_a_bbox_sweep/`
  - Strategy B：`results/strategy_b_bbox_sweep/`
- Result：14 grid images generated successfully for A/B across the multi-test and bbox-sweep groups.

### Strategy C Multi-Test and BBox Sweep Batch

- Batch script：`scripts/infer_c_batch.sh`
- Batch log：`logs/infer_c_batch.log`
- Shared config：
  - checkpoint tag：`final_epoch_0010_step_002560`
  - split：`test`
  - seed：`1234`
  - inference steps：`20`
- Multi-test indices：`1`, `2`, `3`, `4`
- Multi-test grids：`results/strategy_c_multi_test/`
- BBox sweep indices：`0`, `1`, `2`
- BBox sweep grids：`results/strategy_c_bbox_sweep/`
- Result：7 grid images generated successfully for Strategy C across the multi-test and bbox-sweep groups.

### Large-Offset Corner BBox Comparison

- Motivation：the first bbox sweep used indices 0-2, but index 0 spans almost the full image width and index 2 spans the full image height. A horizontal left/center/right sweep is weak evidence on those samples.
- Chosen test indices：
  - index `3`：original bbox `[376, 320, 456, 464]`, prompt `add a cup`, bbox area ratio about `0.044`.
  - index `4`：original bbox `[312, 120, 376, 224]`, prompt `add a fire hydrant`, bbox area ratio about `0.025`.
  - index `5`：original bbox `[288, 224, 352, 288]`, prompt `Add an old square TV`, bbox area ratio about `0.016`.
- Sweep positions：`top_left`, `top_right`, `center`, `bottom_left`, `bottom_right`, `original`.
- Batch script：`scripts/infer_bbox_corner_compare.sh`
- Stack script：`scripts/stack_bbox_sweep_grids.py`
- Batch log：`logs/infer_bbox_corner_compare.log`
- Combined comparison output：`results/bbox_corner_compare/`
- Result：
  - A/B/C corner bbox sweep grids completed for indices `3`, `4`, `5`.
  - Stacked comparison images completed:
    - `results/bbox_corner_compare/bbox_corner_compare_test_3_seed1234.png`
    - `results/bbox_corner_compare/bbox_corner_compare_test_4_seed1234.png`
    - `results/bbox_corner_compare/bbox_corner_compare_test_5_seed1234.png`
- Follow-up batch：
  - Requested indices：`1`, `2`.
  - Same corner sweep positions and A/B/C stacked layout.
  - Log：`logs/infer_bbox_corner_compare_12.log`.
  - Result：
    - `results/bbox_corner_compare/bbox_corner_compare_test_1_seed1234.png`
    - `results/bbox_corner_compare/bbox_corner_compare_test_2_seed1234.png`

### Spoon Test 0 Custom and Original-Size BBox Sweeps

- Motivation：test index `0` is the spoon sample, but its original bbox `[0, 144, 504, 344]` is almost full image width.
- Small custom bbox sweep：
  - Mode：`small-corners`.
  - The small bbox preserves the original bbox aspect ratio and is limited to `224 x 112`; for test 0 it becomes `224 x 89`.
  - Positions：`top_left`, `top_right`, `center`, `bottom_left`, `bottom_right`, `original`.
- Original-size vertical sweep：
  - Mode：`vertical`.
  - Original bbox size remains `504 x 200`.
  - Positions：`top`, `middle`, `bottom`, `original`.
- Batch script：`scripts/infer_spoon_bbox_compare.sh`
- Batch log：`logs/infer_spoon_bbox_compare.log`
- Combined output target：`results/bbox_spoon_compare/`
- Result：
  - Small custom bbox A/B/C comparison：`results/bbox_spoon_compare/spoon_small_bbox_compare_test_0_seed1234.png`
  - Original-size vertical A/B/C comparison：`results/bbox_spoon_compare/spoon_original_vertical_compare_test_0_seed1234.png`

### 風險與處理

- PIPE dataset 與 model cache 很大，需設定到大容量磁碟或 Hugging Face cache，不放入 project submission。
- Starter code 預設 `/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def`，若本機不存在，需設定 `PIPE_HF_CACHE_DIR` 或調整可設定的 output/cache path。
- `evaluate.py` 目前空白，後續需補正式 metrics pipeline。
- Streaming 讀 PIPE 第一筆 sample 可行，但 Python 結束時發生 `PyGILState_Release` crash；後續正式資料讀取建議使用一般 cache 模式，並從 test split smoke test 開始。
