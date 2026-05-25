# Codex Log

## 2026-05-20 初始盤點與環境準備

### 做了什麼

- 以 `/home/chrceu/Desktop/master/1-2/DL/lab/final_project` 作為 project root。
- 盤點專案結構、文件、framework source code、環境檔。
- 讀取 proposal PDF、paper PDF 的重點文字。
- 檢查 Python / conda / GPU / compiler / tmux 狀態。
- 建立工作資料夾：`download/`、`data/`、`configs/`、`scripts/`、`logs/`、`checkpoints/`、`results/curves/`、`outputs/`。
- 建立初始文件：`DOC/rule.md`、`DOC/implementation_plan.md`、`DOC/code_change.md`、`DOC/experiment_log.md`。

### 為什麼做

- 使用者要求開始 final project 前先整理規則、讀 README/spec/PDF/starter code/config，並建立 `DOC/` log 規範。
- 使用者要求用 `framework/environment.yml` 建立名為 `DL_Final` 的專用環境，後續都在此環境工作。

### 實際執行過的 command

```bash
pwd && find . -maxdepth 3 -print | sed 's#^./##' | sort | head -200
rg --files -g 'README*' -g '*.md' -g '*.txt' -g '*.pdf' -g '*.yml' -g '*.yaml' -g '*.json' -g '*.toml' -g '*.py' -g '*.ipynb'
git status --short
sed -n '1,240p' framework/environment.yml
sed -n '1,240p' DOC/proposal.md
wc -l framework/sourcecode/*.py framework/sourcecode/model/*.py
rg -n "^(class|def|if __name__|import |from |parser|    parser|@)" framework/sourcecode -g '*.py'
ls -lh DOC framework framework/sourcecode framework/sourcecode/model
pdfinfo DOC/2404.18212v3.pdf
pdfinfo DOC/Group14_Proposal.pptx.pdf
find . -maxdepth 4 -type f -printf '%p %k KB\n' | sort
pdftotext -layout DOC/Group14_Proposal.pptx.pdf - | sed -n '1,260p'
pdftotext -layout DOC/2404.18212v3.pdf - | sed -n '1,260p'
pdftotext -layout DOC/2404.18212v3.pdf - | rg -n "(Abstract|Introduction|Method|Dataset|Evaluation|Experiment|PIPE|paint|inpaint|mask|bbox|bounding|DINO|InstructPix2Pix|Conclusion|Limitations|metric|LPIPS|CLIP|L1|IoU)"
pdftotext -layout DOC/2404.18212v3.pdf - | sed -n '293,560p'
pdftotext -layout DOC/2404.18212v3.pdf - | sed -n '948,990p'
pdftotext -layout DOC/2404.18212v3.pdf - | sed -n '796,835p'
sed -n '1,220p' framework/sourcecode/dataset.py
sed -n '1,260p' framework/sourcecode/pipe_bbox_dataset.py
sed -n '1,180p' framework/sourcecode/model/modelA.py
sed -n '1,140p' framework/sourcecode/model/modelB.py
sed -n '1,150p' framework/sourcecode/model/modelC.py
sed -n '1,280p' framework/sourcecode/train.py
sed -n '1,260p' framework/sourcecode/train_strategy_b.py
sed -n '1,330p' framework/sourcecode/train_strategy_c.py
sed -n '1,170p' framework/sourcecode/run_paint_by_inpaint.py
sed -n '1,170p' framework/sourcecode/export_pipe_samples.py
sed -n '160,360p' framework/sourcecode/infer_strategy_a.py
sed -n '188,340p' framework/sourcecode/infer_strategy_b.py
sed -n '160,320p' framework/sourcecode/infer_strategy_c.py
python --version && command -v python
command -v conda && conda --version && conda env list
command -v nvidia-smi && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
command -v node || true; command -v npm || true
command -v gcc && gcc --version | sed -n '1p' || true
command -v g++ && g++ --version | sed -n '1p' || true
command -v tmux && tmux -V || true
command -v python3 && python3 --version
df -h . /home/chrceu/miniconda3 2>/dev/null
mkdir -p DOC download data configs scripts logs checkpoints results/curves outputs
```

### 重要結果摘要

- Project root 正確：`/home/chrceu/Desktop/master/1-2/DL/lab/final_project`。
- 目前不是 git repository，因此沒有 git 版本追蹤資訊可用。
- 初始讀取 `DOC/proposal.md` 時沒有輸出；後續重新檢查發現檔案有 445 bytes，內容是中文 proposal note，重點是以 Paint by Inpaint 為主軸，額外加入 bbox conditioning，dataset 使用 PIPE。
- Paper PDF：21 pages；Proposal PDF：17 pages。
- `framework/environment.yml` 原始 env name 是 `dlp-pipe`，但會依使用者要求用 `conda env create -n DL_Final -f framework/environment.yml` 覆蓋建立名稱。
- Base shell 沒有 `python` 指令，但有 `/usr/bin/python3`：Python 3.12.3。
- Conda：`/home/chrceu/miniconda3/condabin/conda`，version `26.1.1`。
- 現有 conda env 內沒有 `DL_Final`。
- GPU：NVIDIA GeForce RTX 3090，driver `590.48.01`，VRAM 24576 MiB。
- GCC/G++：13.3.0。
- tmux：3.4。
- Node/npm 未偵測到。
- 根目錄磁碟約 731G 可用。

### 目前進度

- 初始文件與資料夾已建立。
- `DL_Final` conda 環境已建立完成。
- 已完成 package import、CUDA matmul、framework import、py_compile smoke test。

### 遇到的問題與處理方式

- `python` command 不存在：改查 `python3`，系統 Python 為 3.12.3；後續會使用 conda env 內的 Python。
- `git status` 顯示不是 git repo：後續改以文件 log 記錄改動。
- Starter code 硬編碼 `/media/zia/...` 大磁碟路徑：先記為風險，等環境建立後再確認是否需要設定 `PIPE_HF_CACHE_DIR`。

## 2026-05-20 建立 `DL_Final` 環境與 smoke test

### 做了什麼

- 用 `framework/environment.yml` 建立 conda env `DL_Final`。
- 因為安裝 PyTorch CUDA wheel 可能較久，使用 tmux session `final_project_env`，完整輸出 tee 到 `logs/env_create_DL_Final.log`。
- 在 `DL_Final` 中驗證 Python、pip、PyTorch CUDA、主要套件 import。
- 對 starter code 做 `py_compile`。
- 用 `PYTHONPATH=framework` 做 framework import smoke test。

### 為什麼做

- 使用者明確要求建立名為 `DL_Final` 的環境，並在此環境工作。
- 在開始訓練或修改程式前，先確認 dependency、GPU、starter code 基本可用。

### 實際執行過的 command

```bash
tmux has-session -t final_project_env 2>/dev/null; printf '%s\n' $?
conda env list | rg '^DL_Final\s' || true
tmux new-session -d -s final_project_env -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda env create -n DL_Final -f framework/environment.yml 2>&1 | tee -a logs/env_create_DL_Final.log'"
tmux capture-pane -t final_project_env -p | tail -80
tail -80 logs/env_create_DL_Final.log 2>/dev/null || true
tmux list-sessions | rg '^final_project_env' || true
du -sh /home/chrceu/miniconda3/envs/DL_Final 2>/dev/null || true
pgrep -af 'conda|pip|python' | rg 'DL_Final|env create|pip|conda' || true
conda env list | rg '^DL_Final\s' || true
tail -160 logs/env_create_DL_Final.log 2>/dev/null || true
ls -1 /home/chrceu/miniconda3/envs/DL_Final/bin/python /home/chrceu/miniconda3/envs/DL_Final/bin/pip 2>/dev/null
conda run -n DL_Final python --version && conda run -n DL_Final pip --version
conda run -n DL_Final python -m py_compile framework/sourcecode/*.py framework/sourcecode/model/*.py && printf 'py_compile_ok\n'
conda run -n DL_Final python -c "import sys, torch, torchvision, diffusers, transformers, accelerate, peft, safetensors, datasets, PIL, fsspec; print('executable', sys.executable); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('cuda_device_count', torch.cuda.device_count()); print('cuda_device_0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__); print('safetensors', safetensors.__version__); print('datasets', datasets.__version__); print('pillow', PIL.__version__); print('fsspec', fsspec.__version__)"
conda run -n DL_Final python -c "import torch; print('cpu_sum', round(float(torch.randn(2,3).sum()), 6)); print('cuda_available', torch.cuda.is_available()); import sys; sys.stdout.flush(); x=torch.randn(64,64,device='cuda') if torch.cuda.is_available() else None; y=(x @ x.T) if x is not None else None; torch.cuda.synchronize() if x is not None else None; print('cuda_matmul_ok', tuple(y.shape), str(y.dtype)) if y is not None else print('cuda_matmul_skipped')"
PIPE_HF_CACHE_DIR=/home/chrceu/Desktop/master/1-2/DL/lab/final_project/data/hf_cache_smoke PYTHONPATH=/home/chrceu/Desktop/master/1-2/DL/lab/final_project/framework conda run -n DL_Final python -c "import sourcecode.dataset, sourcecode.pipe_bbox_dataset, sourcecode.model.modelA, sourcecode.model.modelB, sourcecode.model.modelC, sourcecode.train, sourcecode.train_strategy_b, sourcecode.train_strategy_c, sourcecode.infer_strategy_a, sourcecode.infer_strategy_b, sourcecode.infer_strategy_c, sourcecode.run_paint_by_inpaint, sourcecode.export_pipe_samples; print('framework_import_ok')"
find /home/chrceu/Desktop/master/1-2/DL/lab/DOC -maxdepth 1 -type f -printf '%p %s bytes\n' 2>/dev/null | sort
mv /home/chrceu/Desktop/master/1-2/DL/lab/DOC/*.md /home/chrceu/Desktop/master/1-2/DL/lab/final_project/DOC/ && rmdir /home/chrceu/Desktop/master/1-2/DL/lab/DOC
find DOC -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

### 重要結果摘要

- `DL_Final` 建立成功：`/home/chrceu/miniconda3/envs/DL_Final`。
- 環境大小約 `5.6G`。
- 安裝 log：`logs/env_create_DL_Final.log`。
- Python：`3.11.15`。
- pip：`26.1.1`。
- torch：`2.12.0+cu130`。
- torchvision：`0.27.0+cu130`。
- CUDA available：`True`。
- CUDA device：`NVIDIA GeForce RTX 3090`。
- diffusers：`0.37.1`。
- transformers：`5.8.1`。
- accelerate：`1.13.0`。
- peft：`0.19.1`。
- safetensors：`0.7.0`。
- datasets：`4.8.5`。
- pillow：`12.2.0`。
- fsspec：`2026.2.0`。
- CUDA smoke test：`cuda_matmul_ok (64, 64) torch.float32`。
- Starter code compile：`py_compile_ok`。
- Framework import smoke：`framework_import_ok`。

### 目前進度

- 環境設定已完成，可以用 `conda activate DL_Final` 進入工作環境。
- 下一步建議做 dataset smoke test；需先確認大型 cache 路徑，例如設定 `PIPE_HF_CACHE_DIR` 到有足夠空間的位置。

### 遇到的問題與處理方式

- `framework/environment.yml` 原始 env name 是 `dlp-pipe`：使用 `conda env create -n DL_Final -f framework/environment.yml` 依使用者指定覆蓋環境名稱。
- `conda run` 搭配 heredoc 沒有印出預期輸出：改用 `python -c` 完成版本與 CUDA 檢查。
- Import `sourcecode.dataset` 會檢查 cache/mount：smoke test 時設定 `PIPE_HF_CACHE_DIR` 到 project 內的 `data/hf_cache_smoke`，避免觸發預設 `/media/zia/...` mount error。此目錄目前沒有下載大型資料。
- 修正一次路徑落點問題：`apply_patch` 的相對路徑一開始落到外層 `/home/chrceu/Desktop/master/1-2/DL/lab/DOC`。確認該資料夾只有本次 Codex 產生的 5 個 markdown 後，已移回 project root 的 `final_project/DOC/`，並移除外層誤建的 `DOC/`。

## 2026-05-20 重新閱讀 proposal.md 與 proposal PDF

### 做了什麼

- 重新讀取 `DOC/proposal.md`。
- 重新讀取 `DOC/Group14_Proposal.pptx.pdf` 的 metadata 與文字內容。
- 交叉確認 markdown note 與簡報內容是否一致。

### 為什麼做

- 使用者要求再讀一次 `proposal.md` 以及 proposal PDF，以確認 final project 方向。

### 實際執行過的 command

```bash
sed -n '1,220p' DOC/proposal.md
pdfinfo DOC/Group14_Proposal.pptx.pdf
pdftotext -layout DOC/Group14_Proposal.pptx.pdf - | sed -n '1,260p'
```

### 重要結果摘要

- `DOC/proposal.md` 重點：專案主軸環繞 Paint by Inpaint；原文以 conditional diffusion model 使用輸入影像與文字 prompt 作為條件；本專案差異是額外加入 bbox 作為條件；proposal 提出三種 bbox conditioning 方法；dataset 使用 PIPE。
- `DOC/Group14_Proposal.pptx.pdf`：17 pages，title 為 `Group14_Proposal.pptx`。
- PDF 題目：`Controllable Object Addition for Text-guided Image Editing using Spatial Conditions`。
- Task motivation：現有 diffusion-based image editing 可由文字 prompt 真實新增物件，但物件位置由模型自動決定，使用者無法明確控制 location 或 scale。
- Goal：`bounding-box guidance for controllable object addition`。
- Related work：mask-based editing、mask-free editing、instruction-based editing；核心參考 Paint by Inpaint / Stable Diffusion 1.5。
- Methods：Strategy A input-level conditioning、Strategy B attention-level conditioning、Strategy C feature-level conditioning。
- Dataset：training/validation 使用 PIPE Dataset；benchmark/generalization 使用 MagicBrush Dataset。
- Expected metrics：CLIP Score、edit-region IoU、outside LPIPS、inside LPIPS。

### 目前進度

- Proposal 方向已再次確認：在 Paint by Inpaint base 上做 bbox-guided controllable object addition。

## 2026-05-20 PIPE 資料來源與載入方式調查

### 做了什麼

- 查論文中與 PIPE dataset 相關的描述。
- 查 framework 內的 `dataset.py`、`pipe_bbox_dataset.py`、`export_pipe_samples.py`。
- 查官方 GitHub / Hugging Face dataset card。
- 用 Hugging Face API 查 `paint-by-inpaint/PIPE` 與 `paint-by-inpaint/PIPE_Masks` metadata。
- 用 streaming 方式讀取 test split 第一筆 sample，避免完整下載資料集。

### 為什麼做

- 使用者詢問資料在哪裡、怎麼載入，以及論文是否有提供資料相關資訊。

### 實際執行過的 command

```bash
pdftotext -layout DOC/2404.18212v3.pdf - | rg -n "(PIPE Dataset|PIPE dataset|Paint by InPaint|paint-by-inpaint|huggingface|hf.co|project page|dataset|Dataset|PIPE_Masks|download|released|1 million|889,230|1,879,919|COCO|Open Images|LVIS)"
sed -n '1,220p' framework/sourcecode/dataset.py
sed -n '1,240p' framework/sourcecode/pipe_bbox_dataset.py
sed -n '1,170p' framework/sourcecode/export_pipe_samples.py
find /media -maxdepth 3 -type d -printf '%p\n' 2>/dev/null | head -80
df -h . /media/* 2>/dev/null || true
/home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo_id in ['paint-by-inpaint/PIPE', 'paint-by-inpaint/PIPE_Masks']:
    info = api.dataset_info(repo_id, files_metadata=True)
    print(repo_id, info.sha, info.last_modified, info.downloads, info.likes)
    parquet = [s for s in info.siblings if s.rfilename.endswith('.parquet')]
    print(len(parquet), sum((s.size or 0) for s in parquet))
PY
/home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder
for repo_id in ['paint-by-inpaint/PIPE', 'paint-by-inpaint/PIPE_Masks']:
    print(repo_id)
    print(get_dataset_config_names(repo_id))
    print(get_dataset_split_names(repo_id))
    print(load_dataset_builder(repo_id).info.features)
PY
PIPE_HF_CACHE_DIR=/home/chrceu/Desktop/master/1-2/DL/lab/final_project/data/hf_cache_smoke /home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from datasets import load_dataset
pipe = load_dataset('paint-by-inpaint/PIPE', split='test', streaming=True)
mask = load_dataset('paint-by-inpaint/PIPE_Masks', split='test', streaming=True)
print(next(iter(pipe)))
print(next(iter(mask)))
PY
du -sh data data/hf_cache_smoke 2>/dev/null || true
df -h .
```

### 重要結果摘要

- 論文有描述 PIPE dataset：利用 COCO、Open Images、LVIS segmentation masks 和 Stable Diffusion inpainting 產生 source-target pairs；約 889,230 unique images、超過 1,400 classes、約 1M image pairs、近 1.9M instructions。
- 實際資料下載在 Hugging Face：
  - `paint-by-inpaint/PIPE`
  - `paint-by-inpaint/PIPE_Masks`
- 官方 GitHub 也指向 Hugging Face，並提到 PIPE test set 另有 Google Drive 版本。
- `paint-by-inpaint/PIPE` metadata：
  - splits：`train`, `test`
  - parquet files：163
  - parquet size：約 74.94 GB
  - download_size：80,461,136,624 bytes
  - dataset_size：80,872,351,058.76 bytes
  - columns：`source_img`, `target_img`, `Instruction_VLM-LLM`, `Instruction_Class`, `Instruction_Ref_Dataset`, `object_location`, `target_img_dataset`, `img_id`, `ann_id`
- `paint-by-inpaint/PIPE_Masks` metadata：
  - splits：`train`, `test`
  - parquet files：7
  - parquet size：約 1.72 GB
  - download_size：681,492,456 bytes
  - dataset_size：2,591,591,666.36 bytes
  - columns：`mask`, `target_img_dataset`, `img_id`, `ann_id`
- Streaming 第一筆 test sample：
  - source/target image：512x512 RGB
  - instruction：`Add a metallic spoon`
  - target dataset：COCO
  - `img_id=129113`, `ann_id=1529722`
  - mask：512x512 L
- Streaming 成功印出 sample 後，Python finalizing 階段發生 `PyGILState_Release` crash，exit code 134。初步判斷是 streaming / pyarrow / zstd teardown 問題；正式訓練建議使用一般 cache 模式，而非 streaming。
- Project 所在磁碟約 724GB 可用，容量可容納資料集，但大型 dataset 不應放入最終提交。

### 目前建議

- 正式下載/訓練前先設定：

```bash
export PIPE_HF_CACHE_DIR=/home/chrceu/Desktop/master/1-2/DL/lab/final_project/data/hf_cache
export HF_HOME=$PIPE_HF_CACHE_DIR/home
export HF_DATASETS_CACHE=$PIPE_HF_CACHE_DIR
export HF_HUB_CACHE=$PIPE_HF_CACHE_DIR/hub
```

- 若只要先看 test sample，可先下載/載入 `test` split；完整 train 會抓約 75GB PIPE + 1.7GB masks。

## 2026-05-20 開始下載完整 PIPE data

### 做了什麼

- 新增 `scripts/download_pipe_data.py`。
- 確認 project 所在磁碟約 724GB 可用。
- 啟動 tmux session `final_project_pipe_download` 下載完整 `paint-by-inpaint/PIPE` 與 `paint-by-inpaint/PIPE_Masks`。
- 下載 cache 位置：`data/hf_cache`。
- 下載 log：`logs/download_pipe_data.log`。

### 為什麼做

- 使用者要求直接完成下載 PIPE dataset。
- PIPE 本體約 75GB，PIPE_Masks 約 1.7GB，屬於 long-running task，所以用 tmux 並 tee log。

### 實際執行過的 command

```bash
df -h . data /home/chrceu/.cache/huggingface 2>/dev/null || true
du -sh data data/hf_cache data/hf_cache_smoke 2>/dev/null || true
tmux has-session -t final_project_pipe_download 2>/dev/null; printf '%s\n' $?
tmux new-session -d -s final_project_pipe_download -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; stdbuf -oL -eL python scripts/download_pipe_data.py --cache-dir data/hf_cache --splits test train --repos PIPE PIPE_Masks 2>&1 | tee -a logs/download_pipe_data.log'"
tmux capture-pane -t final_project_pipe_download -p | tail -100
tail -100 logs/download_pipe_data.log 2>/dev/null || true
du -sh data data/hf_cache 2>/dev/null || true
```

### 重要結果摘要

- tmux session：`final_project_pipe_download`
- 監控指令：

```bash
tmux attach -t final_project_pipe_download
tmux capture-pane -t final_project_pipe_download -p | tail -120
tail -f logs/download_pipe_data.log
du -sh data/hf_cache
```

- 起始 metadata：
  - `paint-by-inpaint/PIPE`：163 parquet files，約 80,461,136,624 bytes。
  - `paint-by-inpaint/PIPE_Masks`：7 parquet files，約 1,845,016,264 bytes。
- 啟動後，`load_dataset` 顯示正在下載 PIPE parquet files。

### 目前進度

- 已於 2026-05-20 21:40:23 完成下載與 HF dataset cache 生成。
- 已驗證 `PIPEBBoxDataset(split="test")` 可讀 sample。

## 2026-05-20 完成 PIPE data 下載與驗證

### 做了什麼

- 持續監控 tmux session `final_project_pipe_download` 到結束。
- 完成 `paint-by-inpaint/PIPE` 與 `paint-by-inpaint/PIPE_Masks` 的 `test` / `train` split cache。
- 確認資料主要存在 project 內 `data/hf_cache`，不是預設 `~/.cache/huggingface`。
- 用 `DL_Final` 環境做 Hugging Face dataset 離線載入驗證。
- 用 framework 的 `PIPEBBoxDataset(split="test")` 做 sample smoke test。

### 為什麼做

- 使用者要求直接完成 PIPE dataset 下載。
- 後續 bbox-guided object addition training 需要 PIPE source/target image pairs 與 PIPE_Masks 產生 object mask / bbox。

### 實際執行過的 command

```bash
tmux capture-pane -t final_project_pipe_download -p -S -260 | tail -20
du -sh data/hf_cache
df -h .
while tmux has-session -t final_project_pipe_download 2>/dev/null; do date '+%F %T'; tmux capture-pane -t final_project_pipe_download -p -S -260 | rg 'Downloading data|Generating|===|ERROR|Traceback' | tail -5; du -sh data/hf_cache; sleep 60; done; date '+%F %T'; echo 'tmux session ended'; tail -80 logs/download_pipe_data.log
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m json.tool data/hf_cache/download_pipe_data_summary.json | sed -n '1,240p'
du -sh data/hf_cache logs/download_pipe_data.log DOC || true
du -sh ~/.cache/huggingface 2>/dev/null || true
PIPE_HF_CACHE_DIR="$PWD/data/hf_cache" HF_HOME="$PWD/data/hf_cache/home" HF_DATASETS_CACHE="$PWD/data/hf_cache" HF_HUB_CACHE="$PWD/data/hf_cache/hub" HF_XET_CACHE="$PWD/data/hf_cache/xet" PYTHONPATH="$PWD/framework" /home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from datasets import DownloadConfig, load_dataset
cache = 'data/hf_cache'
dc = DownloadConfig(local_files_only=True)
for repo in ['paint-by-inpaint/PIPE', 'paint-by-inpaint/PIPE_Masks']:
    for split in ['test', 'train']:
        ds = load_dataset(repo, split=split, cache_dir=cache, download_config=dc)
        sample = ds[0]
        print(f'{repo} {split}: rows={len(ds):,}, keys={list(sample.keys())}')
print('offline_load_ok')
PY
PIPE_HF_CACHE_DIR="$PWD/data/hf_cache" HF_HOME="$PWD/data/hf_cache/home" HF_DATASETS_CACHE="$PWD/data/hf_cache" HF_HUB_CACHE="$PWD/data/hf_cache/hub" HF_XET_CACHE="$PWD/data/hf_cache/xet" PYTHONPATH="$PWD/framework" /home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from sourcecode.pipe_bbox_dataset import PIPEBBoxDataset
ds = PIPEBBoxDataset(split='test', image_size=512)
s = ds[0]
print('dataset_len', len(ds))
print('source_img_shape', tuple(s['source_img'].shape))
print('target_img_shape', tuple(s['target_img'].shape))
print('object_mask_shape', tuple(s['object_mask'].shape))
print('bbox_mask_shape', tuple(s['bbox_mask'].shape))
print('bbox', s['bbox'].tolist())
print('instruction', s['instruction'])
print('ids', s['img_id'], s['ann_id'])
print('framework_dataset_smoke_ok')
PY
```

### 重要結果摘要

- tmux session 正常結束，最後顯示 `download complete`。
- Summary 檔案：`data/hf_cache/download_pipe_data_summary.json`。
- Download log：`logs/download_pipe_data.log`。
- `data/hf_cache` 大小：約 155G。
- `~/.cache/huggingface` 大小：約 64K，確認大型 dataset cache 沒有放到 home 預設 cache。
- 磁碟狀態：project 所在 partition 約 937G，已用 320G，剩餘 570G。
- `paint-by-inpaint/PIPE`：
  - train rows：888,230
  - test rows：752
  - image：512x512 RGB
  - columns：`source_img`, `target_img`, `Instruction_VLM-LLM`, `Instruction_Class`, `Instruction_Ref_Dataset`, `object_location`, `target_img_dataset`, `img_id`, `ann_id`
- `paint-by-inpaint/PIPE_Masks`：
  - train rows：888,230
  - test rows：752
  - mask：512x512 L
  - columns：`mask`, `target_img_dataset`, `img_id`, `ann_id`
- 離線載入驗證結果：`offline_load_ok`。
- Framework dataset smoke test：
  - `dataset_len 752`
  - `source_img_shape (3, 512, 512)`
  - `target_img_shape (3, 512, 512)`
  - `object_mask_shape (1, 512, 512)`
  - `bbox_mask_shape (1, 512, 512)`
  - first bbox：`[0.0, 144.0, 504.0, 344.0]`
  - first instruction：`Add a metallic spoon`
  - result：`framework_dataset_smoke_ok`

### 遇到的問題與處理方式

- 下載中曾遇到一次 Hugging Face `HTTP Error 504`，下載器自動 retry 並成功繼續，無需人工重跑。
- 一開始用 base shell 跑 `python -m json.tool` 時出現 `python: command not found`；改用 `/home/chrceu/miniconda3/envs/DL_Final/bin/python` 後成功。
- `load_dataset(..., DownloadConfig(local_files_only=True))` 仍顯示 unauthenticated HF warning，但沒有重新下載資料；dataset 從本地 cache 正常讀取。

### 目前進度

- PIPE / PIPE_Masks 完整資料已下載完成。
- Dataset smoke test 已完成。
- 下一步可開始小樣本 training smoke test，先確認 training loop、checkpoint、metrics CSV 與曲線輸出。

## 2026-05-20 Final Checklist for Data Download

### 實際執行過的 command

```bash
tmux list-sessions 2>/dev/null | rg '^final_project_pipe_download' || true
test -s data/hf_cache/download_pipe_data_summary.json && printf 'summary_exists\n'
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile scripts/download_pipe_data.py framework/sourcecode/pipe_bbox_dataset.py && printf 'py_compile_ok\n'
du -sh data/hf_cache logs/download_pipe_data.log DOC
find DOC -maxdepth 1 -type f -printf '%f\n' | sort
git status --short
```

### 重要結果摘要

- `final_project_pipe_download` tmux session 已結束。
- `summary_exists`。
- `py_compile_ok`。
- 大小：
  - `data/hf_cache`: 155G
  - `logs/download_pipe_data.log`: 96K
  - `DOC`: 19M
- `DOC/` 內包含：
  - `2404.18212v3.pdf`
  - `Group14_Proposal.pptx.pdf`
  - `proposal.md`
  - `rule.md`
  - `implementation_plan.md`
  - `codex_log.md`
  - `code_change.md`
  - `experiment_log.md`
- `git status --short` 顯示目前資料夾不是 git repository，因此沒有 repo status 可列。

### Final checklist

- 必做資料下載：完成。
- PIPE 與 PIPE_Masks split 筆數：已驗證。
- Framework dataset sample：已驗證。
- 文件與 log：已更新。
- 大型 dataset：位於 `data/hf_cache`，後續提交時應排除。
- 官方/starter source：未手動修改；只讀取與編譯檢查。

## 2026-05-21 進度查核與 Strategy B Training Smoke

### 做了什麼

- 回覆目前進度前，重新讀取 `DOC/codex_log.md`、`DOC/experiment_log.md` 與 project 目前檔案狀態。
- 確認環境與資料準備已完成，但 training / evaluation 尚未開始。
- 讀取 Strategy A/B/C training scripts 與 model loaders。
- 發現 model loaders 把 model cache 寫死在不存在的 `/media/zia/...` mount。
- 修改 model loaders 讓 `PIPE_MODEL_CACHE_DIR` 可指定本機 project cache。
- 為 Strategy B training 加入 `--metrics-csv`，先跑最小 1-step smoke training。

### 為什麼做

- 目前 project 的下一個風險點是 training pipeline 是否真的能從 PIPE data 與 pretrained model 走到 loss、checkpoint、metrics CSV。
- Strategy B 是 LoRA-only baseline，最適合先做最低成本的 smoke run。

### 實際執行過的 command

```bash
tail -120 DOC/codex_log.md
tail -120 DOC/experiment_log.md
find . -maxdepth 2 -type f | sort | sed -n '1,220p'
du -sh data/hf_cache logs DOC checkpoints results outputs 2>/dev/null || true
find framework -maxdepth 3 -type f | sort
rg -n "argparse|train|evaluate|infer|PIPEBBoxDataset|create_bbox|Strategy|strategy|checkpoint|metrics|csv" framework -g '!**/__pycache__/**'
sed -n '1,340p' framework/sourcecode/train_strategy_b.py
sed -n '1,220p' framework/sourcecode/model/modelB.py
test -d /media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def && printf 'data_disk_exists\n' || printf 'data_disk_missing\n'
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/model/modelA.py framework/sourcecode/model/modelB.py framework/sourcecode/model/modelC.py framework/sourcecode/train_strategy_b.py && printf 'py_compile_ok\n'
PIPE_HF_CACHE_DIR="$PWD/data/hf_cache" PIPE_MODEL_CACHE_DIR="$PWD/data/model_cache" PYTHONPATH="$PWD/framework" /home/chrceu/miniconda3/envs/DL_Final/bin/python framework/sourcecode/train_strategy_b.py --help
tmux new-session -d -s final_project_smoke_train_b -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL python framework/sourcecode/train_strategy_b.py --output-dir checkpoints/smoke_train_b --metrics-csv logs/smoke_train_b_metrics.csv --max-train-samples 1 --epochs 1 --batch-size 1 --num-workers 0 --image-size 256 --save-every 1 --fp16 2>&1 | tee -a logs/smoke_train_b.log'"
sed -n '1,40p' logs/smoke_train_b_metrics.csv
find checkpoints/smoke_train_b -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
```

### 重要結果摘要

- Current status at start of this pass：
  - `DL_Final` environment ready。
  - PIPE / PIPE_Masks data ready。
  - Dataset smoke test ready。
  - Training/evaluation/report not completed yet。
- Hard-coded model cache check：`data_disk_missing`。
- Compile / CLI verification：`py_compile_ok`，Strategy B CLI exposes `--metrics-csv`。
- GPU before run：RTX 3090 24GB available。
- tmux run：`final_project_smoke_train_b`；run 已正常結束。
- Model cache created at `data/model_cache`，目前約 2.0G。
- Strategy B smoke training：
  - U-Net parameters：860,329,668
  - Trainable U-Net parameters：797,184
  - Training samples：1
  - Optimization steps：1
  - Step 1 loss：`0.5391167402267456`
  - Checkpoint output：`checkpoints/smoke_train_b`
  - Training log：`logs/smoke_train_b.log`
  - Metrics CSV：`logs/smoke_train_b_metrics.csv`

### 目前進度

- Environment：完成。
- Proposal/paper/rules/data audit：完成。
- PIPE / PIPE_Masks download and dataset smoke：完成。
- Strategy B one-step training smoke：完成。
- Pending：Strategy A/C smoke or main Strategy C run、inference comparison、evaluation metrics、curves、report draft、submission checklist。

## 2026-05-21 Strategy A FP16 NaN 排錯

### 做了什麼

- 根據使用者貼出的 Strategy A training traceback，檢查 failure point。
- 用相同 Strategy A path 與 512px image size 做不帶 `--fp16` 的 2-step smoke。

### 實際執行過的 command

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
  --save-every 1 \
  2>&1 | tee logs/smoke_train_a_fp32.log
```

### 重要結果摘要

- 使用者的 failed run：
  - Strategy A 以 `--fp16` 跑到 epoch 1，第 1 step 顯示 loss 後，第 2 step 出現 non-finite output。
  - Error 摘要：`noise_pred finite=False model_input finite=True target_latents finite=True`。
- FP32 smoke run：
  - 2 training samples。
  - 2 optimization steps。
  - losses：約 `0.8218`, `0.9837`。
  - checkpoint 成功輸出到 `checkpoints/smoke_train_a_fp32`。
- 目前判斷：
  - 此 failure 不是 PIPE data download 問題。
  - 現有 Strategy A training path 直接以 fp16 訓練，沒有 mixed-precision loss scaling / fp32 master-weight 保護，較容易在 optimizer update 後產生 NaN/Inf。
  - 目前最直接的 workaround 是先移除 `--fp16` 跑 Strategy A。

## 2026-05-21 啟動 Strategy B 正式訓練

### 做了什麼

- 確認沒有同名 `final_project_train_b` tmux session，也沒有既有 `checkpoints/strategy_b` 正式 checkpoint。
- 以 tmux 啟動 Strategy B 256-sample / 10-epoch / 512px training。
- 保留 training log、metrics CSV 與 checkpoint 到 project 內。

### 實際執行過的 command

```bash
tmux new-session -d -s final_project_train_b -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL python framework/sourcecode/train_strategy_b.py --output-dir checkpoints/strategy_b --metrics-csv logs/strategy_b_metrics.csv --max-train-samples 256 --epochs 10 --batch-size 1 --num-workers 2 --image-size 512 --lr 5e-5 --lora-rank 4 --lora-alpha 4 --bbox-loss-weight 8.0 --save-every 1 2>&1 | tee -a logs/strategy_b_train.log'"
tmux capture-pane -t final_project_train_b -p -S -260 | tail -120
tail -80 logs/strategy_b_train.log
tail -5 logs/strategy_b_metrics.csv
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
```

### 重要結果摘要

- tmux session：`final_project_train_b`。
- Training log：`logs/strategy_b_train.log`。
- Metrics CSV：`logs/strategy_b_metrics.csv`。
- Checkpoint dir：`checkpoints/strategy_b`。
- Initial progress check：
  - Strategy B reached epoch 1, step 98/256.
  - GPU check：RTX 3090 using about 6164 MiB at 100% utilization during training.
  - Metrics CSV rows were being appended with `step`, `epoch`, `loss`, `metric`, `learning_rate`.

### 監控指令

```bash
tmux attach -t final_project_train_b
tmux capture-pane -t final_project_train_b -p | tail -120
tail -f logs/strategy_b_train.log
tail -f logs/strategy_b_metrics.csv
```

## 2026-05-22 B 完成後啟動 Strategy C

### 做了什麼

- 確認 Strategy B full run 已結束、checkpoint 與 metrics 完整。
- 為 Strategy C training 補上 `--metrics-csv`。
- 以 512px FP32 做 Strategy C one-step smoke，確認 ControlNet training 能在 RTX 3090 上跑通。
- 以 tmux 啟動 Strategy C 256-sample / 10-epoch full run。

### 實際執行過的 command

```bash
find checkpoints/strategy_b -maxdepth 2 -type d -printf '%p\n' | sort | tail -30
tail -8 logs/strategy_b_metrics.csv
wc -l logs/strategy_b_metrics.csv
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/train_strategy_c.py && printf 'py_compile_ok\n'
python framework/sourcecode/train_strategy_c.py --output-dir checkpoints/smoke_train_c_fp32_512 --metrics-csv logs/smoke_train_c_fp32_512_metrics.csv --max-train-samples 1 --epochs 1 --batch-size 1 --num-workers 0 --image-size 512 --save-every 1 2>&1 | tee logs/smoke_train_c_fp32_512.log
tmux new-session -d -s final_project_train_c -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL python framework/sourcecode/train_strategy_c.py --output-dir checkpoints/strategy_c --metrics-csv logs/strategy_c_metrics.csv --max-train-samples 256 --epochs 10 --batch-size 1 --num-workers 2 --image-size 512 --lr 1e-5 --bbox-loss-weight 8.0 --bbox-shift-prob 0.5 --controlnet-conditioning-scale 1.0 --save-every 1 2>&1 | tee -a logs/strategy_c_train.log'"
tmux capture-pane -t final_project_train_c -p -S -260 | tail -120
tail -5 logs/strategy_c_metrics.csv
```

### 重要結果摘要

- Strategy B：
  - Session 已結束。
  - Final checkpoint：`checkpoints/strategy_b/strategy_b_final_epoch_0010_step_002560_lora/`。
  - Metrics CSV 有 2561 lines，對應 header + 2560 steps。
  - Final metrics row loss：`0.08246304839849472`。
- Strategy C smoke：
  - 512px FP32 one-step loss：`0.6548`。
  - Checkpoint 已輸出到 `checkpoints/smoke_train_c_fp32_512`。
- Strategy C full run：
  - tmux session：`final_project_train_c`。
  - Initial progress：epoch 1, step 39/256。
  - GPU initial check：約 12410 MiB，utilization 98%。
  - Log：`logs/strategy_c_train.log`。
  - Metrics：`logs/strategy_c_metrics.csv`。
  - Checkpoints：`checkpoints/strategy_c/`。

### 監控指令

```bash
tmux attach -t final_project_train_c
tmux capture-pane -t final_project_train_c -p | tail -120
tail -f logs/strategy_c_train.log
tail -f logs/strategy_c_metrics.csv
```

## 2026-05-22 產生 Strategy A/B 測試圖

### 做了什麼

- 在 Strategy C full run 繼續訓練時，確認 GPU 尚有可用空間。
- 修正 A/B inference 的舊 mount check，讓它們可使用 `PIPE_HF_CACHE_DIR` 指向 project PIPE cache。
- 用 A/B final checkpoint 對同一筆 PIPE test sample 產圖。

### 實際執行過的 command

```bash
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/infer_strategy_a.py framework/sourcecode/infer_strategy_b.py && printf 'py_compile_ok\n'
tmux new-session -d -s final_project_infer_ab_index0 -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL python framework/sourcecode/infer_strategy_a.py --checkpoint-dir checkpoints/strategy_a --checkpoint-tag final_epoch_0010_step_002560 --output-dir results/strategy_a_infer --split test --index 0 --steps 20 --seed 1234 2>&1 | tee -a logs/infer_ab_index0.log; stdbuf -oL -eL python framework/sourcecode/infer_strategy_b.py --checkpoint-dir checkpoints/strategy_b --checkpoint-tag final_epoch_0010_step_002560 --output-dir results/strategy_b_infer --split test --index 0 --steps 20 --seed 1234 2>&1 | tee -a logs/infer_ab_index0.log'"
find results/strategy_a_infer results/strategy_b_infer -maxdepth 1 -type f -printf '%p %s bytes\n' | sort
cat results/strategy_a_infer/strategy_a_final_epoch_0010_step_002560_test_0_seed1234_info.txt
cat results/strategy_b_infer/strategy_b_final_epoch_0010_step_002560_test_0_seed1234_info.txt
```

### 重要結果摘要

- A/B inference 完成，log 在 `logs/infer_ab_index0.log`。
- 共用 sample：
  - prompt：`Add a metallic spoon`
  - test index：0
  - bbox：`[0.0, 144.0, 504.0, 344.0]`
  - seed：1234
  - steps：20
- A grid：
  - `results/strategy_a_infer/strategy_a_final_epoch_0010_step_002560_test_0_seed1234_grid.png`
- B grid：
  - `results/strategy_b_infer/strategy_b_final_epoch_0010_step_002560_test_0_seed1234_grid.png`
- C training remained active; follow-up metrics check reached epoch 3, step 612.

## 2026-05-22 A/B 多測資與 bbox sweep 批次生圖

### 做了什麼

- 確認 Strategy C full run 已完成，GPU 空出。
- 新增 `scripts/infer_ab_batch.sh`，批次生 A/B 多測資與 bbox sweep 圖。
- 用 tmux 跑完 batch inference。

### 實際執行過的 command

```bash
bash -n scripts/infer_ab_batch.sh && printf 'bash_syntax_ok\n'
tmux new-session -d -s final_project_infer_ab_batch -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL bash scripts/infer_ab_batch.sh 2>&1 | tee -a logs/infer_ab_batch.log'"
find results/strategy_a_multi_test results/strategy_b_multi_test results/strategy_a_bbox_sweep results/strategy_b_bbox_sweep -maxdepth 1 -name '*_grid.png' -type f -printf '%p\n' | sort
```

### 重要結果摘要

- Strategy C full run completed before batch inference:
  - final checkpoint：`checkpoints/strategy_c/strategy_c_final_epoch_0010_step_002560_controlnet/`
  - final metrics row loss：`0.017148436978459358`
- A/B batch inference completed:
  - Multi-test indices：1, 2, 3, 4
  - BBox sweep indices：0, 1, 2
  - Total grid images：14
  - Batch log：`logs/infer_ab_batch.log`
- Output folders：
  - `results/strategy_a_multi_test/`
  - `results/strategy_b_multi_test/`
  - `results/strategy_a_bbox_sweep/`
  - `results/strategy_b_bbox_sweep/`

## 2026-05-22 Strategy C 多測資與 bbox sweep 批次生圖

### 做了什麼

- 讀取 `infer_strategy_c.py` CLI 與 cache 設定，確認它支援 `--bbox-sweep`。
- 修正 Strategy C inference 的舊 mount check，讓它使用 `PIPE_HF_CACHE_DIR` 指到 project 內 PIPE cache。
- 新增 `scripts/infer_c_batch.sh`，使用與 A/B 相同的 test indices、seed 與 inference steps 跑 C。
- 以 tmux 啟動 Strategy C batch inference，避免長時間生圖綁住 shell。

### 實際執行過的 command

```bash
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/infer_strategy_c.py
bash -n scripts/infer_c_batch.sh
tmux new-session -d -s final_project_infer_c_batch -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL bash scripts/infer_c_batch.sh 2>&1 | tee -a logs/infer_c_batch.log'"
tmux capture-pane -t final_project_infer_c_batch -p | tail -80
tail -80 logs/infer_c_batch.log
find results/strategy_c_multi_test results/strategy_c_bbox_sweep -maxdepth 1 -name '*_grid.png' -type f -printf '%p\n' | sort
```

### 重要結果摘要

- Compile / shell syntax check passed.
- tmux session：`final_project_infer_c_batch`。
- Log：`logs/infer_c_batch.log`。
- Output folders：
  - `results/strategy_c_multi_test/`
  - `results/strategy_c_bbox_sweep/`
- Initial check：first Strategy C inference began loading the cached pipeline.
- Completion check：
  - Multi-test grids completed for indices 1, 2, 3, 4.
  - BBox sweep grids completed for indices 0, 1, 2.
  - Total Strategy C grid images from this batch：7.
  - tmux session exited after the last grid was written.

### 監控指令

```bash
tmux attach -t final_project_infer_c_batch
tmux capture-pane -t final_project_infer_c_batch -p | tail -120
tail -f logs/infer_c_batch.log
```

## 2026-05-22 大位移 bbox A/B/C 合併對照圖

### 做了什麼

- 檢查先前 bbox sweep 的 sample bbox，確認 indices 0 與 2 形狀太大，原本水平平移差異不夠顯眼。
- 掃描 test indices 0-39 的 bbox 尺寸，挑選 bbox 較小的 indices 3、4、5。
- 將 Strategy A/B/C bbox sweep 改為四角、中央、原始 bbox 的 large-offset 測試。
- 新增 batch script，讓同一筆 test sample 依序跑 A/B/C，並把三張 sweep grid 疊成一張比較圖。

### 實際執行過的 command

```bash
PIPE_HF_CACHE_DIR="$PWD/data/hf_cache" HF_HOME="$PWD/data/hf_cache/home" HF_DATASETS_CACHE="$PWD/data/hf_cache" HF_HUB_CACHE="$PWD/data/hf_cache/hub" HF_XET_CACHE="$PWD/data/hf_cache/xet" PIPE_MODEL_CACHE_DIR="$PWD/data/model_cache" PYTHONPATH="$PWD/framework" /home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from sourcecode.pipe_bbox_dataset import PIPEBBoxDataset

dataset = PIPEBBoxDataset(split='test', image_size=512)
for index in range(40):
    sample = dataset[index]
    x1, y1, x2, y2 = sample['bbox'].tolist()
    w, h = x2 - x1, y2 - y1
    area = w * h / (512 * 512)
    print(f"{index:02d} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) wh=({w:.0f},{h:.0f}) area={area:.3f} prompt={sample['instruction']}")
PY
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/infer_strategy_a.py framework/sourcecode/infer_strategy_b.py framework/sourcecode/infer_strategy_c.py scripts/stack_bbox_sweep_grids.py
bash -n scripts/infer_bbox_corner_compare.sh
tmux new-session -d -s final_project_bbox_corner_compare -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL bash scripts/infer_bbox_corner_compare.sh 2>&1 | tee -a logs/infer_bbox_corner_compare.log'"
```

### 重要結果摘要

- Test sample scan selected indices 3, 4, 5 because their bbox area ratios are about 4.4%, 2.5%, and 1.6%.
- Compile and shell syntax checks passed.
- tmux session：`final_project_bbox_corner_compare`。
- Log：`logs/infer_bbox_corner_compare.log`。
- Combined output target：`results/bbox_corner_compare/`。
- Completion check：
  - A/B/C per-strategy corner sweep grids completed for test indices 3, 4, 5.
  - Stacked comparison images completed for all three indices.
  - Viewed `results/bbox_corner_compare/bbox_corner_compare_test_4_seed1234.png` to verify the A/B/C rows and corner bbox overlays are visible.
  - tmux session exited after the last comparison image was written.

## 2026-05-22 補 test 1/2 corner bbox 合併圖

### 做了什麼

- 將 `scripts/infer_bbox_corner_compare.sh` 改成可由 command line 指定 test indices，預設仍保留 indices 3、4、5。
- 對使用者指定的 test indices 1、2 啟動同一套 A/B/C corner bbox sweep 與 stacked comparison。

### 實際執行過的 command

```bash
bash -n scripts/infer_bbox_corner_compare.sh
tmux new-session -d -s final_project_bbox_corner_compare_12 -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL bash scripts/infer_bbox_corner_compare.sh 1 2 2>&1 | tee -a logs/infer_bbox_corner_compare_12.log'"
```

### 重要結果摘要

- Shell syntax check passed.
- tmux session：`final_project_bbox_corner_compare_12`。
- Log：`logs/infer_bbox_corner_compare_12.log`。
- Combined output target：`results/bbox_corner_compare/`。
- Completion check：
  - A/B/C per-strategy corner sweep grids completed for test indices 1 and 2.
  - Stacked comparison images completed:
    - `results/bbox_corner_compare/bbox_corner_compare_test_1_seed1234.png`
    - `results/bbox_corner_compare/bbox_corner_compare_test_2_seed1234.png`
  - tmux session exited after the test 2 comparison image was written.

## 2026-05-22 Spoon test 0 自訂小 bbox 與原 bbox 上中下

### 做了什麼

- 為 Strategy A/B/C inference 新增 `small-corners` 與 `vertical` bbox sweep modes。
- Spoon test 0 縮小 bbox 後保留長寬比做 corner sweep，另外保留原 bbox 大小做 top/middle/bottom/original vertical sweep。
- 新增 spoon batch script，並讓 grid stack script 可讀不同 strategy output suffix。

### 實際執行過的 command

```bash
/home/chrceu/miniconda3/envs/DL_Final/bin/python -m py_compile framework/sourcecode/infer_strategy_a.py framework/sourcecode/infer_strategy_b.py framework/sourcecode/infer_strategy_c.py scripts/stack_bbox_sweep_grids.py
bash -n scripts/infer_spoon_bbox_compare.sh scripts/infer_bbox_corner_compare.sh
PYTHONPATH="$PWD/framework" /home/chrceu/miniconda3/envs/DL_Final/bin/python - <<'PY'
from sourcecode.infer_strategy_a import select_bbox_sweep
bbox = [0, 144, 504, 344]
for mode in ('small-corners', 'vertical'):
    print(mode)
    for name, box in select_bbox_sweep(bbox, (512, 512), mode).items():
        print(f'  {name}: {box}')
PY
tmux new-session -d -s final_project_spoon_bbox_compare -c /home/chrceu/Desktop/master/1-2/DL/lab/final_project "bash -lc 'set -o pipefail; source /home/chrceu/miniconda3/etc/profile.d/conda.sh; conda activate DL_Final; export PIPE_HF_CACHE_DIR=\"$PWD/data/hf_cache\"; export HF_HOME=\"$PWD/data/hf_cache/home\"; export HF_DATASETS_CACHE=\"$PWD/data/hf_cache\"; export HF_HUB_CACHE=\"$PWD/data/hf_cache/hub\"; export HF_XET_CACHE=\"$PWD/data/hf_cache/xet\"; export PIPE_MODEL_CACHE_DIR=\"$PWD/data/model_cache\"; export PYTHONPATH=\"$PWD/framework\"; stdbuf -oL -eL bash scripts/infer_spoon_bbox_compare.sh 2>&1 | tee -a logs/infer_spoon_bbox_compare.log'"
```

### 重要結果摘要

- Compile and shell syntax checks passed.
- Test 0 small bbox coordinates include `top_left=(0,0,224,89)` and `bottom_right=(288,423,512,512)`.
- Original-size vertical coordinates are `top=(0,0,504,200)`, `middle=(0,156,504,356)`, `bottom=(0,312,504,512)`, `original=(0,144,504,344)`.
- tmux session：`final_project_spoon_bbox_compare`。
- Log：`logs/infer_spoon_bbox_compare.log`。
- Completion check：
  - Small custom bbox comparison completed at `results/bbox_spoon_compare/spoon_small_bbox_compare_test_0_seed1234.png`.
  - Original-size vertical comparison completed at `results/bbox_spoon_compare/spoon_original_vertical_compare_test_0_seed1234.png`.
  - Viewed both spoon comparison images to verify A/B/C rows, bbox overlays, and sweep ordering.
  - tmux session exited after the original-size vertical comparison was written.

## 2026-05-25 GitHub 打包準備

### 做了什麼

- 盤點 project 大小與各資料夾用途。
- 確認 `data/` 約 157 GB、`checkpoints/` 約 18 GB，不適合直接推 GitHub。
- 新增 `.gitignore`，避免大型 cache/checkpoints/results/logs 被誤 commit。
- 新增 `README.md`，讓隊友從 GitHub 進入後可快速看到 setup、dataset、checkpoint、results 位置。
- 新增 `DOC/github_packaging.md`，整理 GitHub repo 應包含與不應包含的內容，以及大型 checkpoint 的分享方式。

### 實際執行過的 command

```bash
du -h -d 1 .
find . -maxdepth 3 -type f -size +50M -printf '%s %p\n' | sort -nr | head -80
du -h -d 2 checkpoints results data logs DOC framework scripts | sort -h | tail -80
find results -maxdepth 2 -type f -name '*grid.png' -o -name '*compare*.png' | sort
```

### 重要結果摘要

- Project total：約 175 GB。
- `data/`：約 157 GB，主要是 Hugging Face PIPE dataset/model cache，不推 GitHub。
- `checkpoints/`：約 18 GB，Strategy C checkpoints 約 1.4 GB each，不推 GitHub。
- `results/`：約 330 MB，只建議保留 selected comparison PNGs。
- GitHub repo 建議包含 code、scripts、DOC、metrics CSV、selected comparison figures。
