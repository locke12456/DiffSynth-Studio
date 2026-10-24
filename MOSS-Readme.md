**0. 環境需求**

DiffSynth 官方只寫 `pip install -e .`，但 MOSS-Image + 6-bit TE 實際還需要下面這些（來源：`pyproject.toml` 的 `[quant]` extra、`docs/zh/Pipeline_Usage/Quantization.md:87`、`diffsynth/models/moss_image_text_encoder.py`）：

| 套件 | 版本 | 原因 |
|---|---|---|
| Python | 3.10 以上（目前 3.12） | DiffSynth / entropack |
| torch + torchvision | **CUDA 版**，torch **≥ 2.10**，兩者要同一個 cu 版本 | entropack 需要 torch ≥ 2.10 |
| transformers | **≥ 5** | MOSS TE 用 `Qwen3_5TextConfig`，4.x 沒有 |
| entropack[cuda13] | 會一併裝 cupy-cuda13x | 載入 `*_6bit.safetensors` |
| accelerate、peft | | train.py |
| bitsandbytes | 只有 `--nf4` 才需要 | |
| causal-conv1d、flash-linear-attention | 選用 | 只加速 cache 階段的 TE；沒裝會改用較慢的 PyTorch 實作，結果相同 |

venv 安裝（**torch 必須單獨先裝**，跟其他套件一起裝的話 uv 會從 PyPI 抓 CPU 版）：
```
uv venv D:\AI\venv-diffsynth --python 3.12
uv pip install --python D:\AI\venv-diffsynth\Scripts\python.exe "torch>=2.10" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python D:\AI\venv-diffsynth\Scripts\python.exe -e D:\AI\DiffSynth-Studio "transformers>=5" "entropack[cuda13]"
```
script 裡的 `PY` 要指向 `D:\AI\venv-diffsynth\Scripts\python.exe`。不要用 ComfyUI 的 `python_embeded`，它的 transformers 是 4.x，而且沒有 venv 模組。

`--check` 或 `--dry-run` 會在 `PY` 裡實際 import 上面這些套件，缺什麼就印出安裝指令。

**1. 準備新資料集**
```
D:\AI\dataset\
  metadata.csv
  group1\xxx.png
  ...
```
`metadata.csv` 的格式：
```
image,prompt
group1/xxx.png,"caption 文字"
```
`image` 欄填的是相對於 `D:\AI\dataset` 的路徑。

**2. 下載並確認模型**
```
python moss_image_lora_train.py --download            # 只抓缺少的：Base DiT / 6-bit TE / VAE / tokenizer
python moss_image_lora_train.py --check               # 模型 hash + 套件環境
python moss_image_lora_train.py --stage cache --dry-run   # 全部預檢，只印出指令不執行
```

**3. 兩階段訓練（推薦，比較省顯存）**
```
python moss_image_lora_train.py --stage cache
python moss_image_lora_train.py --stage train
```
- **cache 階段**：載入 6-bit TE 和 VAE，把 prompt 和圖片預先編碼，存到 `.\data\cache\moss_image`。
- **train 階段**：只載入 DiT 訓練 LoRA，輸出到 `.\models\train\moss_image_lora`。
- 改了 caption 或圖片之後，要重跑 cache。

也可以一次跑完：`python moss_image_lora_train.py`，但這樣三個模型會同時放在顯存裡。

**常用參數**（預設是畫風 LoRA 設定，依據見 `MOSS-LoRA-Report.md`）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dataset-base-path` | `D:\AI\dataset` | 資料集位置 |
| `--num-epochs` | 10 | 訓練輪數（每輪存一個 checkpoint） |
| `--dataset-repeat` | 10 | 每個 epoch 重複幾次資料 |
| `--learning-rate` | 1e-4 | 學習率 |
| `--lora-rank` | 32 | LoRA rank |
| `--gradient-accumulation-steps` | 4 | 梯度累積步數 |
| `--max-pixels` | 1048576 | 最大像素數。顯存不夠可以改成 786432 |
| `--save-steps` | 無 | 每 N 步存一次 checkpoint |
| `--lora-max-layer N` | 41 | 只訓練第 0 到 N 層（總共 42 層） |
| `--lora-text-embedder` | 關 | 連 text_embedder 一起訓練（學新概念用，畫風不要開） |
| `--low-vram` | 關 | 把模型 offload 到 CPU |
| `--nf4` | 關 | 把 DiT 量化成 nf4，省更多顯存 |

**接著訓練（續訓）**
```
python moss_image_lora_train.py --stage train --resume-from-checkpoint <checkpoint 路徑>
```

**注意事項：**
- 在 Windows 上 `--num-workers` 請維持 0。
- 如果 cache 階段有加 `--low-vram`，train 階段也要加。gradient checkpointing 的設定會跟著存進 cache 檔，兩個階段要一致。
