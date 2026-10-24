# MOSS-Image 畫風 LoRA 預設值調整報告

日期：2026-10-24
對象：`moss_image_lora_train.py`（DiffSynth-Studio + MOSS-Image-Base DiT + EntroPack 6-bit TE）

## 1. 原本的預設是不是畫風 LoRA？

不是。原預設幾乎照抄 DiffSynth 官方範例 `examples/moss_image/model_training/lora/MOSS-Image-Base.sh`。那是通用的 LoRA 範例，沒有針對畫風調整：

| 參數 | 官方範例 | 原 script |
|---|---|---|
| lora_rank | 32 | 32 |
| learning_rate | 1e-4 | 1e-4 |
| num_epochs | 5 | 5 |
| dataset_repeat | 50 | 10 |
| gradient_accumulation | 1 | 8 |
| lora_target_modules | `""`（自動，所有 Linear） | `""` |

`""` 會讓 DiffSynth 對 DiT 的所有 Linear 掛 LoRA，連 `text_embedder`（把 TE 輸出轉進 DiT 的 2 層 transformer）也包含在內。對畫風 LoRA 來說，這會讓 LoRA 動到「怎麼理解 prompt」，而不只是「怎麼畫」。

## 2. 查到的資料

| 來源 | 內容 |
|---|---|
| KohakuMossTrainer `configs/moss/_base/base_lora.py`（目前唯一專門針對 MOSS-Image 的 LoRA trainer） | rank 32、alpha 32、lr 1e-4、AdamW、weight decay 0.01、warmup 100 + cosine（min 0.1）、4000 steps、preset `attn-mlp`、text dropout 0.1、grad clip 1.0、timestep 取樣用 `moss`（shift 1.0） |
| KohakuMossTrainer README | 畫風 LoRA 範例用 **rank 8**、Base 模型、100 steps、batch 約 4 |
| KohakuMossTrainer `recipes_lora_5090.py` | rank 32、bf16、約 1MP、每個 optimizer step 累積 64 個 micro-batch（大資料集用） |
| Flux 畫風 LoRA 社群指南 | rank 16–32（紋理多的畫風用 32，rank 越高越容易過擬合）；lr 1e-4，畫面「燒掉」就減半；約 1000–2000 steps 或每張圖 50–100 steps；**定期存 checkpoint，最好的通常不是最後一個** |
| FLUX.2 指南 | rank 32–128、lr 1e-4 到 4e-4；訓練崩掉時降成 lr 5e-5、rank 16 |

注意：MOSS-Image 很新，除了 KohakuMoss 之外，沒有經過驗證的畫風 LoRA 參數。上表中 Flux 的數字是同類 flow-matching DiT 的經驗值，不是 MOSS 的實測結果。

## 3. 新預設值

| 參數 | 舊 | 新 | 理由 |
|---|---|---|---|
| LoRA 範圍 | `""` 自動，含 text_embedder | **42 層 × `q,k,v,o,ffn_up_gate,ffn_down`，共 252 個模組，不含 text_embedder** | 對應 KohakuMoss 的 `attn-mlp` preset；畫風只學「怎麼畫」，不改 prompt 理解 |
| lora_rank | 32 | 32（不變） | KohakuMoss 的預設；也落在 Flux 畫風建議的 16–32 範圍內 |
| learning_rate | 1e-4 | 1e-4（不變） | KohakuMoss 和 Flux 指南都用這個值 |
| weight_decay | 0.01 | 0.01（不變） | KohakuMoss |
| num_epochs | 5 | **10** | 每個 epoch 存一次，可以挑出最好的那個 checkpoint |
| gradient_accumulation | 8 | **4** | 小資料集的畫風訓練需要足夠的 optimizer steps；4 接近 KohakuMoss 畫風範例的 batch 約 4 |
| dataset_repeat | 10 | 10（不變） | |
| max_pixels | 1MP | 1MP（不變） | KohakuMoss 也用約 1MP |
| 模型 | — | **Base**（不變） | KohakuMoss 的畫風範例用 Base；SFT 已經有自己的畫風偏好，比較難蓋過去 |

步數估算：optimizer steps = 圖片數 × repeat × epochs ÷ accumulation。
例如 30 張圖：30 × 10 × 10 ÷ 4 = 750 steps，每個 epoch 75 steps。這落在 Flux 指南「每張圖 50–100 steps」的範圍附近（30 張 × 25 個樣本）。資料少於約 20 張時，建議把 `--dataset-repeat` 調高到 20。

### 3.1 依資料量調整 `--dataset-repeat`（實測修正）

repeat 10 的預設是以約 30 張圖估算的，圖多時會訓練過頭。實際跑 226 張圖的結果（RTX 5090、`--stage train`、1MP）：

- 進度條 `24/2260 [00:43<1:07:37, 1.81s/it]`，每個 epoch 是 226 張 × repeat 10 = 2260 個樣本
- 1 個 epoch 約 68 分鐘，10 個 epoch 超過 11 小時，每張圖會被看 100 次，以畫風來說太多

依每張圖總共被看的次數（= repeat × epochs）來選 repeat。畫風建議每張圖看 10–20 次，總共約 500–1000 個 optimizer steps：

| 圖片數 | 建議 repeat（10 epochs） | 每張圖看幾次 | optimizer steps（accum 4） |
|---|---|---|---|
| ≤ 20 | 10–20 | 100–200 | 少於 1000 |
| 約 30–50 | 5 | 50 | 375–625 |
| 約 100 | 2 | 20 | 500 |
| 200 以上 | **1** | 10 | 226 張：565 |

226 張這一輪改用 `--dataset-repeat 1`：每個 epoch 約 7 分鐘，10 個 epoch 約 70 分鐘。畫風太弱的話，改用 `--dataset-repeat 2`（約 2.3 小時），或用 `--lora-checkpoint` 從最好的 checkpoint 接著訓練。
`--dataset-repeat` 只影響訓練階段，改了不用重跑 cache。

待辦：把 repeat 預設改成依圖片數量自動計算（目前 script 的預設仍是 10，資料多時要手動調）。

新增旗標 `--lora-text-embedder`：恢復舊的自動行為，連 text_embedder 一起訓練，適合用在要學新概念或新詞彙、而不是畫風的情況。

## 4. DiffSynth 做不到的部分（跟 KohakuMoss 的差異）

| KohakuMoss | DiffSynth train.py | 影響 |
|---|---|---|
| warmup 100 + cosine 衰減 | `ConstantLR`（只有前 5 步是 1/3 lr） | 後期沒有降 lr，所以更需要靠各 epoch 的 checkpoint 來挑 |
| text dropout 0.1 | 沒有 | LoRA 比較容易把畫風綁在 caption 用詞上；可以在 metadata 裡放一部分空 caption 或很短的 caption 來模擬 |
| `moss` timestep 取樣 | DiffSynth 自己的 FlowMatch scheduler | 無法調整 |
| grad clip 1.0 | 沒有參數可以設定 | — |
| LoRA+、LyCORIS（LoKr、LoHa） | 只有標準 LoRA | — |

## 5. 調整建議

- 畫面燒掉、顏色過飽和，或很早就過擬合：把 `--learning-rate` 降到 `5e-5`，或把 `--lora-rank` 降到 16。
- 畫風太弱：先多訓練幾個 epoch，再考慮把 rank 調到 64。
- 檔案想小一點：`--lora-rank 8`（KohakuMoss 的畫風範例用的就是 rank 8）。
- 只想影響構圖或粗略的畫風：試 `--lora-max-layer`（只訓練前段的層）。這是實驗性的用法，沒有找到 MOSS 的相關資料。

## 6. 修改的檔案

- `moss_image_lora_train.py`：上述預設值；新增 `--lora-text-embedder`；LoRA 範圍預設改成明確的 252 個模組
- `MOSS-Readme.md`：更新參數表

驗證：`--stage train --dry-run` 印出 `[lora] layers 0-41 -> 252 modules (6 per layer)`，預檢通過。實際訓練的成果還沒驗證。

## 參考來源

- KohakuMossTrainer：https://github.com/KohakuBlueleaf/KohakuMossTrainer
- KohakuMossTrainer `configs/moss/_base/base_lora.py`、`configs/moss/recipes/recipes_lora_5090.py`
- KohakuMoss 介紹（ComfyUI Wiki）：https://comfyui-wiki.com/en/news/2026-10-19-kohakumoss-moss-image-finetuning
- Flux LoRA dataset/captions/settings：https://www.promptzone.com/santiago_saleh/training-a-flux-lora-dataset-captions-and-settings-59m1.md
- FLUX 2 Dev LoRA：https://www.mindstudio.ai/blog/what-is-flux-2-dev-lora
- FLUX.2 Klein 訓練指南：https://www.runcomfy.com/trainer/ai-toolkit/ai-toolkit-flux-klein-9b-high-likeness-character-training
- DiffSynth 官方範例：`examples/moss_image/model_training/lora/MOSS-Image-Base.sh`
