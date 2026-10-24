# lora_train_option.py 修正報告 (Anima concept / style LoRA)

Dataset: `D:\AI\dataset` — 77 張圖 (皆 1536x1536 等面積 2.36MP)，metadata.csv 77 行，無缺圖、無空 prompt。
觸發詞：`saika_magoichi`, `locke_chibi`，以及角度標籤 `chibi_0d/45d/90d/135d`。

## 修正項目
| # | 問題 | 修正 |
|---|------|------|
| 1 | learning rate 寫死 `2e-5`。官方 DiffSynth Anima LoRA 範例 (`examples/anima/model_training/lora/anima-preview.sh`) 用 `1e-4`；2e-5 是先前 226 張圖、1 epoch 的設定，77 張圖用它學習量偏少 | 新增 `--learning-rate`，預設 `1e-4` |
| 2 | `--save-steps` 預設 680，但 77×10 repeat ÷ grad-accum 8 ≈ 96 step/epoch，5 epoch ≈ 480 step → 永遠不會觸發，help 還寫 500 | 預設改為 None = 每 epoch 存一次；有給才傳 `--save_steps` |
| 3 | lora rank 寫死 32 | 新增 `--lora-rank` (預設 32) |
| 4 | `from PIL import ...` 重複匯入、docstring 用錯檔名 `lora_train.py`、`--max-pixels` help 與實作不符 | 清理 |
| 5 | Python 路徑寫死，不存在時直接失敗 | 找不到時退回 `sys.executable` |
| 6 | 沒有資料檢查 | 新增 `check_metadata()`：訓練前檢查 metadata.csv 列出的圖是否存在、prompt 是否為空 |

## 建議用法
- **Concept / 角色 (目前資料集)**：預設即可 `python lora_train_option.py`，lr 1e-4、rank 32、5 epoch；每 epoch 存一個 checkpoint，用 `test_anima.py` 比較，挑還沒過擬合的那個。
- **Style**：`--learning-rate 5e-5 --lora-rank 16`，caption 不要描述風格本身 (flat color 等) 讓 LoRA 吸收風格；過擬合就降 epoch。
- 若過擬合很快 (先前經驗：226 張、>1 epoch 就過擬合)，降到 `--learning-rate 2e-5` 或 `--num-epochs 2`。
- 圖片 2.36MP 會被縮到 `max_pixels=1048576` (1024²)，這是 Anima 的訓練解析度，正常。

## 參考來源
- DiffSynth-Studio Anima 文件: https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Anima.html
- 先前 style LoRA 訓練紀錄: https://huggingface.co/locke12/oppai_loli
- Anima discussions: https://huggingface.co/circlestone-labs/Anima/discussions/157
- LoRA 超參數一般指南: https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide
