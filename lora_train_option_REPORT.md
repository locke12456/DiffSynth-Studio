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

## 第二輪：style / concept preset
新增 `--preset {concept,style}`，自動設定 lr / rank / epochs；明確給的參數 (如 `--learning-rate`) 會覆蓋 preset。
啟動時印出 `[plan]` 預估總 step 數，超出 ~300–1500 會警告。

| preset | lr | rank | epochs | 77 張 × repeat 10 / accum 8 的總 step |
|--------|----|------|--------|-----|
| concept (角色/物件，預設) | 1e-4 | 32 | 5 | ~480 |
| style (畫風) | 5e-5 | 16 | 8 | ~768 |

依據 (網路上的 Anima 社群指南，數值不一致，取中間值)：
- sd-scripts Anima 範例：rank 8–16、lr 1e-4、約 10 epoch。
- artificialguy.com：rank 32 建議從 lr 2e-5 開始調整；總 step 盡量 < 1000。
- **Concept**：要學特定身分，用較高 lr 搭配較少 step；caption 用觸發詞，並詳細描述會變動的服裝與背景，讓角色特徵集中在觸發詞上。
- **Style**：用較低 lr、較多 step 泛化較好，rank 16 就夠；caption 照常描述內容，但**不要**寫風格相關詞 (flat color、simple coloring 等)，畫風才會被 LoRA 吸收。
- 目前資料集 caption 含 `flat color, simple coloring`。若要訓練成 style LoRA，建議把這些詞拿掉；若是 concept LoRA，則保留不影響。

## 建議用法
- **Concept / 角色 (目前資料集)**：`python lora_train_option.py` (預設 `--preset concept`)；每 epoch 存一個 checkpoint，用 `test_anima.py` 比較，挑還沒過擬合的那個。
- **Style**：`python lora_train_option.py --preset style --output-path ./models/train/<name>_style`。
- 若過擬合很快 (先前經驗：226 張、>1 epoch 就過擬合)，降到 `--learning-rate 2e-5` 或 `--num-epochs 2`。
- 圖片 2.36MP 會被縮到 `max_pixels=1048576` (1024²)，這是 Anima 的訓練解析度，正常。

## 參考來源
- sd-scripts Anima 文件: https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md
- sd-scripts Anima 討論: https://huggingface.co/circlestone-labs/Anima/discussions/35
- Anima LoRA Training Settings: https://artificialguy.com/blog/anima-lora-training-settings/
- Civitai sd-scripts 設定: https://civitai.com/articles/31972/anima-lora-training-settings-for-sd-scripts
- Civitai 新手指南: https://civitai.com/articles/31678/a-complete-beginners-guide-to-local-anima-lora-training
- 12GB 角色 LoRA 指南: https://note.com/sepiablue/n/nf6cb2520d2a2
- DiffSynth-Studio Anima 文件: https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Anima.html
- 先前 style LoRA 訓練紀錄: https://huggingface.co/locke12/oppai_loli
- Anima discussions: https://huggingface.co/circlestone-labs/Anima/discussions/157
- LoRA 超參數一般指南: https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide
