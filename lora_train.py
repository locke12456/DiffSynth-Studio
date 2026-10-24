"""
Anima preview3-base LoRA training (Windows / cu130 / RTX 5090).

Dataset: D:\AI\dataset (225 張, 1024x1365)
Trigger: oppai_loli (放在 caption 的第一個 tag)
Base model: circlestone-labs/Anima preview3-base

Run:
    cd /d D:\AI\DiffSynth-Studio
    D:\AI\python_embeded\python.exe train_anima_lora.py

Resume (warm-restart, optimizer/scheduler/epoch 會 reset):
    改下面 LORA_CHECKPOINT 變數為要接續的 .safetensors 路徑
"""
import os
import subprocess
import sys

# === Env (必須在 import diffsynth / subprocess 之前 set) ===
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Paths ===
PY = r"D:\AI\python_embeded\python.exe"
DATASET_DIR = r"D:\AI\dataset"
METADATA_CSV = r"D:\AI\dataset\metadata.csv"
OUTPUT_DIR = r"./models/train/anima_oppai_loli_base_lora"

# === Resume (第一次訓黄不用動，設 None) ===
# 接續訓練時填入 epoch checkpoint 路徑，例如：
# LORA_CHECKPOINT = r"./models/train/anima_oppai_loli_lora/epoch-4.safetensors"
LORA_CHECKPOINT = None

# === Training config ===
cmd = [
    PY, "-m", "accelerate.commands.launch",
    "examples/anima/model_training/train.py",
    "--mixed_precision", "bf16",
    # --- Dataset ---
    "--dataset_base_path", DATASET_DIR,
    "--dataset_metadata_path", METADATA_CSV,
    "--max_pixels", "1397760",       # 1024*1365 → round 到 1024x1360 (保留原比例)
    # 不設 --height / --width → dynamic resolution
    "--dataset_repeat", "10",        # 225*10 = 2250 step/epoch
    "--dataset_num_workers", "0",    # Windows 上太高反而拖慢
    "--data_file_keys", "image",     # default "image,video"，純圖集只要 image

    # --- Base model (preview3-base, 不是預設 preview1) ---
    "--model_id_with_origin_paths",
    #"circlestone-labs/Anima:split_files/diffusion_models/anima-preview3-base.safetensors,"
    "circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors,"
    "circlestone-labs/Anima:split_files/text_encoders/qwen_3_06b_base.safetensors,"
    "circlestone-labs/Anima:split_files/vae/qwen_image_vae.safetensors",

    # --- Tokenizers ---
    "--tokenizer_path", "Qwen/Qwen3-0.6B:./",
    "--tokenizer_t5xxl_path", "stabilityai/stable-diffusion-3.5-large:tokenizer_3/",

    # --- Training ---
    "--learning_rate", "2e-5",       # HF model card 推薦 (官方 sh 是 1e-4)
    "--weight_decay", "0.01",        # default
    "--num_epochs", "5",
    "--gradient_accumulation_steps", "8",  # effective batch = 1*4 = 4
    "--remove_prefix_in_ckpt", "pipe.dit.",

    # --- LoRA ---
    "--lora_base_model", "dit",
    "--lora_target_modules", "",     # 空 = parser default "q,k,v,o,ffn.0,ffn.2"
    "--lora_rank", "32",

    # --- VRAM ---
    "--use_gradient_checkpointing",
    # 爆 VRAM 再加： "--use_gradient_chzeckpointing_offload",

    # --- Output ---
    "--output_path", OUTPUT_DIR,
    # 不設 --save_steps → 每 epoch 存一個 (5 個存檔)
    "--save_steps", "500",
]

# Resume (warm-restart)
if LORA_CHECKPOINT is not None:
    cmd += ["--lora_checkpoint", LORA_CHECKPOINT]

print(">>", " ".join(f'"{c}"' if " " in c else c for c in cmd))
sys.exit(subprocess.call(cmd))
