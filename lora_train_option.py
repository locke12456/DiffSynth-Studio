#!/usr/bin/env python3
"""
DiffSynth Anima LoRA training wrapper.

Usage:
  python lora_train.py                          # all defaults
  python lora_train.py --num-epochs 1           # override one
  python lora_train.py --lora-checkpoint ./models/train/anima_oppai_loli_lora/epoch-0.safetensors
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

PY = r"D:\AI\python_embeded\python.exe"

DEFAULT_DATASET    = r"D:\AI\dataset"
DEFAULT_OUTPUT     = r"./models/train/opus-girl"
DEFAULT_BASE_MODEL = "circlestone-labs/Anima:split_files/diffusion_models/anima-base-v1.0.safetensors"

# These two are not promoted to flags because they almost never change for Anima.
TEXT_ENC = "circlestone-labs/Anima:split_files/text_encoders/qwen_3_06b_base.safetensors"
VAE      = "circlestone-labs/Anima:split_files/vae/qwen_image_vae.safetensors"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


from PIL import Image, ImageOps

DEFAULT_MAX_PIXELS = 1024 * 1024   # 1048576，Anima 的合理訓練解析度

def auto_max_pixels(dataset_dir: str, cap: int = DEFAULT_MAX_PIXELS) -> int:
    """Report dataset area stats and return an area budget for DiffSynth.

    max_pixels is an UPPER BOUND on h*w after aspect-preserving resize;
    DiffSynth then snaps h and w to multiples of 16 itself. Aligning the
    *area* is meaningless -- only the cap value matters.
    """
    root = Path(dataset_dir)
    areas = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        try:
            with Image.open(p) as im:
                w, h = ImageOps.exif_transpose(im).size
        except Exception as e:
            print(f"[auto_max_pixels] skip {p.name}: {e}")
            continue
        areas.append(w * h)

    if not areas:
        raise RuntimeError(f"No images found under {root}")

    areas.sort()
    mn, md, mx = areas[0], areas[len(areas) // 2], areas[-1]
    print(f"[auto_max_pixels] {len(areas)} images | "
          f"min={mn} median={md} max={mx} | using cap={cap}")
    if mx > cap:
        over = sum(1 for a in areas if a > cap)
        print(f"[auto_max_pixels] {over} image(s) exceed the cap "
              f"and will be downscaled by DiffSynth")
    return cap

def parse_args():
    p = argparse.ArgumentParser(description="Anima LoRA training wrapper")
    p.add_argument("--dataset-base-path", default=DEFAULT_DATASET,
                   help=f"Default: {DEFAULT_DATASET}")
    p.add_argument("--dataset-repeat", type=int, default=10,
                   help="Default: 10")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="If omitted, scan dataset and take max(w*h) aligned to 256.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help=f"Format 'repo:path/file.safetensors'. Default: {DEFAULT_BASE_MODEL}")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8,
                   help="Default: 8")
    p.add_argument("--num-epochs", type=int, default=5,
                   help="Default: 5")
    p.add_argument("--output-path", default=DEFAULT_OUTPUT,
                   help=f"Default: {DEFAULT_OUTPUT}")
    p.add_argument("--save-steps", type=int, default=680,
                   help="Default: 500")
    p.add_argument("--lora-checkpoint", default=None,
                   help="Path to an existing LoRA .safetensors for warm-restart. "
                        "Note: warm-restart only loads weights; optimizer/scheduler/epoch "
                        "counter are reset on current DiffSynth main.")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    max_pixels = args.max_pixels or auto_max_pixels(args.dataset_base_path)
    metadata_csv = str(Path(args.dataset_base_path) / "metadata.csv")
    model_id_paths = ",".join([args.base_model, TEXT_ENC, VAE])

    cmd = [
        PY, "-m", "accelerate.commands.launch",
        "--mixed_precision", "bf16",
        "examples/anima/model_training/train.py",
        "--dataset_base_path",     args.dataset_base_path,
        "--dataset_metadata_path", metadata_csv,
        "--max_pixels",            str(max_pixels),
        "--dataset_repeat",        str(args.dataset_repeat),
        "--dataset_num_workers",   "0",                          # Windows: must be 0
        "--data_file_keys",        "image",
        "--model_id_with_origin_paths", model_id_paths,
        "--tokenizer_path",        "Qwen/Qwen3-0.6B:./",
        "--tokenizer_t5xxl_path",  "stabilityai/stable-diffusion-3.5-large:tokenizer_3/",
        "--learning_rate",         "2e-5",
        "--weight_decay",          "0.01",
        "--num_epochs",            str(args.num_epochs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--save_steps",            str(args.save_steps),
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--lora_base_model",       "dit",
        "--lora_target_modules",   "",
        "--lora_rank",             "32",
        "--use_gradient_checkpointing",
        "--output_path",           args.output_path,
    ]

    if args.lora_checkpoint:
        cmd.extend(["--lora_checkpoint", args.lora_checkpoint])

    print(">>>", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
