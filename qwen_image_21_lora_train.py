#!/usr/bin/env python3
r"""
DiffSynth-Studio  Qwen-Image-2.1  LoRA training wrapper.

Usage:
  python qwen_image_21_lora_train.py --check              # verify every model file loads
  python qwen_image_21_lora_train.py --stage cache        # 1) bake text/VAE embeddings to .pth
  python qwen_image_21_lora_train.py --stage train        # 2) train from cache (no text encoder)
  python qwen_image_21_lora_train.py                      # single-pass training
  python qwen_image_21_lora_train.py --edit               # image-editing LoRA (metadata.json)

  python qwen_image_21_lora_train.py --stage train --lora-max-layer 17
      Inject LoRA only into transformer_blocks.0..17 and leave the last layers
      untouched. Qwen-Image-2.1's DiT is 32 single-stream layers; LoRA on the
      tail layers wrecks fine structure (hands, fingers) and forces you to run
      the adapter below strength 1.0.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Environment  -- edit these four
# --------------------------------------------------------------------------- #
PY         = r"D:\AI\python_embeded\python.exe"
DIFFSYNTH  = r"D:\AI\DiffSynth-Studio"                  # the checkout that is pip-installed
OFFICIAL   = r"D:\AI\Qwen Image\2.1\official"           # huggingface-cli --local-dir
DATASET    = r"D:\AI\dataset"

TRAIN_PY   = os.path.join(DIFFSYNTH, "examples", "qwen_image_21", "model_training", "train.py")
OUTPUT     = r"./models/train/qwen21_oppai"
CACHE      = r"./data/cache/qwen21"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MAX_PIXELS = 1024 * 1024   # 1048576; drop to 786432 (~896x896) if VRAM is tight


# --------------------------------------------------------------------------- #
# Official diffusers layout. A multi-file model is ONE nested list, because
# hash_model_file() merges every shard's keys before hashing.
# --------------------------------------------------------------------------- #
def shards(subdir, pattern):
    files = sorted(glob.glob(os.path.join(OFFICIAL, subdir, pattern)))
    if not files:
        return None
    return files[0] if len(files) == 1 else files


DIT       = shards("transformer",  "diffusion_pytorch_model*.safetensors")
TE        = shards("text_encoder", "model*.safetensors")
VAE       = shards("vae",          "diffusion_pytorch_model*.safetensors")
PROCESSOR = os.path.join(OFFICIAL, "processor")


def require_models(names):
    missing = [n for n, v in names if v is None]
    if missing:
        sys.exit(
            f"Missing model files for: {', '.join(missing)}\n"
            f"Looked under {OFFICIAL}. Download with:\n"
            f'  huggingface-cli download Qwen/Qwen-Image-2.1 '
            f'--include "transformer/*" "text_encoder/*" "vae/*" "processor/*" '
            f'--local-dir "{OFFICIAL}"'
        )


def check_models():
    """Verify DiffSynth recognises each file before burning time on a real run."""
    code = (
        "import json,sys;"
        "from diffsynth.core.loader import hash_model_file;"
        "from diffsynth.configs import MODEL_CONFIGS;"
        "reg={c['model_hash']:c['model_name'] for c in MODEL_CONFIGS};"
        "\nfor label,path in json.loads(sys.argv[1]).items():\n"
        "    h=hash_model_file(path)\n"
        "    print(f'{label:<14} {h}  ->  ' + reg.get(h,'*** NOT REGISTERED ***'))\n"
    )
    targets = {k: v for k, v in (("transformer", DIT), ("text_encoder", TE), ("vae", VAE)) if v}
    sys.exit(subprocess.call([PY, "-c", code, json.dumps(targets)], cwd=DIFFSYNTH))


def auto_max_pixels(dataset_dir, cap=DEFAULT_MAX_PIXELS):
    root, areas = Path(dataset_dir), []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            try:
                with Image.open(p) as im:
                    w, h = ImageOps.exif_transpose(im).size
                areas.append(w * h)
            except Exception as e:
                print(f"[auto_max_pixels] skip {p.name}: {e}")
    if not areas:
        raise RuntimeError(f"No images found under {root}")
    areas.sort()
    print(f"[auto_max_pixels] {len(areas)} images | min={areas[0]} "
          f"median={areas[len(areas)//2]} max={areas[-1]} | cap={cap}")
    return cap


def safetensors_shapes(path):
    """Read tensor name -> shape from the header only. No safetensors package needed."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


def layer_limited_targets(max_layer, suffixes=""):
    """
    Build an explicit LoRA target list covering transformer_blocks.0..max_layer.

    Mirrors DiffSynth's auto-search rule (2-D Linear with min(in, out) >= 512) but
    stops at max_layer. `suffixes` optionally narrows it further, e.g.
    "to_q,to_k,to_v,to_out.0" for attention only.
    """
    shapes = {}
    for f in (DIT if isinstance(DIT, list) else [DIT]):
        shapes.update(safetensors_shapes(f))

    want = [s.strip() for s in suffixes.split(",") if s.strip()]
    modules = []
    for key, shape in shapes.items():
        if not key.endswith(".weight") or len(shape) != 2 or min(shape) < 512:
            continue
        name = key[: -len(".weight")]
        m = BLOCK_RE.match(name)
        if not m or int(m.group(1)) > max_layer:
            continue
        if want and not any(name == w or name.endswith("." + w) for w in want):
            continue
        modules.append(name)

    modules.sort(key=lambda n: (int(BLOCK_RE.match(n).group(1)), n))
    return modules


def parse_args():
    p = argparse.ArgumentParser(description="Qwen-Image-2.1 LoRA training wrapper")
    p.add_argument("--check", action="store_true", help="Only verify the model files, then exit.")
    p.add_argument("--stage", choices=["all", "cache", "train"], default="all")
    p.add_argument("--cache-path", default=CACHE)
    p.add_argument("--edit", action="store_true")
    # dataset
    p.add_argument("--dataset-base-path", default=DATASET)
    p.add_argument("--metadata", default=None)
    p.add_argument("--dataset-repeat", type=int, default=10)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0, help="Windows: keep 0")
    # training
    p.add_argument("--learning-rate", default="1e-4")
    p.add_argument("--weight-decay", default="0.01")
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-target-modules", default="")
    p.add_argument("--lora-max-layer", type=int, default=None,
                   help="Inject LoRA only into transformer_blocks.0..N (DiT has 32 layers). "
                        "Combine with --lora-target-modules to also narrow by module name.")
    # output / resume
    p.add_argument("--output-path", default=OUTPUT)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--lora-checkpoint", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    # memory
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--nf4", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.check:
        require_models([("transformer", DIT), ("text_encoder", TE), ("vae", VAE)])
        check_models()

    caching, from_cache = args.stage == "cache", args.stage == "train"

    # Load only what each stage actually needs. On a 32 GB card the DiT (14.2 GB)
    # and the text encoder (17.5 GB) cannot be resident at the same time.
    if from_cache:                      # embeddings are already baked
        needed = [("transformer", DIT)]
        models = [DIT]
    elif caching:                       # encoding prompts + images only
        needed = [("text_encoder", TE), ("vae", VAE)]
        models = [TE, VAE]
    else:                               # single pass: needs everything at once
        needed = [("transformer", DIT), ("text_encoder", TE), ("vae", VAE)]
        models = [DIT, TE, VAE]
    require_models(needed)

    cmd = [
        PY, "-m", "accelerate.commands.launch",
        "--mixed_precision", "bf16",
        TRAIN_PY,
        "--model_paths", json.dumps(models),
        "--processor_path", PROCESSOR,
        "--dataset_num_workers", str(args.num_workers),
        "--find_unused_parameters",
        "--output_path", args.cache_path if caching else args.output_path,
    ]

    # ---------- dataset ----------
    if from_cache:
        # no metadata_path -> UnifiedDataset scans base_path for *.pth
        cmd += ["--dataset_base_path", args.cache_path]
    else:
        max_pixels = args.max_pixels or auto_max_pixels(args.dataset_base_path)
        meta = args.metadata or str(Path(args.dataset_base_path) /
                                    ("metadata.json" if args.edit else "metadata.csv"))
        cmd += [
            "--dataset_base_path",     args.dataset_base_path,
            "--dataset_metadata_path", meta,
            "--max_pixels",            str(max_pixels),
            "--data_file_keys",        "image,edit_image" if args.edit else "image",
        ]
        if args.edit:
            cmd += ["--extra_inputs", "edit_image"]

    # ---------- task ----------
    if caching:
        # IMPORTANT: gradient-checkpointing flags are baked into the cached .pth,
        # because sft:data_process stores the whole inputs tuple and sft:train
        # replays it verbatim. Passing them only at train time has no effect.
        cmd += ["--task", "sft:data_process", "--use_gradient_checkpointing"]
        if args.low_vram:
            cmd += ["--use_gradient_checkpointing_offload"]
    else:
        # ---------- LoRA injection scope ----------
        targets = args.lora_target_modules
        if args.lora_max_layer is not None:
            modules = layer_limited_targets(args.lora_max_layer, args.lora_target_modules)
            if not modules:
                sys.exit(f"--lora-max-layer {args.lora_max_layer} matched no modules. "
                         f"Check --lora-target-modules.")
            per_layer = len(modules) / (args.lora_max_layer + 1)
            print(f"[lora-max-layer] blocks 0-{args.lora_max_layer} -> {len(modules)} modules "
                  f"({per_layer:.0f} per layer), e.g. {modules[0]}")
            targets = ",".join(modules)

        cmd += [
            "--task", "sft:train" if from_cache else "sft",
            "--dataset_repeat",        str(args.dataset_repeat),
            "--learning_rate",         args.learning_rate,
            "--weight_decay",          args.weight_decay,
            "--num_epochs",            str(args.num_epochs),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--use_gradient_checkpointing",
            "--lora_base_model",       "dit",
            "--lora_target_modules",   targets,
            "--lora_rank",             str(args.lora_rank),
            "--remove_prefix_in_ckpt", "pipe.dit.",
        ]
        for flag, val in (("--save_steps", args.save_steps),
                          ("--lora_checkpoint", args.lora_checkpoint),
                          ("--resume_from_checkpoint", args.resume_from_checkpoint)):
            if val:
                cmd += [flag, str(val)]

    # ---------- VRAM savers ----------
    if args.low_vram:
        cmd += ["--use_gradient_checkpointing_offload", "--initialize_model_on_cpu"]
        # --fp8_models matches by exact string, so it only works for single-file entries.
        fp8 = [m for m in (TE, VAE) if isinstance(m, str) and m in models]
        if fp8:
            cmd += ["--fp8_models", ",".join(fp8)]
        multi = [n for n, m in (("text_encoder", TE), ("vae", VAE))
                 if isinstance(m, list) and m in models]
        if multi:
            print(f"[warn] fp8 cannot be applied to multi-shard models: {', '.join(multi)}. "
                  f"Use --stage cache / --stage train instead.")
    if args.nf4:
        cmd += ["--quant_options", f"{json.dumps(DIT) if isinstance(DIT, list) else DIT}: bitsandbytes_nf4"]

    print(">>>", " ".join(cmd))
    # run from the DiffSynth checkout so `import diffsynth` and the example agree
    sys.exit(subprocess.call(cmd, cwd=DIFFSYNTH))


if __name__ == "__main__":
    main()
