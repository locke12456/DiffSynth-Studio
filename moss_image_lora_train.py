#!/usr/bin/env python3
r"""
DiffSynth-Studio  MOSS-Image  LoRA training wrapper.

Usage:
  python moss_image_lora_train.py --check                 # verify every model file loads
  python moss_image_lora_train.py --stage cache           # 1) bake text/VAE embeddings to .pth
  python moss_image_lora_train.py --stage train           # 2) train from cache (no text encoder)
  python moss_image_lora_train.py                         # single-pass training
  python moss_image_lora_train.py --variant base          # train on MOSS-Image-Base instead of SFT
  python moss_image_lora_train.py --variant entropack     # 6-bit EntroPack DiT/TE (low VRAM)

  python moss_image_lora_train.py --stage train --lora-max-layer 27
      Inject LoRA only into layers.0..27 and leave the last layers untouched.
      MOSS-Image's DiT has 42 single-stream layers.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Environment  -- edit these
# --------------------------------------------------------------------------- #
PY         = r"D:\AI\python_embeded\python.exe"
DIFFSYNTH  = r"D:\AI\DiffSynth-Studio"                  # the checkout that is pip-installed
OFFICIAL   = {                                          # huggingface-cli --local-dir
    "sft":  r"D:\AI\MOSS-Image\SFT",
    "base": r"D:\AI\MOSS-Image\Base",
}
ENTROPACK  = r"D:\AI\MOSS-Image\EntroPack"              # DiffSynth-Studio/MOSS-Image-EntroPack
DATASET    = r"D:\AI\dataset"

TRAIN_PY   = os.path.join(DIFFSYNTH, "examples", "moss_image", "model_training", "train.py")
OUTPUT     = r"./models/train/moss_image_lora"
CACHE      = r"./data/cache/moss_image"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MAX_PIXELS = 1024 * 1024   # 1048576; drop to 786432 (~896x896) if VRAM is tight

NUM_LAYERS = 42                    # MOSSImageDiT default
LORA_PROJ  = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


# --------------------------------------------------------------------------- #
# Official diffusers layout. A multi-file model is ONE nested list, because
# hash_model_file() merges every shard's keys before hashing.
# --------------------------------------------------------------------------- #
def shards(root, subdir, pattern):
    files = sorted(glob.glob(os.path.join(root, subdir, pattern)))
    if not files:
        return None
    return files[0] if len(files) == 1 else files


def resolve_models(variant):
    """Return (DIT, TE, VAE, TOKENIZER) for the chosen variant."""
    if variant == "entropack":
        # EntroPack ships only the quantized DiT/TE; VAE + tokenizer come from SFT.
        official = OFFICIAL["sft"]
        dit = shards(ENTROPACK, os.path.join("models", "MOSS-Image-SFT"), "dit_6bit.safetensors")
        te  = shards(ENTROPACK, "models", "text_encoder_6bit.safetensors")
    else:
        official = OFFICIAL[variant]
        dit = shards(official, "transformer",  "diffusion_pytorch_model*.safetensors")
        te  = shards(official, "text_encoder", "model*.safetensors")
    vae = shards(official, "vae", "diffusion_pytorch_model*.safetensors")
    tok = os.path.join(official, "tokenizer")
    return dit, te, vae, (tok if os.path.isdir(tok) else None)


def require_models(names, variant):
    missing = [n for n, v in names if v is None]
    if missing:
        repo = "MOSS-Image-Base" if variant == "base" else "MOSS-Image-SFT"
        official = OFFICIAL["base" if variant == "base" else "sft"]
        msg = (
            f"Missing model files for: {', '.join(missing)}\n"
            f"Download with:\n"
            f'  huggingface-cli download openmoss/{repo} '
            f'--include "transformer/*" "text_encoder/*" "vae/*" "tokenizer/*" '
            f'--local-dir "{official}"'
        )
        if variant == "entropack":
            msg += (f'\n  huggingface-cli download DiffSynth-Studio/MOSS-Image-EntroPack '
                    f'--include "models/MOSS-Image-SFT/*" "models/text_encoder_6bit.safetensors" '
                    f'--local-dir "{ENTROPACK}"')
        sys.exit(msg)


def check_models(targets):
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
    targets = {k: v for k, v in targets.items() if v}
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


def layer_limited_targets(max_layer, suffixes=""):
    """
    Build an explicit LoRA target list covering layers.0..max_layer.

    The DiT block layout is fixed (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down},
    all >= 3072 wide), so names are generated rather than read from the file --
    this also works for the EntroPack checkpoint, whose keys are quantized.
    Note: DiffSynth's auto-search ("" targets) additionally hits text_embedder.*;
    this list deliberately covers only the main layers.
    `suffixes` optionally narrows it further, e.g. "q,k,v,o" for attention only.
    """
    want = [s.strip() for s in suffixes.split(",") if s.strip()]
    modules = []
    for i in range(max_layer + 1):
        for proj in LORA_PROJ:
            name = f"layers.{i}.proj.{proj}"
            if want and not any(name == w or name.endswith("." + w) for w in want):
                continue
            modules.append(name)
    return modules


def parse_args():
    p = argparse.ArgumentParser(description="MOSS-Image LoRA training wrapper")
    p.add_argument("--check", action="store_true", help="Only verify the model files, then exit.")
    p.add_argument("--variant", choices=["sft", "base", "entropack"], default="sft")
    p.add_argument("--stage", choices=["all", "cache", "train"], default="all")
    p.add_argument("--cache-path", default=CACHE)
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
                   help=f"Inject LoRA only into layers.0..N (DiT has {NUM_LAYERS} layers). "
                        "Combine with --lora-target-modules to also narrow by module name "
                        f"({','.join(LORA_PROJ)}).")
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

    DIT, TE, VAE, TOKENIZER = resolve_models(args.variant)

    if args.check:
        require_models([("transformer", DIT), ("text_encoder", TE), ("vae", VAE)], args.variant)
        check_models({"transformer": DIT, "text_encoder": TE, "vae": VAE})

    caching, from_cache = args.stage == "cache", args.stage == "train"

    # Load only what each stage actually needs.
    if from_cache:                      # embeddings are already baked
        needed = [("transformer", DIT)]
        models = [DIT]
    elif caching:                       # encoding prompts + images only
        needed = [("text_encoder", TE), ("vae", VAE), ("tokenizer", TOKENIZER)]
        models = [TE, VAE]
    else:                               # single pass: needs everything at once
        needed = [("transformer", DIT), ("text_encoder", TE), ("vae", VAE), ("tokenizer", TOKENIZER)]
        models = [DIT, TE, VAE]
    require_models(needed, args.variant)

    cmd = [
        PY, "-m", "accelerate.commands.launch",
        "--mixed_precision", "bf16",
        TRAIN_PY,
        "--model_paths", json.dumps(models),
        "--dataset_num_workers", str(args.num_workers),
        "--find_unused_parameters",
        "--output_path", args.cache_path if caching else args.output_path,
    ]
    # train.py falls back to downloading openmoss/MOSS-Image-Base:tokenizer/ if
    # --tokenizer_path is missing, so always pass the local one when we have it.
    if TOKENIZER:
        cmd += ["--tokenizer_path", TOKENIZER]

    # ---------- dataset ----------
    if from_cache:
        # no metadata_path -> UnifiedDataset scans base_path for *.pth
        cmd += ["--dataset_base_path", args.cache_path]
    else:
        max_pixels = args.max_pixels or auto_max_pixels(args.dataset_base_path)
        meta = args.metadata or str(Path(args.dataset_base_path) / "metadata.csv")
        cmd += [
            "--dataset_base_path",     args.dataset_base_path,
            "--dataset_metadata_path", meta,
            "--max_pixels",            str(max_pixels),
            "--data_file_keys",        "image",
        ]

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
            if not 0 <= args.lora_max_layer < NUM_LAYERS:
                sys.exit(f"--lora-max-layer must be in 0..{NUM_LAYERS - 1}")
            modules = layer_limited_targets(args.lora_max_layer, args.lora_target_modules)
            if not modules:
                sys.exit(f"--lora-max-layer {args.lora_max_layer} matched no modules. "
                         f"Check --lora-target-modules (valid: {','.join(LORA_PROJ)}).")
            per_layer = len(modules) / (args.lora_max_layer + 1)
            print(f"[lora-max-layer] layers 0-{args.lora_max_layer} -> {len(modules)} modules "
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
        # EntroPack weights are already quantized; don't stack fp8 on top of them.
        fp8_candidates = (VAE,) if args.variant == "entropack" else (TE, VAE)
        fp8 = [m for m in fp8_candidates if isinstance(m, str) and m in models]
        if fp8:
            cmd += ["--fp8_models", ",".join(fp8)]
        multi = [n for n, m in (("text_encoder", TE), ("vae", VAE))
                 if isinstance(m, list) and m in models]
        if multi:
            print(f"[warn] fp8 cannot be applied to multi-shard models: {', '.join(multi)}. "
                  f"Use --stage cache / --stage train instead.")
    if args.nf4:
        if args.variant == "entropack":
            sys.exit("--nf4 cannot be combined with --variant entropack (DiT is already 6-bit).")
        cmd += ["--quant_options", f"{json.dumps(DIT) if isinstance(DIT, list) else DIT}: bitsandbytes_nf4"]

    print(">>>", " ".join(cmd))
    # run from the DiffSynth checkout so `import diffsynth` and the example agree
    sys.exit(subprocess.call(cmd, cwd=DIFFSYNTH))


if __name__ == "__main__":
    main()
