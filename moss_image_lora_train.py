#!/usr/bin/env python3
r"""
DiffSynth-Studio  MOSS-Image  LoRA training wrapper.

Usage:
  python moss_image_lora_train.py --download              # fetch missing Base DiT / 6-bit TE / VAE / tokenizer
  python moss_image_lora_train.py --check                 # verify every model file loads
  python moss_image_lora_train.py --stage cache --dry-run # preflight only, print the command
  python moss_image_lora_train.py --stage cache           # 1) bake text/VAE embeddings to .pth
  python moss_image_lora_train.py --stage train           # 2) train from cache (no text encoder)
  python moss_image_lora_train.py                         # single-pass training

Model files are picked up from the folders below (searched recursively, so the
huggingface-cli --local-dir subfolders are fine). Other files in the same folders,
e.g. ComfyUI checkpoints, are ignored:
  DIT_DIR        transformer/diffusion_pytorch_model*.safetensors  (MOSS-Image-Base/SFT)
                 or dit_6bit.safetensors                           (EntroPack)
  TE_DIR         text_encoder_6bit.safetensors                     (EntroPack, preferred)
                 or text_encoder/model*.safetensors                (bf16)
  VAE_DIR        vae/diffusion_pytorch_model*.safetensors
  TOKENIZER_DIR  tokenizer.json etc.

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
PY = r"D:\AI\venv-diffsynth\Scripts\python.exe"
DIFFSYNTH  = r"D:\AI\DiffSynth-Studio"                  # the checkout that is pip-installed
MOSS_ROOT     = r"D:\AI\MOSS-Image"
DIT_DIR       = os.path.join(MOSS_ROOT, "diffusion")
TE_DIR        = os.path.join(MOSS_ROOT, "text-encoder")
VAE_DIR       = os.path.join(MOSS_ROOT, "vae")
TOKENIZER_DIR = os.path.join(MOSS_ROOT, "tokenizer")
DATASET    = r"D:\AI\dataset"

TRAIN_PY   = os.path.join(DIFFSYNTH, "examples", "moss_image", "model_training", "train.py")
OUTPUT     = r"./models/train/moss_image_lora"
CACHE      = r"./data/cache/moss_image"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MAX_PIXELS = 1024 * 1024   # 1048576; drop to 786432 (~896x896) if VRAM is tight

NUM_LAYERS = 42                    # MOSSImageDiT default
LORA_PROJ  = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


# --------------------------------------------------------------------------- #
# Model discovery. A multi-file model is ONE nested list, because
# hash_model_file() merges every shard's keys before hashing.
# --------------------------------------------------------------------------- #
def find(root, pattern):
    files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not files:
        return None
    return files[0] if len(files) == 1 else files


def find_tokenizer(root):
    hits = glob.glob(os.path.join(root, "**", "tokenizer.json"), recursive=True)
    return os.path.dirname(sorted(hits)[0]) if hits else None


def is_prequantized(m):
    """EntroPack weights are already 6-bit; fp8/nf4 must not be stacked on top."""
    return isinstance(m, str) and "_6bit" in os.path.basename(m)


def resolve_models():
    """Return (DIT, TE, VAE, TOKENIZER)."""
    dit = find(DIT_DIR, "diffusion_pytorch_model*.safetensors") or find(DIT_DIR, "dit_6bit.safetensors")
    te  = find(TE_DIR, "text_encoder_6bit.safetensors") or find(TE_DIR, "model*.safetensors")
    vae = find(VAE_DIR, "diffusion_pytorch_model*.safetensors")
    return dit, te, vae, find_tokenizer(TOKENIZER_DIR)


# label -> (repo_id, allow_patterns, local_dir)
DOWNLOADS = {
    "transformer":  ("OpenMOSS-Team/MOSS-Image-Base", ["transformer/*"], DIT_DIR),
    "text_encoder": ("DiffSynth-Studio/MOSS-Image-EntroPack", ["models/text_encoder_6bit.*"], TE_DIR),
    "vae":          ("OpenMOSS-Team/MOSS-Image-Base", ["vae/*"], VAE_DIR),
    "tokenizer":    ("OpenMOSS-Team/MOSS-Image-Base", ["tokenizer/*"], MOSS_ROOT),
}


def download_models(labels):
    """Fetch the given models with huggingface_hub (no huggingface-cli needed)."""
    from huggingface_hub import snapshot_download
    for label in labels:
        repo, patterns, local_dir = DOWNLOADS[label]
        print(f"[download] {label}: {repo} {patterns} -> {local_dir}")
        snapshot_download(repo_id=repo, allow_patterns=patterns, local_dir=local_dir)


def require_models(names):
    missing = [n for n, v in names if v is None]
    if missing:
        sys.exit(f"Missing model files for: {', '.join(missing)}\n"
                 f"Run:  python {os.path.basename(__file__)} --download")


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
    return subprocess.call([PY, "-c", code, json.dumps(targets)], cwd=DIFFSYNTH)


ENV_PROBE = r'''
import importlib, json, sys
need = json.loads(sys.argv[1])
problems = []
def probe(label, stmt, fix):
    try:
        exec(stmt, {})
        print(f"[env] ok    {label}")
    except Exception as e:
        print(f"[env] FAIL  {label}: {type(e).__name__}: {e}".splitlines()[0])
        problems.append(fix)
probe("torch + CUDA", "import torch\nassert torch.cuda.is_available(), 'CUDA not available'", "torch")
probe("transformers Qwen3.5", "from transformers import Qwen3_5TextConfig", '"transformers>=5"')
probe("accelerate", "import accelerate", "accelerate")
probe("peft", "import peft", "peft")
probe("diffsynth MOSS pipeline", "from diffsynth.pipelines.moss_image import MOSSImagePipeline", "-e " + need["diffsynth"])
if need["entropack"]:
    probe("entropack (6-bit weights)",
          "import torch\nv = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])\n"
          "assert v >= (2, 10), f'entropack needs torch>=2.10, found {torch.__version__}'\n"
          "import entropack as ep\nassert hasattr(ep, 'CompressedLinear')",
          '"entropack[cuda13]"')
if need["bitsandbytes"]:
    probe("bitsandbytes (--nf4)", "import bitsandbytes", "bitsandbytes")
print(json.dumps(problems))
'''


def check_env(models, nf4):
    """Import everything train.py will need, in the interpreter it will run under."""
    if not os.path.isfile(PY):
        print(f"[env] FAIL  python not found: {PY}")
        return False
    need = {"diffsynth": DIFFSYNTH, "bitsandbytes": nf4,
            "entropack": any(is_prequantized(m) for m in models)}
    out = subprocess.run([PY, "-c", ENV_PROBE, json.dumps(need)], cwd=DIFFSYNTH,
                         capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    print("\n".join(lines[:-1]))
    try:
        problems = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        print(out.stderr)
        return False
    if "torch" in problems:
        problems.remove("torch")
        print(f"[env] fix torch with:\n  uv pip install --python {PY} torch torchvision "
              f"--index-url https://download.pytorch.org/whl/cu130")
    if problems:
        print(f"[env] fix with:\n  uv pip install --python {PY} " + " ".join(problems))
    return "[env] FAIL" not in out.stdout


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
    p.add_argument("--download", action="store_true", help="Download missing model files, then exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run all preflight checks and print the train.py command without launching it.")
    p.add_argument("--check", action="store_true", help="Only verify the model files, then exit.")
    p.add_argument("--stage", choices=["all", "cache", "train"], default="all")
    p.add_argument("--cache-path", default=CACHE)
    # dataset
    p.add_argument("--dataset-base-path", default=DATASET)
    p.add_argument("--metadata", default=None)
    # Defaults below are a style-LoRA recipe; see MOSS-LoRA-Report.md for sources.
    p.add_argument("--dataset-repeat", type=int, default=10)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0, help="Windows: keep 0")
    # training
    p.add_argument("--learning-rate", default="1e-4")
    p.add_argument("--weight-decay", default="0.01")
    p.add_argument("--num-epochs", type=int, default=10,
                   help="One checkpoint per epoch (unless --save-steps); the best style is often not the last.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-target-modules", default="",
                   help=f"Narrow by module name ({','.join(LORA_PROJ)}); empty = all of them.")
    p.add_argument("--lora-text-embedder", action="store_true",
                   help="Also train text_embedder (DiffSynth auto targets). Off for style LoRAs.")
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
    # train.py runs with cwd=DIFFSYNTH, so pin relative paths there for both sides.
    for k in ("cache_path", "output_path", "dataset_base_path", "metadata"):
        v = getattr(args, k)
        if v and not os.path.isabs(v):
            setattr(args, k, os.path.normpath(os.path.join(DIFFSYNTH, v)))
    os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    DIT, TE, VAE, TOKENIZER = resolve_models()
    if args.download:
        found = {"transformer": DIT, "text_encoder": TE, "vae": VAE, "tokenizer": TOKENIZER}
        missing = [k for k, v in found.items() if v is None]
        download_models(missing)
        if not missing:
            print("[download] all models already present")
        DIT, TE, VAE, TOKENIZER = resolve_models()
    for label, m in (("transformer", DIT), ("text_encoder", TE), ("vae", VAE), ("tokenizer", TOKENIZER)):
        print(f"[model] {label:<13} {json.dumps(m) if isinstance(m, list) else m}")
    if args.download and not args.check:
        return

    if args.check:
        require_models([("transformer", DIT), ("text_encoder", TE), ("vae", VAE)])
        env_ok = check_env([DIT, TE, VAE], args.nf4)
        sys.exit(check_models({"transformer": DIT, "text_encoder": TE, "vae": VAE}) or (0 if env_ok else 1))

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
    require_models(needed)

    # Fail here, not after accelerate has spun up and loaded half the weights.
    errors = []
    if from_cache:
        if not glob.glob(os.path.join(args.cache_path, "**", "*.pth"), recursive=True):
            errors.append(f"no cached *.pth under {args.cache_path} -- run --stage cache first")
    else:
        meta = args.metadata or os.path.join(args.dataset_base_path, "metadata.csv")
        if not os.path.isfile(meta):
            errors.append(f"metadata not found: {meta}")
    if not check_env(models, args.nf4):
        errors.append("python environment is missing packages (see [env] above)")
    if errors:
        sys.exit("\n".join(f"[preflight] {e}" for e in errors))

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
        if args.lora_text_embedder:
            if args.lora_max_layer is not None or args.lora_target_modules:
                sys.exit("--lora-text-embedder uses DiffSynth's auto targets; "
                         "drop --lora-max-layer / --lora-target-modules.")
            targets = ""
        else:
            # Style default: every main layer, attention + MLP, but not text_embedder
            # (a bare "ffn_down" suffix would also hit text_embedder.blocks.*).
            max_layer = NUM_LAYERS - 1 if args.lora_max_layer is None else args.lora_max_layer
            if not 0 <= max_layer < NUM_LAYERS:
                sys.exit(f"--lora-max-layer must be in 0..{NUM_LAYERS - 1}")
            modules = layer_limited_targets(max_layer, args.lora_target_modules)
            if not modules:
                sys.exit(f"--lora-max-layer {max_layer} matched no modules. "
                         f"Check --lora-target-modules (valid: {','.join(LORA_PROJ)}).")
            per_layer = len(modules) / (max_layer + 1)
            print(f"[lora] layers 0-{max_layer} -> {len(modules)} modules "
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
        fp8 = [m for m in (TE, VAE) if isinstance(m, str) and m in models and not is_prequantized(m)]
        if fp8:
            cmd += ["--fp8_models", ",".join(fp8)]
        multi = [n for n, m in (("text_encoder", TE), ("vae", VAE))
                 if isinstance(m, list) and m in models]
        if multi:
            print(f"[warn] fp8 cannot be applied to multi-shard models: {', '.join(multi)}. "
                  f"Use --stage cache / --stage train instead.")
    if args.nf4:
        if is_prequantized(DIT):
            sys.exit("--nf4 cannot be used with the EntroPack DiT (already 6-bit).")
        cmd += ["--quant_options", f"{json.dumps(DIT) if isinstance(DIT, list) else DIT}: bitsandbytes_nf4"]

    print(">>>", subprocess.list2cmdline(cmd))
    if args.dry_run:
        print("[dry-run] preflight passed; not launching.")
        return
    # run from the DiffSynth checkout so `import diffsynth` and the example agree
    sys.exit(subprocess.call(cmd, cwd=DIFFSYNTH))


if __name__ == "__main__":
    main()
