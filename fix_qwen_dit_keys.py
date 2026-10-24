#!/usr/bin/env python3
"""
Diagnose (and if possible repair) a ComfyUI-repacked single-file checkpoint
so DiffSynth-Studio can detect it.

  python fix_qwen_dit_keys.py "D:\\AI\\Qwen Image\\2.1\\diffusion_models\\qwen_image_2.1.safetensors"
  python fix_qwen_dit_keys.py <file> --write out.safetensors
"""
import argparse, hashlib, os, sys
from collections import Counter

from safetensors import safe_open

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.core.loader.file import convert_keys_dict_to_single_str

# prefixes ComfyUI / kohya / civitai repacks commonly bolt on
CANDIDATE_PREFIXES = [
    "", "model.diffusion_model.", "diffusion_model.", "model.",
    "transformer.", "net.", "module.", "pipe.dit.",
]


def hash_keys(keys_dict):
    s = convert_keys_dict_to_single_str(keys_dict, with_shape=True)
    return hashlib.md5(s.encode("UTF-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", default=None, help="Write the fixed file here (needs ~2x file size free).")
    args = ap.parse_args()

    with safe_open(args.path, framework="pt", device="cpu") as f:
        keys_dict = {k: f.get_slice(k).get_shape() for k in f.keys()}

    print(f"{len(keys_dict)} tensors")
    print("\n-- first 8 keys --")
    for k in list(keys_dict)[:8]:
        print(f"  {k}  {keys_dict[k]}")

    # what do the keys start with?
    print("\n-- top-level name segments --")
    for seg, n in Counter(k.split(".")[0] for k in keys_dict).most_common(10):
        print(f"  {seg:<28} {n}")

    print(f"\ncurrent hash: {hash_keys(keys_dict)}   (not in MODEL_CONFIGS)")

    registered = {c["model_hash"]: c for c in MODEL_CONFIGS}

    print("\n-- trying prefix strips --")
    hit = None
    for pre in CANDIDATE_PREFIXES:
        if pre and not any(k.startswith(pre) for k in keys_dict):
            continue
        stripped = {(k[len(pre):] if k.startswith(pre) else k): v for k, v in keys_dict.items()}
        h = hash_keys(stripped)
        cfg = registered.get(h)
        mark = f"  <== MATCH: {cfg['model_name']}" if cfg else ""
        print(f"  strip {pre!r:<26} -> {h}{mark}")
        if cfg and hit is None:
            hit = (pre, cfg)

    if hit is None:
        print("\nNo prefix strip matches. The repack renamed keys, not just prefixed them.")
        print("Download the official shards instead:")
        print('  huggingface-cli download Qwen/Qwen-Image-2.1 --include "transformer/*" --local-dir <dir>')
        sys.exit(2)

    pre, cfg = hit
    print(f"\nFixable: strip {pre!r} -> {cfg['model_name']}")
    if not args.write:
        print("Re-run with --write <out.safetensors> to produce the converted file.")
        return

    from safetensors.torch import save_file
    print("Loading full tensors (this needs RAM roughly the size of the file)...")
    with safe_open(args.path, framework="pt", device="cpu") as f:
        sd = {(k[len(pre):] if k.startswith(pre) else k): f.get_tensor(k) for k in f.keys()}
    os.makedirs(os.path.dirname(os.path.abspath(args.write)) or ".", exist_ok=True)
    save_file(sd, args.write)
    print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()
