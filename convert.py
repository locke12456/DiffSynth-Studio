r"""
Convert DiffSynth-trained Anima LoRA (PEFT format) to ComfyUI format.

Key transform:
    blocks.0.cross_attn.k_proj.lora_A.default.weight
 -> diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight

Usage:
    python convert_lora.py <input.safetensors> [output.safetensors]
"""
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

PREFIX = "diffusion_model."  # ComfyUI LoRA convention


def convert(src: Path, dst: Path) -> None:
    sd = load_file(str(src))
    out = {}
    skipped = []
    for k, v in sd.items():
        nk = k
        # PEFT ".default" infix -> remove
        nk = nk.replace(".lora_A.default.weight", ".lora_A.weight")
        nk = nk.replace(".lora_B.default.weight", ".lora_B.weight")
        # ComfyUI "diffusion_model." prefix
        if not nk.startswith(PREFIX):
            nk = PREFIX + nk
        if nk == k:
            skipped.append(k)
        out[nk] = v

    save_file(out, str(dst))
    print(f"OK -> {dst}")
    print(f"     {len(out)} keys ({len(skipped)} unchanged)")
    # Show samples
    sample_in = list(sd.keys())[:2]
    sample_out = list(out.keys())[:2]
    print("Sample:")
    for a, b in zip(sample_in, sample_out):
        print(f"  {a}\n  -> {b}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"ERR file not found: {src}")
        sys.exit(1)
    dst = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else src.with_name(src.stem + "_comfy.safetensors")
    )
    convert(src, dst)
