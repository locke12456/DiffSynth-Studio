import sys
from safetensors.torch import load_file, save_file

src = sys.argv[1]
dst = src.replace(".safetensors", "_comfy.safetensors")
sd = load_file(src)
new = {}
for k, v in sd.items():
    if k.startswith("diffusion_model.lora_"):
        k = k[len("diffusion_model."):]
    new[k] = v
save_file(new, dst)
print("saved:", dst, len(new), "keys")
