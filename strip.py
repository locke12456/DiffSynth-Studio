import re, sys
from safetensors.torch import load_file, save_file

src, dst, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sd = load_file(src)
pat = re.compile(r"transformer_blocks\.(\d+)\.")
kept = {}
dropped = 0
for k, v in sd.items():
    m = pat.search(k)
    if m and lo <= int(m.group(1)) <= hi:
        dropped += 1
        continue
    kept[k] = v
print(f"kept {len(kept)} / dropped {dropped}")
print("sample:", list(sd)[0])
save_file(kept, dst)
