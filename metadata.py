"""
Generate metadata.csv for Anima LoRA training.

Recursively scans D:\\AI\\dataset/ for images and matching .clean.txt or .txt captions.
"""
import csv
import pathlib

DATASET = pathlib.Path(r"D:\AI\dataset")
TRIGGER = "saika_magoichi"
OUT = DATASET / "metadata.csv"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_caption(img_path: pathlib.Path) -> str | None:
    for suffix in (".clean.txt", ".txt"):
        cap = img_path.with_suffix(suffix)
        if cap.exists():
            return cap.read_text(encoding="utf-8").strip()
    return None


rows = []
missing = []
for img in sorted(DATASET.rglob("*")):
    if not img.is_file() or img.suffix.lower() not in IMG_EXT:
        continue

    cap = find_caption(img)
    rel = img.relative_to(DATASET).as_posix()   # e.g. "group1/001.jpg"
    if cap is None:
        missing.append(rel)
        continue

    tags = [t.strip() for t in cap.split(",") if t.strip()]
    if TRIGGER in tags:
        tags.remove(TRIGGER)
    prompt = f"{TRIGGER}, " + ", ".join(tags)

    rows.append({"image": rel, "prompt": prompt})

with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "prompt"])
    w.writeheader()
    w.writerows(rows)

print(f"OK  -> {OUT}")
print(f"     {len(rows)} rows written")
if missing:
    print(f"WARN missing captions for {len(missing)} images:")
    for m in missing[:10]:
        print(f"  {m}")
    if len(missing) > 10:
        print(f"  ...and {len(missing)-10} more")
