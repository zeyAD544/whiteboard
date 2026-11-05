"""Stream a small KHATT subset into a local manifest for research training.

Note: KHATT is distributed for research purposes. Do not redistribute the
images or use them for commercial products without checking the license.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_khatt")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    root = Path(args.out)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("johnlockejrr/KHATT_v1.0_dataset", split="train", streaming=True)
    rows = []
    for i, item in enumerate(ds):
        if i >= args.n:
            break
        img = item["image"].convert("L")
        text = (item.get("text") or "").strip()
        if not text or len(text) > 80:
            continue
        name = f"{i:05d}.png"
        img.save(img_dir / name)
        rows.append((str(img_dir / name), text, "ar"))

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text", "lang"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
