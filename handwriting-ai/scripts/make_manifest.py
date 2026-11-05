"""Build a manifest.csv from images + per-image text files.

Layout:
  images/*.png      (or jpg/jpeg)
  labels/*.txt      (same basename, UTF-8 text)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARABIC = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءةىأإآؤئ")


def guess_lang(text: str) -> str:
    return "ar" if any(c in ARABIC for c in text) else "en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images = Path(args.images)
    labels = Path(args.labels)
    rows = []
    for img in sorted(images.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        txt = labels / (img.stem + ".txt")
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        rows.append((str(img), text, guess_lang(text)))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "text", "lang"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
