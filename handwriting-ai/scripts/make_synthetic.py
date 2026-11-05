"""Generate a tiny synthetic dataset to smoke-test the training loop."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.out)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    words = ["hello", "paper", "study", "write", "note", "ink", "line", "hand"]
    font = ImageFont.load_default()

    rng = random.Random(7)
    manifest_rows = []
    for i in range(args.n):
        text = rng.choice(words)
        img = Image.new("L", (220, 48), 255)
        draw = ImageDraw.Draw(img)
        draw.text((8, 10), text, fill=0, font=font)
        name = f"{i:04d}.png"
        img.save(img_dir / name)
        manifest_rows.append((str(img_dir / name), text, "en"))

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text", "lang"])
        w.writerows(manifest_rows)

    # Synthetic student->ideal stroke pairs: ideal is smooth, student noisy.
    def ideal(n=40):
        t = np.linspace(0, 2 * np.pi, n)
        x = np.cumsum(np.sin(t) * 2 + 4)
        y = 40 + 18 * np.sin(t * 2)
        p = np.full(n, 0.7)
        return np.stack([x, y, p], axis=1)

    pairs = []
    for _ in range(args.n):
        t = ideal(48)
        s = t.copy()
        s[:, 0] += np.random.normal(0, 2.2, len(s))
        s[:, 1] += np.random.normal(0, 2.8, len(s))
        s[:, 1] += np.linspace(-6, 6, len(s))  # slant drift
        pairs.append({"text": "demo", "lang": "en",
                      "student": [s.tolist()],
                      "target": [t.tolist()]})

    (root / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    print("wrote", root / "manifest.csv", "and", root / "pairs.json")


if __name__ == "__main__":
    main()
