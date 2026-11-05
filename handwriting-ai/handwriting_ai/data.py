"""Manifest-driven datasets for handwritten lines (Arabic + English).

Expected CSV manifest: path,text,lang
  path  : image file (any size)
  text  : ground-truth transcription
  lang  : 'ar' or 'en'
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CHARS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-()/"
    "اابتثجحخدذرزسشصضطظعغفقكلمنهويءةىأإآؤئ "
)
STOI = {c: i + 1 for i, c in enumerate(CHARS)}
ITOI = {i: c for c, i in STOI.items()}
PAD_IDX = 0
VOCAB_SIZE = len(STOI) + 1


def encode_text(text: str) -> list[int]:
    return [STOI.get(c, STOI[" "]) for c in text]


def decode_text(ids: list[int]) -> str:
    return "".join(ITOI.get(i, "") for i in ids)


def load_manifest(path: str | Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["path"], row["text"], row.get("lang", "en")))
    return rows


def resize_height(img: Image.Image, height: int = 32, min_width: int = 0) -> Image.Image:
    img = ImageOps.grayscale(img)
    w, h = img.size
    ratio = height / max(h, 1)
    new_w = max(1, int(w * ratio))
    if min_width > new_w:
        new_w = min_width
    return img.resize((new_w, height), Image.LANCZOS)


class LineImageDataset(Dataset):
    def __init__(self, manifest: str | Path, root: str | Path = ".", height: int = 32):
        self.rows = load_manifest(manifest)
        self.root = Path(root)
        self.height = height

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel, text, lang = self.rows[idx]
        y = torch.tensor(encode_text(text), dtype=torch.long)
        img = resize_height(Image.open(self.root / rel), self.height, min_width=len(y) * 4 + 16)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = 1.0 - arr  # ink = 1
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return {"image": x, "target": y, "lang": lang, "text": text}


def collate_line(batch):
    heights = batch[0]["image"].shape[1]
    width = max(b["image"].shape[2] for b in batch)
    images = torch.zeros(len(batch), 1, heights, width)
    targets, target_lengths = [], []
    for i, b in enumerate(batch):
        w = b["image"].shape[2]
        images[i, :, :, :w] = b["image"]
        targets.append(b["target"])
        target_lengths.append(len(b["target"]))
    images = (images - 0.5) / 0.5
    flat = torch.cat(targets)
    return {
        "images": images,
        "targets": flat,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "lang": [b["lang"] for b in batch],
        "text": [b["text"] for b in batch],
    }
