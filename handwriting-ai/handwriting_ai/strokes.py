"""Stroke-sequence datasets and normalization for handwriting improvement.

Pair file: JSON list of {"text": str, "lang": "ar"|"en",
                         "student": [[[x,y,p], ...], ...],
                         "target":  [[[x,y,p], ...], ...]}
Coordinates are raw canvas points; we normalize per sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_strokes(strokes, size: int = 64):
    pts = np.concatenate(strokes, axis=0)[:, :2].astype(np.float32)
    lo, hi = pts.min(0), pts.max(0)
    scale = max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    center = (lo + hi) / 2.0

    def norm(s):
        s = np.asarray(s, dtype=np.float32).copy()
        s[:, :2] = (s[:, :2] - center) / scale * (size * 0.86)
        return s

    return [norm(s) for s in strokes]


def resample(stroke, n: int = 32):
    pts = np.asarray(stroke, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0)
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(d)])
    if arc[-1] < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0, arc[-1], n)
    out = np.stack([
        np.interp(targets, arc, pts[:, i]) for i in range(pts.shape[1])
    ], axis=1)
    return out.astype(np.float32)


class StrokePairDataset(Dataset):
    def __init__(self, pairs_path: str | Path, max_strokes: int = 8, points_per_stroke: int = 32, size: int = 64):
        self.pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
        self.max_strokes = max_strokes
        self.points = points_per_stroke
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def _stack(self, strokes):
        strokes = normalize_strokes(strokes, self.size)
        out = np.zeros((self.max_strokes, self.points, 3), dtype=np.float32)
        mask = np.zeros(self.max_strokes, dtype=np.float32)
        for i, s in enumerate(strokes[: self.max_strokes]):
            r = resample(s, self.points)
            if r.shape[1] < 3:
                r = np.concatenate([r, np.ones((len(r), 3 - r.shape[1]), dtype=np.float32)], axis=1)
            out[i] = r[:, :3]
            mask[i] = 1.0
        return out, mask

    def __getitem__(self, idx):
        item = self.pairs[idx]
        sx, sm = self._stack(item["student"])
        tx, tm = self._stack(item["target"])
        return {
            "student": torch.from_numpy(sx),
            "student_mask": torch.from_numpy(sm),
            "target": torch.from_numpy(tx),
            "target_mask": torch.from_numpy(tm),
            "lang": item.get("lang", "en"),
            "text": item.get("text", ""),
        }
