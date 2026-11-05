"""Cheap on-device-style stroke feedback: baseline, slant, spacing, Arabic dots."""

from __future__ import annotations

import math


def baseline(stroke_points):
    ys = [p[1] for p in stroke_points]
    return sum(ys) / len(ys)


def slant(stroke_points):
    if len(stroke_points) < 2:
        return 0.0
    dx = stroke_points[-1][0] - stroke_points[0][0]
    dy = stroke_points[-1][1] - stroke_points[0][1]
    if abs(dy) < 1e-6:
        return 90.0
    return math.degrees(math.atan2(dx, -dy))


def spacing(boxes):
    if len(boxes) < 2:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    gaps = [boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)]
    return gaps


def feedback(strokes, lang: str = "ar"):
    flat = [p for s in strokes for p in s]
    if not flat:
        return []
    notes = []
    bl = baseline(flat)
    drift = max(p[1] - bl for p in flat) - min(p[1] - bl for p in flat)
    if drift > 18:
        notes.append("Your baseline drifts a lot — keep the line flatter under the letters.")
    sl = slant(flat)
    if lang == "ar" and abs(sl) > 40:
        notes.append("Arabic letters join better with a slight, consistent slant.")
    if lang == "en" and abs(sl) > 35:
        notes.append("Keep one slant angle across the whole line.")
    if lang == "ar":
        notes.append("Check dot placement: above/below the letter body, never floating far away.")
    return notes
