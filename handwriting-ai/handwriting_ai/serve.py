"""Local HTTP bridge for the site's handwriting AI toggle.

Run: python -m handwriting_ai.serve --model checkpoints/recognizer_khatt.pt --port 8765
The whiteboard posts {image: dataURL, lang} and gets {text, confidence} back.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import VOCAB_SIZE, ITOI, PAD_IDX
from .models import CRNNRecognizer


def load_model(path: str):
    model = CRNNRecognizer(vocab_size=VOCAB_SIZE)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img: Image.Image, height: int = 32):
    img = ImageOps.grayscale(img)
    w, h = img.size
    ratio = height / max(h, 1)
    new_w = max(1, int(w * ratio))
    img = img.resize((new_w, height), Image.LANCZOS)
    gray = np.asarray(img, dtype=np.float32) / 255.0
    dark = gray.mean() < 0.5
    arr = gray if dark else 1.0 - gray  # ink = 1
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return (t - 0.5) / 0.5


def greedy_decode(logits):
    probs = logits.softmax(-1)[0]  # (T, V)
    ids = probs.argmax(-1).tolist()
    chars, prev = [], None
    conf_sum, conf_n = 0.0, 0
    for t, (i, p) in enumerate(zip(ids, probs.max(-1).values.tolist())):
        if i != PAD_IDX and i != prev:
            conf_sum += p
            conf_n += 1
            chars.append(ITOI.get(i, ""))
        prev = i if i != PAD_IDX else None
    text = "".join(chars)
    confidence = conf_sum / conf_n if conf_n else 0.0
    return text, confidence


class Handler(BaseHTTPRequestHandler):
    model = None

    def do_POST(self):
        if self.path != "/feedback":
            self.send_error(404, "not found")
            return
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            data_url = body.get("image", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            raw = base64.b64decode(data_url)
            img = Image.open(BytesIO(raw))
            x = preprocess(img)
            with torch.no_grad():
                logits = self.model(x)
            text, confidence = greedy_decode(logits)
            payload = {"text": text, "confidence": round(confidence, 4)}
            self._json(200, payload)
        except Exception as exc:  # keep the bridge alive on bad input
            self._json(500, {"error": str(exc)})

    def do_GET(self):
        root = Path(__file__).resolve().parent.parent.parent
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/index.html"
        if "\x00" in path or ".." in Path(path).parts:
            self.send_error(400, "bad path")
            return
        file_path = (root / path.lstrip("/")).resolve()
        try:
            file_path.relative_to(root)
        except ValueError:
            self.send_error(403, "outside root")
            return
        if not file_path.is_file():
            self.send_error(404, "not found")
            return
        ctype, _ = mimetypes.guess_type(str(file_path))
        self.send_response(200)
        self.send_header("content-type", ctype or "application/octet-stream")
        self.send_header("content-length", str(file_path.stat().st_size))
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "recognizer_khatt.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.model = load_model(args.model)
    print(f"handwriting bridge listening on http://{args.host}:{args.port} (model={args.model})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
