"""Train the CTC recognizer: python -m handwriting_ai.train_recognition --data manifest.csv"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LineImageDataset, collate_line
from .models import CRNNRecognizer, ctc_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV manifest: path,text,lang")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="checkpoints/recognizer.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    ds = LineImageDataset(args.data, args.root)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_line)
    model = CRNNRecognizer()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tqdm(dl, desc=f"epoch {epoch + 1}"):
            input_lengths = torch.full((batch["images"].size(0),), batch["images"].size(3) // 4, dtype=torch.long)
            if int(batch["target_lengths"].max()) > int(input_lengths.max()):
                print("skip batch: target longer than time axis")
                continue
            logits = model(batch["images"])
            loss = ctc_loss(logits, batch["targets"], batch["target_lengths"])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                model.zero_grad(set_to_none=True)
                continue
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1} loss {total / max(1, len(dl)):.4f}")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
