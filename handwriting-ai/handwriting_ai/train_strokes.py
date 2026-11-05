"""Train the stroke improver: python -m handwriting_ai.train_strokes --pairs pairs.json"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import StrokeImprover
from .strokes import StrokePairDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="JSON list of student/target stroke pairs")
    ap.add_argument("--out", default="checkpoints/improver.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    ds = StrokePairDataset(args.pairs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    model = StrokeImprover()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tqdm(dl, desc=f"epoch {epoch + 1}"):
            pred = model(batch["student"], target=batch["target"])
            loss = torch.nn.functional.mse_loss(pred, batch["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1} mse {total / max(1, len(dl)):.4f}")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
