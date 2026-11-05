"""Model skeletons. Train recognition first; the improver is the research bet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import VOCAB_SIZE


class CRNNRecognizer(nn.Module):
    """CNN + BiLSTM + CTC, TrOCR-sized alternative that is easy to train."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(256 * 2, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        f = self.cnn(x)
        b, c, h, w = f.shape
        seq = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(seq)
        return F.log_softmax(self.head(out), dim=-1)  # (B, T, V)


class StrokeImprover(nn.Module):
    """Student strokes in, ideal strokes out (sequence-to-sequence)."""

    def __init__(self, points: int = 3, d_model: int = 128, nhead: int = 4, layers: int = 3, ff: int = 512):
        super().__init__()
        self.points = points
        self.embed = nn.Linear(points, d_model)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True), layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, ff, batch_first=True), layers)
        self.out = nn.Linear(d_model, points)

    def forward(self, student, memory_mask=None, target=None):
        # student: (B, S, P, 3) with mask (B, S)
        b, s, p, _ = student.shape
        src = self.embed(student.reshape(b * s, p, 3))
        src = self.enc(src)
        if target is not None:
            tgt = self.embed(target.reshape(b * s, p, 3))
            tgt = self.dec(tgt, src)
            return self.out(tgt).reshape(b, s, p, 3)
        return self.out(src).reshape(b, s, p, 3)


def ctc_loss(logits, targets, target_lengths, blank: int = 0):
    log_probs = logits.transpose(0, 1)  # (T, B, V)
    input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long, device=log_probs.device)
    return nn.CTCLoss(blank=blank, zero_infinity=True)(log_probs, targets, input_lengths, target_lengths)
