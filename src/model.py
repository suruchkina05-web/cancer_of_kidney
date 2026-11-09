"""Attention-MIL (gated attention, Ilse et al. 2018): срез = «мешок» патчей."""
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, in_dim=2048, hid=256, n_classes=3, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(hid, 128), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(hid, 128), nn.Sigmoid())
        self.att_w = nn.Linear(128, 1)
        self.classifier = nn.Linear(hid, n_classes)

    def forward(self, x):
        """x: (N патчей, in_dim) -> logits (n_classes,), внимание (N,)"""
        h = self.embed(x)
        a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1)
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * h).sum(dim=0)
        return self.classifier(z), a
