import math
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.dk = n_heads, d_model // n_heads
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        b, t, d = x.shape
        def split(layer):
            return layer(x).view(b, t, self.n_heads, self.dk).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if padding_mask is not None:
            if padding_mask.all(dim=1).any():
                raise ValueError("Each sequence needs at least one non-padding token")
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        context = (self.dropout(weights) @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = self.norm1(x + self.dropout(self.out(context)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return (x, weights) if return_attention else x
