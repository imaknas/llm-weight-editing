"""用 torch 訓練 wse.mlp.MLP(Adam、小批次)。純 Python 的 MLP.train 在幾千個權重以上太慢時用這個。

MLP 仍是唯一的權重表示;這裡只在訓練時把它搬進張量,訓練完寫回去。
"""
from __future__ import annotations

import random

import torch

from .mlp import MLP


def adam_trainer(epochs: int, lr: float = 0.01, batch_size: int = 64):
    """回傳符合 wse.defects.Trainer 的函數。亂數種子取自傳入的 rng,結果可重現。"""

    def train(net: MLP, data: list[tuple[list[float], int]], rng: random.Random) -> None:
        gen = torch.Generator().manual_seed(rng.getrandbits(63))
        Ws = [torch.tensor(W, dtype=torch.float64, requires_grad=True) for W in net.weights]
        bs = [torch.tensor(b, dtype=torch.float64, requires_grad=True) for b in net.biases]
        X = torch.tensor([x for x, _ in data], dtype=torch.float64)
        Y = torch.tensor([[float(y)] for _, y in data], dtype=torch.float64)
        opt = torch.optim.Adam(Ws + bs, lr=lr)
        for _ in range(epochs):
            order = torch.randperm(len(data), generator=gen)
            for start in range(0, len(data), batch_size):
                idx = order[start:start + batch_size]
                h = X[idx]
                for l, (W, b) in enumerate(zip(Ws, bs)):
                    h = h @ W.T + b
                    if l < len(Ws) - 1:
                        h = torch.tanh(h)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(h, Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.weights = [W.detach().tolist() for W in Ws]
        net.biases = [b.detach().tolist() for b in bs]

    return train
