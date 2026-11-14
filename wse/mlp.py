"""純標準庫的小型 MLP:tanh 隱藏層、sigmoid 輸出,做二元分類。

隨機性一律由呼叫端傳入的 random.Random 提供,方便重現。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class MLP:
    sizes: list[int]
    # weights[l][j][i]:第 l 層第 j 個輸出單元接到第 i 個輸入的權重
    weights: list[list[list[float]]] = field(default_factory=list)
    biases: list[list[float]] = field(default_factory=list)

    @classmethod
    def random_init(cls, sizes: list[int], rng: random.Random) -> "MLP":
        ws, bs = [], []
        for n_in, n_out in zip(sizes, sizes[1:]):
            scale = 1.0 / math.sqrt(n_in)
            ws.append([[rng.uniform(-scale, scale) * 2 for _ in range(n_in)] for _ in range(n_out)])
            bs.append([0.0] * n_out)
        return cls(sizes, ws, bs)

    # ---- 推論 ----
    def _forward_all(self, x: list[float]) -> list[list[float]]:
        acts = [x]
        last = len(self.weights) - 1
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            prev = acts[-1]
            z = [sum(w * a for w, a in zip(row, prev)) + bj for row, bj in zip(W, b)]
            if l == last:
                acts.append([1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v)))) for v in z])
            else:
                acts.append([math.tanh(v) for v in z])
        return acts

    def predict(self, x: list[float]) -> float:
        return self._forward_all(x)[-1][0]

    def accuracy(self, data: list[tuple[list[float], int]]) -> float:
        ok = sum((self.predict(x) > 0.5) == bool(y) for x, y in data)
        return ok / len(data)

    # ---- 訓練(逐筆 SGD + 動量)----
    def train(self, data: list[tuple[list[float], int]], epochs: int, lr: float, rng: random.Random) -> None:
        momentum = 0.9
        vW = [[[0.0] * len(r) for r in W] for W in self.weights]
        vb = [[0.0] * len(b) for b in self.biases]
        data = list(data)
        for _ in range(epochs):
            rng.shuffle(data)
            for x, y in data:
                acts = self._forward_all(x)
                delta = [acts[-1][0] - y]  # sigmoid + 交叉熵的梯度
                for l in range(len(self.weights) - 1, -1, -1):
                    W, prev = self.weights[l], acts[l]
                    if l > 0:
                        back = [sum(W[j][i] * delta[j] for j in range(len(W))) * (1 - prev[i] ** 2)
                                for i in range(len(prev))]
                    for j, d in enumerate(delta):
                        vb[l][j] = momentum * vb[l][j] - lr * d
                        self.biases[l][j] += vb[l][j]
                        row, vrow = W[j], vW[l][j]
                        for i, a in enumerate(prev):
                            vrow[i] = momentum * vrow[i] - lr * d * a
                            row[i] += vrow[i]
                    if l > 0:
                        delta = back

    # ---- 權重操作 ----
    def num_params(self) -> int:
        return sum(len(r) for W in self.weights for r in W) + sum(len(b) for b in self.biases)

    def copy(self) -> "MLP":
        return MLP(list(self.sizes), [[list(r) for r in W] for W in self.weights],
                   [list(b) for b in self.biases])

    def permute_hidden(self, rng: random.Random) -> "MLP":
        """隨機重排每個隱藏層的單元。函數完全不變,但權重文字長得不一樣。"""
        m = self.copy()
        for l in range(len(m.sizes) - 2):  # 隱藏層 l 的輸出 = 第 l 層權重的列
            perm = list(range(m.sizes[l + 1]))
            rng.shuffle(perm)
            m.weights[l] = [m.weights[l][p] for p in perm]
            m.biases[l] = [m.biases[l][p] for p in perm]
            m.weights[l + 1] = [[row[p] for p in perm] for row in m.weights[l + 1]]
        return m

    def shuffle_weights(self, rng: random.Random) -> "MLP":
        """把所有權重值整批打亂位置:數值分佈不變,但函數被破壞。"""
        m = self.copy()
        flat = [w for W in m.weights for r in W for w in r]
        rng.shuffle(flat)
        it = iter(flat)
        m.weights = [[[next(it) for _ in r] for r in W] for W in m.weights]
        return m

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {"sizes": self.sizes, "weights": self.weights, "biases": self.biases}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        return cls(d["sizes"], d["weights"], d["biases"])

    def to_text(self, digits: int = 3) -> str:
        """給 LLM 讀的純文字格式。"""
        f = lambda v: f"{v:.{digits}f}"
        lines = [f"MLP,輸入 {self.sizes[0]} 維,隱藏層 tanh,輸出 1 個 sigmoid;層寬 {self.sizes}"]
        for l, (W, b) in enumerate(zip(self.weights, self.biases), 1):
            lines.append(f"[層 {l}] 權重矩陣 W{l}(每列 = 一個輸出單元,共 {len(W)} 列 x {len(W[0])} 欄):")
            lines += ["  " + " ".join(f(v) for v in row) for row in W]
            lines.append(f"[層 {l}] 偏置 b{l}: " + " ".join(f(v) for v in b))
        return "\n".join(lines)
