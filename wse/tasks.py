"""2D 二元分類任務:輸入落在 [-1,1]^2,標籤由規則決定。"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    label: Callable[[float, float], int]

    def sample(self, rng: random.Random, n: int) -> list[tuple[list[float], int]]:
        pts = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
        return [([x, y], self.label(x, y)) for x, y in pts]


TASKS: dict[str, Task] = {t.name: t for t in [
    Task("circle", "圓:x^2 + y^2 < 0.5 時為 1", lambda x, y: int(x * x + y * y < 0.5)),
    Task("xor", "XOR:x 與 y 同號時為 1(四象限棋盤)", lambda x, y: int(x * y > 0)),
    Task("stripe", "垂直條帶:|x| < 0.4 時為 1(與 y 無關)", lambda x, y: int(abs(x) < 0.4)),
    Task("diagonal", "對角線:x + y > 0 時為 1", lambda x, y: int(x + y > 0)),
]}
