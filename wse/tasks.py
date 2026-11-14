"""2D 二元分類任務:輸入落在 [-1,1]^2,標籤由規則決定。"""
from __future__ import annotations

import math
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


# ---- 沒有公式可抄的目標函數 ----
# 幾個隨機正弦波相加後取正負號,邊界是彎的。用在「正確答案只能靠查詢得知」的實驗:
# 目標若能用一句規則描述,修的人可以乾脆重寫一個,而不是修原本的網路。
Wave = tuple[float, float, float, float]  # (振幅, x 頻率, y 頻率, 相位)


def wavy_task(waves: list[Wave], name: str = "wavy") -> Task:
    ws = [tuple(w) for w in waves]
    return Task(name, "(沒有公式;正確類別只能查詢)",
                lambda x, y: int(sum(a * math.sin(u * x + v * y + p) for a, u, v, p in ws) > 0))


def _best_line_accuracy(task: Task, n: int = 16) -> float:
    """最好的一條直線能把這個函數分對多少(粗略掃角度與截距)。"""
    pts = [(-1 + 2 * (i + 0.5) / n, -1 + 2 * (j + 0.5) / n) for i in range(n) for j in range(n)]
    labels = [task.label(x, y) for x, y in pts]
    best = 0.0
    for a in range(12):
        c, s = math.cos(math.pi * a / 12), math.sin(math.pi * a / 12)
        for t in range(-6, 7):
            hit = sum((c * x + s * y > t / 5) == bool(l) for (x, y), l in zip(pts, labels)) / len(pts)
            best = max(best, hit, 1 - hit)
    return best


def random_wavy_waves(rng: random.Random, n_waves: int = 4, tries: int = 1000) -> list[Wave]:
    """抽一組波:兩類大致各半,而且不能近似成一條直線。"""
    for _ in range(tries):
        waves = []
        for _ in range(n_waves):
            freq, angle = rng.uniform(3.0, 5.0), rng.uniform(0, 2 * math.pi)
            waves.append((rng.uniform(0.5, 1.0), freq * math.cos(angle), freq * math.sin(angle),
                          rng.uniform(0, 2 * math.pi)))
        task = wavy_task(waves)
        frac = sum(y for _, y in task.sample(rng, 500)) / 500
        if 0.4 <= frac <= 0.6 and _best_line_accuracy(task) <= 0.75:
            return waves
    raise ValueError("抽不到符合條件的目標函數")
