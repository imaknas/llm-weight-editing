"""缺陷:輸入空間中「網路應該答對、卻被訓練成答錯」的區域。

分散程度是可調的:錯誤總面積固定,拆成 k 個不相鄰的小方塊。k=1 是一整塊(局部缺陷),
k 越大越零散。這樣不同 k 之間能比的是「同樣多的錯,散得多開」,而不是錯得多不多。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .mlp import MLP
from .tasks import Task

Point = tuple[float, float]


@dataclass(frozen=True)
class PatchDefect:
    """k 個邊長 side 的方塊;cells 是各方塊左下角座標。"""
    cells: tuple[Point, ...]
    side: float

    @property
    def n_patches(self) -> int:
        return len(self.cells)

    def contains(self, x: float, y: float) -> bool:
        return any(cx <= x < cx + self.side and cy <= y < cy + self.side for cx, cy in self.cells)

    def area_fraction(self) -> float:
        return self.n_patches * self.side ** 2 / 4.0


def scattered_patches(rng: random.Random, k: int, area_fraction: float = 0.25, tries: int = 200) -> PatchDefect:
    """把佔輸入空間 area_fraction 的錯誤區拆成 k 個方塊,放在格子上且彼此不共邊。

    k=1 時格子是 2x2,也就是隨機一個象限。只禁止共邊(對角相碰可以):若連對角都禁止,
    25% 的面積剛好只剩規則的點陣排法,位置就不隨機了。
    """
    side = math.sqrt(4.0 * area_fraction / k)
    n = int(round(2.0 / side))
    if abs(n * side - 2.0) > 1e-9:
        raise ValueError(f"k={k}, area_fraction={area_fraction} 排不進整數格子")
    for _ in range(tries):
        free = [(i, j) for i in range(n) for j in range(n)]
        rng.shuffle(free)
        chosen: list[tuple[int, int]] = []
        for i, j in free:
            if all(abs(i - a) + abs(j - b) != 1 for a, b in chosen):
                chosen.append((i, j))
                if len(chosen) == k:
                    return PatchDefect(tuple((-1 + i * side, -1 + j * side) for i, j in sorted(chosen)), side)
    raise ValueError(f"k={k} 個不共邊方塊放不進 {n}x{n} 格子")


def flipped_sample(task: Task, defect: PatchDefect, rng: random.Random, n: int) -> list[tuple[list[float], int]]:
    """任務資料,但缺陷區內的標籤被翻轉。"""
    return [(x, 1 - y if defect.contains(*x) else y) for x, y in task.sample(rng, n)]


Data = list[tuple[list[float], int]]
# 訓練後端:就地更新 net。純 Python 的 MLP.train 與 torch 版都符合這個形狀,由呼叫端決定用哪個。
Trainer = Callable[[MLP, Data, random.Random], None]


def make_defective(task: Task, defect: PatchDefect, sizes: list[int], rng: random.Random,
                   train: Trainer, finetune: Trainer, n_train: int = 2000, ft_n_train: int = 2000) -> MLP:
    """先照正確規則訓練,再用翻轉過的標籤微調,得到只在缺陷區答錯的網路。"""
    net = MLP.random_init(sizes, rng)
    train(net, task.sample(rng, n_train), rng)
    finetune(net, flipped_sample(task, defect, rng, ft_n_train), rng)
    return net


def evaluate(net: MLP, task: Task, defect: PatchDefect, rng: random.Random, n: int = 2000) -> dict:
    """對正確規則的準確率:全域、缺陷區內、缺陷區外。"""
    data = task.sample(rng, n)
    inside = [d for d in data if defect.contains(*d[0])]
    outside = [d for d in data if not defect.contains(*d[0])]
    return {"overall": net.accuracy(data), "defect_region": net.accuracy(inside),
            "outside": net.accuracy(outside)}
