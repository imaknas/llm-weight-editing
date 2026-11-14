"""「讀」實驗:給 LLM 一個網路的權重文字,問它這個網路學的是哪個任務。

對照組用來分辨「真的讀懂權重」與「靠 prompt 脈絡猜」:
  trained   訓練好的網路                         期望答案 = 真正的任務
  permuted  隱藏單元重排(函數完全相同)         期望答案 = 真正的任務
  shuffled  權重值整批打亂(函數被破壞)         期望答案 = none
  random    隨機初始化、沒訓練(偏置也隨機)     期望答案 = none
  randlabel 用同樣流程訓練在隨機標籤上           期望答案 = none

none 類的對照必須在表面上不留破綻:偏置全為 0、或權重幅度明顯偏小,都會讓 LLM 不必讀懂
函數就認得出來。randlabel 的權重幅度與訓練過的網路相近,是其中最嚴格的一組。

用法:
  python -m wse.read_exp build [--out data/nets.json]       訓練並存檔(純 Python,較慢)
  python -m wse.read_exp build-controls [--out data/nets_controls.json]   只產生 none 類對照
  python -m wse.read_exp prompt --nets data/nets.json       只印出第一個案例的 prompt
  python -m wse.read_exp run --nets data/nets.json --model <registry 名稱>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .llm import LLM, BrunnaCompleteLLM, RetryingLLM
from .mlp import MLP
from .tasks import TASKS

SIZES = [2, 12, 12, 1]
KINDS = ("trained", "permuted", "shuffled", "random", "randlabel")
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "enum": list(TASKS) + ["none"]},
        "reason": {"type": "string"},
    },
    "required": ["task", "reason"],
}


@dataclass
class Case:
    kind: str
    task: str          # 網路真正訓練的任務(random 對照為 "none")
    expected: str      # 期望 LLM 的答案
    net: MLP
    own_accuracy: float  # 此網路在其訓練任務上的實際準確率(確認訓練有效)


def build_cases(rng: random.Random, seeds_per_task: int = 2, epochs: int = 120, lr: float = 0.02,
                n_train: int = 200, n_eval: int = 400, sizes: list[int] = SIZES) -> list[Case]:
    cases: list[Case] = []
    for task in TASKS.values():
        for _ in range(seeds_per_task):
            net = MLP.random_init(sizes, rng)
            net.train(task.sample(rng, n_train), epochs, lr, rng)
            acc = net.accuracy(task.sample(rng, n_eval))
            cases.append(Case("trained", task.name, task.name, net, acc))
            cases.append(Case("permuted", task.name, task.name, net.permute_hidden(rng), acc))
            cases.append(Case("shuffled", task.name, "none", net.shuffle_weights(rng), acc))
    return cases + build_none_controls(rng, len(TASKS), epochs, lr, n_train, sizes)


def untrained_net(sizes: list[int], rng: random.Random) -> MLP:
    """沒訓練的網路,偏置與權重同尺度隨機(不留「偏置全為 0」這個一眼可辨的破綻)。"""
    net = MLP.random_init(sizes, rng)
    net.biases = [[rng.uniform(-1, 1) * 2 / math.sqrt(n_in) for _ in b]
                  for n_in, b in zip(sizes, net.biases)]
    return net


def random_label_net(sizes: list[int], rng: random.Random, epochs: int, lr: float, n_train: int) -> MLP:
    """用與正式任務相同的流程訓練,但標籤是擲硬幣:權重看起來像訓練過,卻不對應任何規則。"""
    data = [([rng.uniform(-1, 1), rng.uniform(-1, 1)], rng.randint(0, 1)) for _ in range(n_train)]
    net = MLP.random_init(sizes, rng)
    net.train(data, epochs, lr, rng)
    return net


def build_none_controls(rng: random.Random, n_each: int, epochs: int = 120, lr: float = 0.02,
                        n_train: int = 200, sizes: list[int] = SIZES) -> list[Case]:
    cases = [Case("random", "none", "none", untrained_net(sizes, rng), 0.0) for _ in range(n_each)]
    cases += [Case("randlabel", "none", "none", random_label_net(sizes, rng, epochs, lr, n_train), 0.0)
              for _ in range(n_each)]
    return cases


def build_prompt(net: MLP) -> str:
    task_lines = "\n".join(f"- {t.name}:{t.description}" for t in TASKS.values())
    return (
        "下面是一個小型神經網路的全部權重(沒有其他資訊)。它的輸入是平面上的點 (x, y),x、y 在 [-1,1];"
        "輸出 > 0.5 代表類別 1。\n"
        "它可能是針對下列四種規則之一訓練出來的:\n"
        f"{task_lines}\n"
        "也可能是沒有訓練過、或權重已被破壞的網路,這種情況請回答 none。\n"
        "請只根據權重判斷,task 欄位填 circle / xor / stripe / diagonal / none 其中之一,"
        "reason 簡述你從哪些權重得到結論。\n\n"
        + net.to_text()
    )


def run_cases(llm: LLM, cases: list[Case], done: Optional[list[dict]] = None,
              on_row: Callable[[dict], None] = lambda row: None) -> list[dict]:
    """逐案呼叫 LLM。done 是先前已完成的列(依 i 跳過,可續跑);每完成一列就呼叫 on_row(逐筆存檔用)。"""
    rows = list(done or [])
    finished = {r["i"] for r in rows}
    for i, c in enumerate(cases):
        if i in finished:
            continue
        ans = llm.complete(build_prompt(c.net), ANSWER_SCHEMA)
        row = {"i": i, "kind": c.kind, "task": c.task, "expected": c.expected,
               "answer": ans.get("task"), "correct": ans.get("task") == c.expected,
               "reason": ans.get("reason", ""), "own_accuracy": c.own_accuracy}
        rows.append(row)
        on_row(row)
    return sorted(rows, key=lambda r: r["i"])


def summarize(rows: list[dict]) -> dict[str, str]:
    out = {}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            out[kind] = f"{sum(r['correct'] for r in rs)}/{len(rs)}"
    return out


# ---- 存取 ----
def save_cases(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"kind": c.kind, "task": c.task, "expected": c.expected,
                                 "own_accuracy": c.own_accuracy, "net": c.net.to_dict()} for c in cases]))


def load_cases(path: Path) -> list[Case]:
    return [Case(d["kind"], d["task"], d["expected"], MLP.from_dict(d["net"]), d["own_accuracy"])
            for d in json.loads(path.read_text())]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="wse.read_exp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data/nets.json")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--seeds-per-task", type=int, default=2)
    c = sub.add_parser("build-controls")
    c.add_argument("--out", default="data/nets_controls.json")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--n-each", type=int, default=4)
    p = sub.add_parser("prompt")
    p.add_argument("--nets", default="data/nets.json")
    r = sub.add_parser("run")
    r.add_argument("--nets", default="data/nets.json")
    r.add_argument("--model", required=True)
    r.add_argument("--out", default="data/read_results.jsonl")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        cases = build_cases(random.Random(args.seed), args.seeds_per_task)
        save_cases(cases, Path(args.out))
        for c in cases:
            if c.kind == "trained":
                print(f"{c.task:9s} acc={c.own_accuracy:.2f}")
        return 0
    if args.cmd == "build-controls":
        save_cases(build_none_controls(random.Random(args.seed), args.n_each), Path(args.out))
        return 0
    cases = load_cases(Path(args.nets))
    if args.cmd == "prompt":
        print(build_prompt(cases[0].net))
        return 0
    # 唯一組裝具體 LLM 的地方;結果逐筆附加到 jsonl,中斷後重跑會從已完成處續跑
    llm = RetryingLLM(BrunnaCompleteLLM(args.model), log=lambda m: print(m, file=sys.stderr),
                      sleep=time.sleep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []

    def append(row: dict) -> None:
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = run_cases(llm, cases, done, append)
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
