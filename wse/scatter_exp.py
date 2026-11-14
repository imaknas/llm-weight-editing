"""分散式缺陷的可修性:錯誤總面積固定,拆成 k 塊,看修復效果怎麼隨 k 變化。

每個缺陷網路比較這幾種修法(準確率一律對正確規則算):
  no_edit      不改
  llm          LLM 看權重(可先查 budget 個點的輸出)後給稀疏修改
  random_edit  與 LLM 同樣數量的隨機修改,多次抽樣取平均
  sparse_ft    用正確資料微調後,只保留變動最大的 n 個權重(n = LLM 改的數量)
  dense_ft     用正確資料微調全部權重(上限)
另外記 min_sparse:sparse_ft 要保留幾個權重才修得好。它不依賴 LLM,用來描述「這個缺陷在權重裡
散得多開」。它取的是微調後變動最大的前 n 個,不是最佳的 n 個,所以是偏高的估計。

用法(需要 torch,用專案的 .venv):
  python -m wse.scatter_exp build   [--ks 1 4 9 16] [--seeds 2]
  python -m wse.scatter_exp baselines
  python -m wse.scatter_exp run --model <registry 名稱> [--budgets 0 400] [--only-ks 1 16] [--only-seeds 0]
  python -m wse.scatter_exp report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .defects import PatchDefect, Trainer, evaluate, make_defective, scattered_patches
from .edit_exp import apply_edits, llm_fix, random_edits
from .llm import LLM, BrunnaCompleteLLM, RetryingLLM
from .mlp import MLP
from .tasks import TASKS, Task

SIZES = [2, 64, 64, 1]
Slot = tuple[int, int, int]  # (layer 從 0 起, row, col;col=-1 是偏置)


def intro(net: MLP, task: Task) -> str:
    return ("下面是一個小型神經網路(輸入平面上的點 (x,y),x、y 在 [-1,1];輸出 > 0.5 為類別 1)。"
            f"它本應實作規則「{task.description}」,但在輸入空間的一部分行為錯誤"
            "(錯在哪裡、有幾處都不告知)。\n\n" + net.to_text())


# ---- 不靠 LLM 的修法 ----
def changed_slots(before: MLP, after: MLP) -> list[tuple[float, Slot, float]]:
    """兩個網路之間每個參數的 (|變動|, 位置, 新值),變動大的在前。"""
    out = []
    for l, (W0, W1) in enumerate(zip(before.weights, after.weights)):
        for r, (row0, row1) in enumerate(zip(W0, W1)):
            out += [(abs(b - a), (l, r, c), b) for c, (a, b) in enumerate(zip(row0, row1))]
        out += [(abs(b - a), (l, r, -1), b) for r, (a, b) in enumerate(zip(before.biases[l], after.biases[l]))]
    return sorted(out, key=lambda t: -t[0])


def keep_top(before: MLP, ranked: list[tuple[float, Slot, float]], n: int) -> MLP:
    edits = [{"layer": l + 1, "row": r, "col": c, "value": v} for _, (l, r, c), v in ranked[:n]]
    return apply_edits(before, edits)[0]


def min_sparse(before: MLP, ranked: list, score: Callable[[MLP], dict], target: float = 0.9) -> Optional[int]:
    """sparse_ft 至少要保留幾個權重,缺陷區與其餘區域才都達到 target;以 2 的次方掃。"""
    n = 1
    while n <= len(ranked):
        s = score(keep_top(before, ranked, n))
        if s["defect_region"] >= target and s["outside"] >= target:
            return n
        n *= 2
    return None


# ---- 案例 ----
def build(ks: list[int], seeds: int, rng: random.Random, task: Task, train: Trainer, finetune: Trainer,
          sizes: list[int] = SIZES) -> list[dict]:
    cases = []
    for k in ks:
        for s in range(seeds):
            defect = scattered_patches(rng, k)
            net = make_defective(task, defect, sizes, rng, train, finetune, ft_n_train=4000)
            cases.append({"k": k, "seed": s, "task": task.name, "cells": list(defect.cells),
                          "side": defect.side, "net": net.to_dict()})
    return cases


def load_case(c: dict) -> tuple[MLP, PatchDefect, Task]:
    return (MLP.from_dict(c["net"]), PatchDefect(tuple(tuple(p) for p in c["cells"]), c["side"]),
            TASKS[c["task"]])


def scorer(case: dict, n_eval: int = 2000) -> Callable[[MLP], dict]:
    """同一個案例的所有修法都用同一批評估點。"""
    _, defect, task = load_case(case)
    seed = hash((case["k"], case["seed"])) & 0xFFFF

    def score(net: MLP) -> dict:
        return evaluate(net, task, defect, random.Random(seed), n_eval)
    return score


def baseline_rows(case: dict, repair: Trainer, rng: random.Random) -> tuple[list[dict], list]:
    net, _, task = load_case(case)
    score = scorer(case)
    fixed = net.copy()
    repair(fixed, task.sample(rng, 2000), rng)
    ranked = changed_slots(net, fixed)
    key = {"k": case["k"], "seed": case["seed"], "budget": 0}
    rows = [{**key, "arm": "no_edit", "changed": 0, **score(net)},
            {**key, "arm": "dense_ft", "changed": net.num_params(), **score(fixed),
             "min_sparse": min_sparse(net, ranked, score)}]
    return rows, ranked


def llm_rows(case: dict, llm: LLM, budget: int, ranked: list, rng: random.Random, draws: int = 20) -> list[dict]:
    net, _, task = load_case(case)
    score = scorer(case)
    edits, used, reason = llm_fix(llm, net, budget, intro(net, task))
    fixed, changed = apply_edits(net, edits)
    key = {"k": case["k"], "seed": case["seed"], "budget": budget}
    n = max(changed, 1)
    rand = [score(apply_edits(net, random_edits(net, n, rng))[0]) for _ in range(draws)]
    mean = lambda f: sum(r[f] for r in rand) / len(rand)
    biggest = max((abs(float(e.get("value", 0))) for e in edits if isinstance(e, dict)), default=0.0)
    return [{**key, "arm": "llm", "queries_used": used, "changed": changed, "max_abs_value": biggest,
             **score(fixed), "edits": edits, "reason": reason},
            {**key, "arm": "random_edit", "changed": n,
             **{f: mean(f) for f in ("overall", "defect_region", "outside")}},
            {**key, "arm": "sparse_ft", "changed": n, **score(keep_top(net, ranked, n))}]


def summarize(rows: list[dict]) -> list[str]:
    lines = []
    for k in sorted({r["k"] for r in rows}):
        for arm, b in sorted({(r["arm"], r["budget"]) for r in rows if r["k"] == k}):
            rs = [r for r in rows if (r["k"], r["arm"], r["budget"]) == (k, arm, b)]
            mean = lambda f: sum(r[f] for r in rs) / len(rs)
            extra = ""
            if arm == "dense_ft":
                extra = " min_sparse=" + ",".join(str(r.get("min_sparse")) for r in rs)
            lines.append(f"k={k:<3d} {arm:11s} budget={b:<4d} n={len(rs)} 全域={mean('overall'):.2f} "
                         f"缺陷區={mean('defect_region'):.2f} 其餘={mean('outside'):.2f} "
                         f"改動數={mean('changed'):.0f}{extra}")
    return lines


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="wse.scatter_exp")
    ap.add_argument("--nets", default="data/scatter_nets.json")
    ap.add_argument("--out", default="data/scatter_results.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ks", type=int, nargs="+", default=[1, 4, 9, 16])
    b.add_argument("--seeds", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--task", default="diagonal", choices=list(TASKS))
    sub.add_parser("baselines")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--budgets", type=int, nargs="+", default=[0, 400])
    r.add_argument("--only-ks", type=int, nargs="*")
    r.add_argument("--only-seeds", type=int, nargs="*")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    nets, out = Path(args.nets), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def append(rows: list[dict]) -> None:
        with out.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.cmd == "report":
        print("\n".join(summarize(read_jsonl(out))))
        return 0

    # 訓練後端只在這裡決定;核心邏輯只認 Trainer 這個形狀
    from .torch_train import adam_trainer
    if args.cmd == "build":
        cases = build(args.ks, args.seeds, random.Random(args.seed), TASKS[args.task],
                      adam_trainer(60), adam_trainer(400))
        nets.write_text(json.dumps(cases))
        for c in cases:
            print(c["k"], c["seed"], {f: round(v, 2) for f, v in scorer(c)(load_case(c)[0]).items()})
        return 0

    cases = json.loads(nets.read_text())
    done = read_jsonl(out)
    have = {(r["k"], r["seed"], r["arm"], r["budget"]) for r in done}
    repair = adam_trainer(60)
    llm = cost = None
    if args.cmd == "run":
        inner = BrunnaCompleteLLM(args.model)
        llm = RetryingLLM(inner, log=lambda m: print(m, file=sys.stderr), sleep=time.sleep)
    for c in cases:
        # 基準與排名只由案例決定(固定種子),續跑時重算出來的會相同
        rows, ranked = baseline_rows(c, repair, random.Random(1000 + 31 * c["k"] + c["seed"]))
        append([row for row in rows if (row["k"], row["seed"], row["arm"], row["budget"]) not in have])
        if args.cmd != "run":
            continue
        if (args.only_ks and c["k"] not in args.only_ks) or \
                (args.only_seeds is not None and args.only_seeds and c["seed"] not in args.only_seeds):
            continue
        for budget in args.budgets:
            if (c["k"], c["seed"], "llm", budget) in have:
                continue
            before = (inner.calls, inner.total_cost_usd)
            rows = llm_rows(c, llm, budget, ranked, random.Random(2000 + 31 * c["k"] + c["seed"] + budget))
            rows[0]["llm_calls"] = inner.calls - before[0]
            rows[0]["cost_usd"] = round(inner.total_cost_usd - before[1], 4)
            append(rows)
            print(f"k={c['k']} seed={c['seed']} budget={budget} 完成,花費 US${rows[0]['cost_usd']}", flush=True)
    print("\n".join(summarize(read_jsonl(out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
