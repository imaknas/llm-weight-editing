"""分散式缺陷的可修性:錯誤總面積固定,拆成 k 塊,看修復效果怎麼隨 k 變化。

為了量的是「修」而不是「重寫」,有兩個限制:
  - 正確函數沒有公式(隨機波形,見 tasks.wavy_task),只能靠查詢得知某個點該是哪一類。
  - 修改後的值不得超出該層原有參數的最大絕對值;超出的會被截到上限並記數。
    (不設上限時,修的人可以用一個極大的權重蓋過整個網路。)

每個缺陷網路比較這幾種做法(準確率一律對正確函數算):
  no_edit      不改
  llm          LLM 看權重,先選 budget 個點查「網路輸出與正確類別」,再給稀疏修改
  random_edit  與 LLM 同樣數量、同樣上限的隨機修改,多次抽樣取平均
  query_ft     只用 LLM 查到的那些點做梯度微調(資訊量與 LLM 相同)
  dense_ft     用大量正確資料微調(上限)

用法(需要 torch,用專案的 .venv):
  python -m wse.scatter_exp build   [--ks 1 4 9 16] [--seeds 2]
  python -m wse.scatter_exp run --model <registry 名稱> [--budgets 100 400] [--only-ks 1 16] [--only-seeds 0]
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

from .defects import Data, PatchDefect, Trainer, evaluate, make_defective, scattered_patches
from .edit_exp import apply_edits
from .llm import LLM, BrunnaCompleteLLM, RetryingLLM
from .mlp import MLP
from .tasks import Task, random_wavy_waves, wavy_task

SIZES = [2, 64, 64, 1]
QUERY_SCHEMA = {"type": "object", "properties": {
    "queries": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    "reason": {"type": "string"}}, "required": ["queries"]}
EDIT_SCHEMA = {"type": "object", "properties": {
    "edits": {"type": "array", "items": {"type": "object", "properties": {
        "layer": {"type": "integer"}, "row": {"type": "integer"},
        "col": {"type": "integer"}, "value": {"type": "number"}},
        "required": ["layer", "row", "col", "value"]}},
    "reason": {"type": "string"}}, "required": ["edits"]}


# ---- 改動幅度上限 ----
def layer_bounds(net: MLP) -> list[float]:
    """每層允許的最大絕對值 = 該層原有權重與偏置的最大絕對值。"""
    return [max(max(abs(w) for row in W for w in row), max(abs(v) for v in b))
            for W, b in zip(net.weights, net.biases)]


def clip_edits(edits: list, bounds: list[float]) -> tuple[list[dict], int]:
    """把超出上限的值截到上限。回傳 (截過的修改, 被截的數量);格式不對的項目原樣留給 apply_edits 忽略。"""
    out, clipped = [], 0
    for e in edits:
        try:
            bound, v = bounds[int(e["layer"]) - 1], float(e["value"])
        except (KeyError, IndexError, ValueError, TypeError):
            out.append(e)
            continue
        if abs(v) > bound:
            clipped += 1
            v = bound if v > 0 else -bound
        out.append({**e, "value": v})
    return out, clipped


def random_edits(net: MLP, n: int, bounds: list[float], rng: random.Random) -> list[dict]:
    out = []
    for _ in range(n):
        l = rng.randrange(len(net.weights))
        r = rng.randrange(len(net.weights[l]))
        c = rng.randrange(-1, len(net.weights[l][r]))
        out.append({"layer": l + 1, "row": r, "col": c, "value": rng.uniform(-bounds[l], bounds[l])})
    return out


# ---- LLM 修復協定 ----
def intro(net: MLP) -> str:
    return ("下面是一個小型神經網路(輸入平面上的點 (x,y),x、y 在 [-1,1];輸出 > 0.5 為類別 1)。"
            "它本應與一個參考函數的分類一致。參考函數沒有公式,只能逐點查詢。"
            "這個網路在輸入空間的一部分與參考函數不一致(錯在哪裡、有幾處都不告知),其餘地方是對的。\n\n"
            + net.to_text())


def llm_repair(llm: LLM, net: MLP, task: Task, budget: int, bounds: list[float]) -> tuple[list, Data, str]:
    """回傳 (LLM 給的修改, 它查到的 (點, 正確類別), 它自述的理由)。"""
    q = llm.complete(intro(net) + f"\n\n你可以查詢最多 {budget} 個點;每個點會回傳網路的輸出與參考函數的正確類別。"
                     "請在 queries 列出這些點,格式 [[x,y],...]。", QUERY_SCHEMA)
    pts = [[float(p[0]), float(p[1])] for p in q.get("queries", [])
           if isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)][:budget]
    seen: Data = [(p, task.label(p[0], p[1])) for p in pts]
    results = "\n".join(f"({p[0]:.3f}, {p[1]:.3f}) -> 網路輸出 {net.predict(p):.3f},正確類別 {y}" for p, y in seen)
    limits = "、".join(f"第 {l} 層 ±{b:.3f}" for l, b in enumerate(bounds, 1))
    ans = llm.complete(
        intro(net) + "\n\n查詢結果:\n" + results +
        "\n\n請修改權重,讓網路在錯的地方變對,同時不要弄壞原本對的地方;修改數量盡量少。"
        "edits 每項為 {layer, row, col, value}:layer 從 1 起算,row 是該層輸出單元編號(從 0 起),"
        "col 是輸入編號(從 0 起),col=-1 代表偏置 b,value 是新值。"
        f"新值的絕對值不得超過該層的上限({limits}),超過的會被截到上限。"
        "reason 簡述你判斷錯在哪裡與怎麼修。", EDIT_SCHEMA)
    return ans.get("edits", []), seen, ans.get("reason", "")


# ---- 案例 ----
def build(ks: list[int], seeds: int, rng: random.Random, train: Trainer, finetune: Trainer,
          sizes: list[int] = SIZES, n_train: int = 4000) -> list[dict]:
    cases = []
    for k in ks:
        for s in range(seeds):
            waves = random_wavy_waves(rng)
            defect = scattered_patches(rng, k)
            net = make_defective(wavy_task(waves), defect, sizes, rng, train, finetune,
                                 n_train=n_train, ft_n_train=n_train)
            cases.append({"k": k, "seed": s, "waves": waves, "cells": list(defect.cells),
                          "side": defect.side, "net": net.to_dict()})
    return cases


def load_case(c: dict) -> tuple[MLP, PatchDefect, Task]:
    return (MLP.from_dict(c["net"]), PatchDefect(tuple(tuple(p) for p in c["cells"]), c["side"]),
            wavy_task(c["waves"]))


def scorer(case: dict, n_eval: int = 2000) -> Callable[[MLP], dict]:
    """同一個案例的所有做法都用同一批評估點。"""
    _, defect, task = load_case(case)
    seed = 7919 * case["k"] + case["seed"]

    def score(net: MLP) -> dict:
        return evaluate(net, task, defect, random.Random(seed), n_eval)
    return score


def baseline_rows(case: dict, repair: Trainer, rng: random.Random) -> list[dict]:
    net, _, task = load_case(case)
    score = scorer(case)
    fixed = net.copy()
    repair(fixed, task.sample(rng, 2000), rng)
    key = {"k": case["k"], "seed": case["seed"], "budget": 0}
    return [{**key, "arm": "no_edit", "changed": 0, **score(net)},
            {**key, "arm": "dense_ft", "changed": net.num_params(), **score(fixed)}]


def llm_rows(case: dict, llm: LLM, budget: int, repair: Trainer, rng: random.Random, draws: int = 20) -> list[dict]:
    net, defect, task = load_case(case)
    score, bounds = scorer(case), layer_bounds(net)
    raw, seen, reason = llm_repair(llm, net, task, budget, bounds)
    edits, clipped = clip_edits(raw, bounds)
    fixed, changed = apply_edits(net, edits)
    key = {"k": case["k"], "seed": case["seed"], "budget": budget}
    n = max(changed, 1)
    rand = [score(apply_edits(net, random_edits(net, n, bounds, rng))[0]) for _ in range(draws)]
    mean = lambda f: sum(r[f] for r in rand) / len(rand)
    rows = [{**key, "arm": "llm", "queries_used": len(seen), "changed": changed, "clipped": clipped,
             "queries_in_defect": sum(defect.contains(*p) for p, _ in seen),
             **score(fixed), "edits": raw, "reason": reason},
            {**key, "arm": "random_edit", "changed": n,
             **{f: mean(f) for f in ("overall", "defect_region", "outside")}}]
    if seen:
        tuned = net.copy()
        repair(tuned, seen, rng)
        rows.append({**key, "arm": "query_ft", "changed": net.num_params(), **score(tuned)})
    return rows


def summarize(rows: list[dict]) -> list[str]:
    """依 (k, 做法, 查詢預算, 由哪個模型改) 取平均。不經 LLM 的基準沒有 editor。"""
    lines = []
    for k in sorted({r["k"] for r in rows}):
        groups = sorted({(r["arm"], r["budget"], r.get("editor", "")) for r in rows if r["k"] == k})
        for arm, b, ed in groups:
            rs = [r for r in rows if (r["k"], r["arm"], r["budget"], r.get("editor", "")) == (k, arm, b, ed)]
            mean = lambda f: sum(r[f] for r in rs) / len(rs)
            lines.append(f"k={k:<3d} {arm:11s} {ed:22s} budget={b:<4d} n={len(rs)} 全域={mean('overall'):.2f} "
                         f"缺陷區={mean('defect_region'):.2f} 其餘={mean('outside'):.2f} "
                         f"改動數={mean('changed'):.0f}")
    return lines


FIRST_EDITOR = "weight-reader"


def labelled(rows: list[dict]) -> list[dict]:
    """由哪個模型改的記在每一列的 editor;這個欄位出現之前的結果都是 FIRST_EDITOR 跑的。"""
    for r in rows:
        if r["arm"] not in ("no_edit", "dense_ft"):
            r.setdefault("editor", FIRST_EDITOR)
    return rows


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
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--budgets", type=int, nargs="+", default=[100, 400])
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
        print("\n".join(summarize(labelled(read_jsonl(out)))))
        return 0

    # 訓練後端只在這裡決定;核心邏輯只認 Trainer 這個形狀
    from .torch_train import adam_trainer
    if args.cmd == "build":
        cases = build(args.ks, args.seeds, random.Random(args.seed), adam_trainer(150), adam_trainer(400))
        nets.write_text(json.dumps(cases))
        repair = adam_trainer(60)
        for c in cases:
            rows = baseline_rows(c, repair, random.Random(1000 + 31 * c["k"] + c["seed"]))
            append(rows)
            print(c["k"], c["seed"], {f: round(rows[0][f], 2) for f in ("overall", "defect_region", "outside")},
                  flush=True)
        return 0

    cases = json.loads(nets.read_text())
    rows_so_far = labelled(read_jsonl(out))
    have = {(r["k"], r["seed"], r["arm"], r["budget"], r.get("editor")) for r in rows_so_far}
    inner = BrunnaCompleteLLM(args.model)
    llm = RetryingLLM(inner, log=lambda m: print(m, file=sys.stderr), sleep=time.sleep)
    query_repair = adam_trainer(200)
    for c in cases:
        if (args.only_ks and c["k"] not in args.only_ks) or (args.only_seeds and c["seed"] not in args.only_seeds):
            continue
        for budget in args.budgets:
            if (c["k"], c["seed"], "llm", budget, args.model) in have:
                continue
            before = (inner.calls, inner.total_cost_usd)
            rows = llm_rows(c, llm, budget, query_repair, random.Random(2000 + 31 * c["k"] + c["seed"] + budget))
            for row in rows:
                row["editor"] = args.model
            rows[0]["llm_calls"] = inner.calls - before[0]
            rows[0]["reported_cost_usd"] = round(inner.total_cost_usd - before[1], 4)
            append(rows)
            print(f"k={c['k']} seed={c['seed']} budget={budget} 完成", flush=True)
    print("\n".join(summarize(labelled(read_jsonl(out)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
