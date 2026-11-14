"""「改」實驗:一個本應實作 diagonal(x+y>0)的網路,在某個象限行為錯誤;讓 LLM 修。

缺陷:象限 x>0 且 y>0 的標籤被翻轉後微調,所以網路在該象限輸出 0(應為 1)。LLM 不被告知缺陷位置。
協定(單次呼叫的 LLM 介面下做成兩輪):
  第 1 輪(budget>0 時):LLM 看權重,最多提出 budget 個查詢點;我們回傳網路在這些點的輸出(= forward pass 工具)。
  第 2 輪:LLM 輸出稀疏修改 edits=[{layer,row,col,value}](col=-1 表示偏置;層從 1 起算)。
對照:no_edit(不改)、random_edit(同樣數量的隨機位置、隨機值)。
指標:全域準確率、缺陷象限準確率、其餘區域準確率(皆對真實規則 x+y>0),以及被改動的權重比例。

用法:
  python -m wse.edit_exp --model weight-reader --budgets 0 10 --seeds 3
"""
from __future__ import annotations

import argparse
import json
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
TRUE_TASK = TASKS["diagonal"]
in_defect_region = lambda x, y: x > 0 and y > 0

QUERY_SCHEMA = {"type": "object", "properties": {
    "queries": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    "reason": {"type": "string"}}, "required": ["queries"]}
EDIT_SCHEMA = {"type": "object", "properties": {
    "edits": {"type": "array", "items": {"type": "object", "properties": {
        "layer": {"type": "integer"}, "row": {"type": "integer"},
        "col": {"type": "integer"}, "value": {"type": "number"}},
        "required": ["layer", "row", "col", "value"]}},
    "reason": {"type": "string"}}, "required": ["edits"]}


@dataclass
class Defective:
    net: MLP
    fixed_acc: dict  # 缺陷網路本身的指標,用來確認缺陷有效


def make_defective(rng: random.Random, n_train: int = 200, epochs: int = 120, ft_epochs: int = 60) -> MLP:
    net = MLP.random_init(SIZES, rng)
    net.train(TRUE_TASK.sample(rng, n_train), epochs, 0.02, rng)
    flipped = [(x, 1 - y if in_defect_region(*x) else y) for x, y in TRUE_TASK.sample(rng, n_train)]
    net.train(flipped, ft_epochs, 0.02, rng)
    return net


def evaluate(net: MLP, rng: random.Random, n: int = 600) -> dict:
    data = TRUE_TASK.sample(rng, n)
    reg = [d for d in data if in_defect_region(*d[0])]
    out = [d for d in data if not in_defect_region(*d[0])]
    return {"overall": net.accuracy(data), "defect_region": net.accuracy(reg), "outside": net.accuracy(out)}


def apply_edits(net: MLP, edits: list[dict]) -> tuple[MLP, int]:
    """套用稀疏修改;越界的修改忽略。回傳 (新網路, 實際被改的權重數)。"""
    m, changed = net.copy(), 0
    for e in edits:
        try:
            l, r, c, v = int(e["layer"]) - 1, int(e["row"]), int(e["col"]), float(e["value"])
            if c == -1:
                m.biases[l][r] = v
            else:
                m.weights[l][r][c] = v
            changed += 1
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return m, changed


def random_edits(net: MLP, k: int, rng: random.Random) -> list[dict]:
    out = []
    for _ in range(k):
        l = rng.randrange(len(net.weights))
        r = rng.randrange(len(net.weights[l]))
        c = rng.randrange(-1, len(net.weights[l][r]))
        out.append({"layer": l + 1, "row": r, "col": c, "value": rng.uniform(-2, 2)})
    return out


def _intro(net: MLP) -> str:
    return ("下面是一個小型神經網路(輸入平面上的點 (x,y),x、y 在 [-1,1];輸出 > 0.5 為類別 1)。"
            "它本應實作規則「x + y > 0 時為 1」,但在輸入空間的某個區域行為錯誤(位置不告知)。\n\n"
            + net.to_text())


def llm_fix(llm: LLM, net: MLP, budget: int, intro: Optional[str] = None) -> tuple[list[dict], int, str]:
    """回傳 (edits, 實際使用的 forward pass 數, LLM 自述的理由)。intro 是任務說明加權重文字,
    預設用本實驗(diagonal、單一區域)的說法。"""
    intro = _intro(net) if intro is None else intro
    context, used = "", 0
    if budget > 0:
        q = llm.complete(intro + f"\n\n你可以先查詢網路在最多 {budget} 個點的輸出。"
                         "請在 queries 列出這些點,格式 [[x,y],...]。", QUERY_SCHEMA)
        pts = [p for p in q.get("queries", []) if isinstance(p, list) and len(p) == 2][:budget]
        used = len(pts)
        results = [f"({p[0]:.3f}, {p[1]:.3f}) -> {net.predict([float(p[0]), float(p[1])]):.3f}" for p in pts]
        context = "\n\n查詢結果:\n" + "\n".join(results)
    ans = llm.complete(
        intro + context +
        "\n\n請用盡量少的權重修改修好它。edits 每項為 {layer, row, col, value}:layer 從 1 起算,"
        "row 是該層輸出單元編號(從 0 起),col 是輸入編號(從 0 起),col=-1 代表偏置 b。"
        "value 是新值。reason 簡述你判斷的缺陷位置與修法。", EDIT_SCHEMA)
    return ans.get("edits", []), used, ans.get("reason", "")


def run(llm: LLM, budgets: list[int], seeds: int, rng: random.Random,
        done: Optional[list[dict]] = None, on_row: Callable[[dict], None] = lambda row: None) -> list[dict]:
    """done 是先前已完成的列:同一個 (seed, budget) 的 LLM 修改直接沿用,不重新呼叫(可續跑)。
    網路與評估資料只由 rng 決定,與 LLM 的回答無關,所以續跑時重建出的網路相同。
    每完成一列就呼叫 on_row(逐筆存檔用)。"""
    prior = {(r["seed"], r["budget"]): r for r in (done or []) if r["arm"] == "llm"}
    seen_seeds = {r["seed"] for r in (done or []) if r["arm"] == "no_edit"}
    rows = []

    def emit(row: dict, fresh: bool) -> None:
        rows.append(row)
        if fresh:
            on_row(row)

    for s in range(seeds):
        net = make_defective(rng)
        ev_rng_state = rng.getstate()
        base = evaluate(net, rng)
        n_params = net.num_params()
        emit({"seed": s, "arm": "no_edit", "budget": 0, "changed": 0, "frac": 0.0, **base}, s not in seen_seeds)
        for budget in budgets:
            old = prior.get((s, budget))
            if old:
                edits, used, reason = old["edits"], old["queries_used"], old.get("reason", "")
            else:
                edits, used, reason = llm_fix(llm, net, budget)
            fixed, changed = apply_edits(net, edits)
            rng.setstate(ev_rng_state)
            emit({"seed": s, "arm": "llm", "budget": budget, "queries_used": used, "changed": changed,
                  "frac": changed / n_params, **evaluate(fixed, rng), "edits": edits, "reason": reason}, not old)
            rnd, rc = apply_edits(net, random_edits(net, max(changed, 1), rng))
            rng.setstate(ev_rng_state)
            emit({"seed": s, "arm": "random_edit", "budget": budget, "changed": rc,
                  "frac": rc / n_params, **evaluate(rnd, rng)}, not old)
    return rows


def summarize(rows: list[dict]) -> list[str]:
    keys = sorted({(r["arm"], r["budget"]) for r in rows})
    lines = []
    for arm, b in keys:
        rs = [r for r in rows if r["arm"] == arm and r["budget"] == b]
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        lines.append(f"{arm:12s} budget={b:<3d} n={len(rs)} 全域={mean('overall'):.2f} "
                     f"缺陷象限={mean('defect_region'):.2f} 其餘={mean('outside'):.2f} 改動比例={mean('frac'):.3f}")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="wse.edit_exp")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 10])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/edit_results.jsonl")
    args = ap.parse_args(argv)
    # 唯一組裝具體 LLM 的地方;結果逐筆附加,中斷後重跑會沿用已完成的 LLM 修改
    llm = RetryingLLM(BrunnaCompleteLLM(args.model), log=lambda m: print(m, file=sys.stderr), sleep=time.sleep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []

    def append(row: dict) -> None:
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = run(llm, args.budgets, args.seeds, random.Random(args.seed), done, append)
    print("\n".join(summarize(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
