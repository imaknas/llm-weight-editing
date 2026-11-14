"""獨立重算:Claude 讀 Qwen3.5-0.8B 的權重後,能否預測權重修改對輸出的影響。

資料來自公開專案 csinva/weight-space-programming-autoresearch(MIT 授權)的 experiments/systematic-tests/。
這份腳本只讀它的原始檔(各輪的 ground_truth.json 與 prediction.json),不匯入也不執行它的任何程式;
指標的定義照它的報告與 analyze.py / analyze_zeroshot.py 的文字描述重新實作:

  項目 = (修改, 句子);真值 = 修改後目標 token 的 log 機率變化 dlogp。
  spearman   預測與真值的等級相關
  mae        平均絕對誤差;zero 是「永遠預測 0」
  big_auc    以 |預測| 分辨 |真值| > 0.1 的項目
  sign       |真值| > 門檻的項目中,預測方向正確的比例
  標準誤     以「修改」為單位重抽(同一個修改的 8 個句子一起進出)

零樣本的集合(與報告相同):第 0 輪、第 2 輪的無範例重跑(rounds_fewshot/r2_k0)、第 3 輪以後;
只取原始提示(資料夾 none、raw);不含 token 類的修改。

用法:python analysis/singh_weight_reading.py <repo>/experiments/systematic-tests
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1]
FAMS = ["scalar", "neuron", "layer"]
NBOOT = 1000


def rank(a):
    """平均名次(同值取平均)。"""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(p, y):
    if len(p) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rank(p), rank(y))[0, 1])


def auc(pos, score):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any() or np.std(score) == 0:
        return np.nan
    r = rank(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2.0) / (n1 * n0))


def load(sources, conds):
    """sources: [(標籤, 真值檔, 預測所在資料夾)];回傳 {(iid, pid): row}。"""
    rows = {}
    for tag, gt_path, pred_dir in sources:
        gt = json.load(open(gt_path))
        for fam in FAMS + ["token"]:
            for cond in conds:
                p = os.path.join(pred_dir, fam, cond, "prediction.json")
                if not os.path.exists(p):
                    continue
                for k, v in json.load(open(p))["predictions"].items():
                    iid, pid = k.split("|")
                    t = gt[iid]["per_prompt"][pid]
                    row = rows.setdefault((tag, iid, pid), dict(src=tag, family=fam, iid=(tag, iid), true=t["dlogp"]))
                    row[cond] = v["dlogp"]
    return list(rows.values())


def stats(rows, a="raw", b="none"):
    sub = [r for r in rows if a in r and b in r]
    if len(sub) < 3:
        return None
    y = np.array([r["true"] for r in sub])
    out = {"n_items": len(sub), "n_edits": len({r["iid"] for r in sub})}
    for k in (b, a):
        p = np.array([r[k] for r in sub], float)
        out[k] = dict(spearman=spearman(p, y), mae=float(np.mean(np.abs(p - y))), big_auc=auc(np.abs(y) > 0.1, np.abs(p)))
    out["zero_mae"] = float(np.mean(np.abs(y)))
    out["diff_spearman"] = out[a]["spearman"] - out[b]["spearman"]
    return out


def boot_se(rows, fn, rng):
    by = {}
    for r in rows:
        by.setdefault(r["iid"], []).append(r)
    groups = list(by.values())
    vals = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(groups), len(groups))
        v = fn([x for i in pick for x in groups[i]])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return float(np.std(vals)) if vals else float("nan")


def sign_acc(rows, key, thr):
    sub = [r for r in rows if abs(r["true"]) > thr and "raw" in r and "none" in r]
    if not sub:
        return float("nan"), 0
    return float(np.mean([np.sign(r[key]) == np.sign(r["true"]) for r in sub])), len(sub)


def report(title, rows, fams):
    rng = np.random.default_rng(0)
    print(f"\n== {title}")
    print(f"{'family':8s} {'修改數':>6s} {'項目':>6s} | ρ 不讀  ρ 讀權重  差 ± SE      | MAE 不讀  讀   永遠0 | bigAUC 不讀 讀")
    for fam in fams + ["all"]:
        sub = [r for r in rows if (fam == "all" and r["family"] in fams) or r["family"] == fam]
        s = stats(sub)
        if not s:
            continue
        se = boot_se(sub, lambda rs: (stats(rs) or {}).get("diff_spearman", np.nan), rng)
        print(f"{fam:8s} {s['n_edits']:6d} {s['n_items']:6d} | {s['none']['spearman']:+.3f}  {s['raw']['spearman']:+.3f}   "
              f"{s['diff_spearman']:+.3f} ± {se:.3f} | {s['none']['mae']:.3f}  {s['raw']['mae']:.3f}  {s['zero_mae']:.3f} | "
              f"{s['none']['big_auc']:.2f}  {s['raw']['big_auc']:.2f}")


def main():
    gt = lambda d, r: os.path.join(ROOT, d, f"r{r}", "ground_truth.json")
    n_rounds = len(glob.glob(os.path.join(ROOT, "rounds", "r*", "ground_truth.json")))

    # Opus,零樣本,目標是第一名的 token
    src = [("r0", gt("rounds", 0), os.path.join(ROOT, "rounds", "r0")),
           ("r2", gt("rounds", 2), os.path.join(ROOT, "rounds_fewshot", "r2_k0"))]
    src += [(f"r{r}", gt("rounds", r), os.path.join(ROOT, "rounds", f"r{r}")) for r in range(3, n_rounds)]
    opus = load(src, ["none", "raw"])
    report("Opus 5.5,零樣本,目標 = 第一名的 token", opus, FAMS)
    for fam in FAMS + ["all"]:
        sub = [r for r in opus if (fam == "all" and r["family"] in FAMS) or r["family"] == fam]
        for thr in (0.1, 1.0):
            a, n = sign_acc(sub, "none", thr)
            b, _ = sign_acc(sub, "raw", thr)
            print(f"  方向正確率 {fam:7s} |真值|>{thr}: 不讀 {a:.2f}  讀權重 {b:.2f}  (n={n})")
    big = [r for r in opus if r["family"] == "scalar" and abs(r["true"]) > 1.0 and "raw" in r]
    print("  scalar 且 |真值|>1 的項目來自幾個修改:", len({r["iid"] for r in big}), "/ 項目數", len(big))

    # Opus,目標改為排名較後的 token(每一輪都是零樣本)
    n_alt = len(glob.glob(os.path.join(ROOT, "rounds_alt", "r*", "ground_truth.json")))
    alt = load([(f"r{r}", gt("rounds_alt", r), os.path.join(ROOT, "rounds_alt", f"r{r}")) for r in range(n_alt)], ["none", "raw"])
    report("Opus 5.5,目標 = 排名較後的 token", alt, FAMS)

    # Fable 5.1:第 0–2 輪(與報告的整體表相同,含 token 類;第 1、2 輪有先前各輪的回饋)
    fab = load([(f"r{r}", gt("rounds", r), os.path.join(ROOT, "rounds_fable", f"r{r}")) for r in range(3)], ["none", "raw"])
    report("Fable 5.1,第 0–2 輪(含 token 類)", fab, FAMS + ["token"])
    # 同樣三輪的 Opus,作為同一批修改上的對照
    op3 = load([(f"r{r}", gt("rounds", r), os.path.join(ROOT, "rounds", f"r{r}")) for r in range(3)], ["none", "raw"])
    report("Opus 5.5,第 0–2 輪(含 token 類;同一批修改)", op3, FAMS + ["token"])
    # Fable 的零樣本部分:第 0 輪,加上第 2 輪的無範例重跑
    fz = load([("r0", gt("rounds", 0), os.path.join(ROOT, "rounds_fable", "r0")),
               ("r2", gt("rounds", 2), os.path.join(ROOT, "rounds_fable_fewshot", "r2_k0"))], ["none", "raw"])
    report("Fable 5.1,零樣本(第 0 輪 + 第 2 輪無範例)", fz, FAMS + ["token"])


if __name__ == "__main__":
    main()
