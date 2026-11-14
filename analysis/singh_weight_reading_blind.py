# 獨立複核用的腳本:由另一個 agent 在看不到 singh_weight_reading.py、作者的分析程式與報告的情況下從頭寫成。
# 兩份腳本刻意不合併。用法同 singh_weight_reading.py。
import sys
"""Independent recomputation from raw JSON only. Reads only interventions.json,
prompts.json, ground_truth.json and prediction.json files."""
import json, math, os, sys, collections
import numpy as np

ROOT = sys.argv[1]  # <repo>/experiments/systematic-tests
NB = 2000
SEED = 12345
notes = []

def jl(p):
    with open(p) as f:
        return json.load(f)

# ---------- metrics ----------
def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    s = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r

def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = math.sqrt((x * x).sum() * (y * y).sum())
    return float("nan") if d == 0 else float((x * y).sum() / d)

def spearman(x, y):
    if len(x) < 2: return float("nan")
    return pearson(rank(x), rank(y))

def auc(score, pos):
    pos = np.asarray(pos, bool)
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rank(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def auc_brute(score, pos):
    s = np.asarray(score, float); pos = np.asarray(pos, bool)
    a = s[pos][:, None]; b = s[~pos][None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.shape[1])) if a.size and b.size else float("nan")

# self-tests
assert abs(spearman(np.array([1., 2, 2, 3]), np.array([1., 3, 2, 4])) - pearson(np.array([1, 2.5, 2.5, 4]), np.array([1., 3, 2, 4]))) < 1e-12
assert list(rank([3, 1, 3, 2])) == [3.5, 1, 3.5, 2]
_rs = np.random.default_rng(0)
for _ in range(20):
    s = np.round(_rs.normal(size=60), 1); p = _rs.random(60) > 0.6
    assert abs(auc(s, p) - auc_brute(s, p)) < 1e-12

# ---------- loading ----------
interv = jl(os.path.join(ROOT, "data/interventions.json"))
IV = {}
iv_dups = []
for fam, lst in interv.items():
    for d in lst:
        if d["iid"] in IV: iv_dups.append(d["iid"])
        IV[d["iid"]] = d
prompts = jl(os.path.join(ROOT, "data/prompts.json"))
PIDS = [p["pid"] for p in prompts]

GT = {}
def load_gt(tree, rnd):
    key = (tree, rnd)
    if key in GT: return GT[key]
    g = jl(os.path.join(ROOT, tree, rnd, "ground_truth.json"))
    out = {}
    for iid, d in g.items():
        pp = d["per_prompt"]
        if sorted(pp) != sorted(PIDS):
            notes.append(f"GT {tree}/{rnd} {iid}: prompts {sorted(pp)}")
        for pid, v in pp.items():
            x = v.get("dlogp")
            if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
                notes.append(f"GT non-finite {tree}/{rnd} {iid}|{pid}: {x!r}")
                continue
            out[(iid, pid)] = (d["family"], float(x))
    GT[key] = out
    return out

ignored = collections.Counter()
bad_pred = []
def load_pred(tree, rnd, fam, cond):
    p = os.path.join(ROOT, tree, rnd, fam, cond, "prediction.json")
    if not os.path.exists(p):
        return None
    j = jl(p)
    if not (isinstance(j, dict) and "predictions" in j and isinstance(j["predictions"], dict)):
        raise SystemExit(f"unexpected shape {p}: {type(j)} keys={list(j)[:5] if isinstance(j, dict) else ''}")
    extra = [k for k in j if k != "predictions"]
    if extra: notes.append(f"{p}: extra top-level keys {extra}")
    out = {}
    for k, v in j["predictions"].items():
        parts = k.split("|")
        if len(parts) != 2:
            bad_pred.append((p, k, "bad key")); continue
        x = v.get("dlogp") if isinstance(v, dict) else v
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
            bad_pred.append((tree, rnd, fam, cond, k, repr(x))); continue
        out[(parts[0], parts[1])] = float(x)
    return out

FAMS = ["scalar", "neuron", "layer", "token"]
integrity = []
def bundle(tree, rnd, gt_tree, gt_rnd, label):
    """rows: dict(src,iid,pid,fam,y,none,raw)"""
    gt = load_gt(gt_tree, gt_rnd)
    rows = []
    d = os.path.join(ROOT, tree, rnd)
    for fam in sorted(os.listdir(d)):
        fd = os.path.join(d, fam)
        if not os.path.isdir(fd): continue
        if fam not in FAMS:
            notes.append(f"unknown family folder {fd}"); continue
        for c in sorted(os.listdir(fd)):
            if c not in ("none", "raw") and os.path.isdir(os.path.join(fd, c)):
                ignored[(tree, c)] += 1
        preds = {c: load_pred(tree, rnd, fam, c) for c in ("none", "raw")}
        gtf = {k: v[1] for k, v in gt.items() if v[0] == fam}
        for c in ("none", "raw"):
            if preds[c] is None:
                integrity.append(f"{label}/{fam}/{c}: prediction.json MISSING"); preds[c] = {}
                continue
            extra = set(preds[c]) - set(gtf)
            miss = set(gtf) - set(preds[c])
            if extra: integrity.append(f"{label}/{fam}/{c}: {len(extra)} pred keys without GT of this family, e.g. {sorted(extra)[:3]}")
            if miss: integrity.append(f"{label}/{fam}/{c}: {len(miss)} GT items without valid prediction, e.g. {sorted(miss)[:3]}")
        for (iid, pid), y in sorted(gtf.items()):
            rows.append(dict(src=label, iid=iid, pid=pid, fam=fam, y=y,
                             none=preds["none"].get((iid, pid), float("nan")),
                             raw=preds["raw"].get((iid, pid), float("nan"))))
    return rows

# ---------- analysis ----------
def arr(rows, k): return np.array([r[k] for r in rows], float)

def boot_diff(rows, stratify=False):
    """cluster bootstrap over edits (cluster = (src, iid)); items with both preds."""
    rows = [r for r in rows if not (math.isnan(r["none"]) or math.isnan(r["raw"]))]
    cl = collections.OrderedDict()
    for i, r in enumerate(rows):
        cl.setdefault((r["src"], r["iid"]), []).append(i)
    keys = list(cl)
    idxs = [np.array(cl[k]) for k in keys]
    y = arr(rows, "y"); a = arr(rows, "none"); b = arr(rows, "raw")
    rng = np.random.default_rng(SEED)
    strata = None
    if stratify:
        fam_of = [rows[cl[k][0]]["fam"] for k in keys]
        strata = [np.array([i for i, f in enumerate(fam_of) if f == ff]) for ff in sorted(set(fam_of))]
    diffs = []; nan = 0
    for _ in range(NB):
        if strata is None:
            pick = rng.integers(0, len(keys), len(keys))
        else:
            pick = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        ii = np.concatenate([idxs[p] for p in pick])
        d = spearman(b[ii], y[ii]) - spearman(a[ii], y[ii])
        if math.isnan(d): nan += 1
        else: diffs.append(d)
    diffs = np.array(diffs)
    if len(diffs) < 2: return float("nan"), nan, (float("nan"),) * 2
    return float(diffs.std(ddof=1)), nan, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def f(x): return "  nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"

def sign_stats(rows, cond, T):
    sel = [r for r in rows if abs(r["y"]) > T and not math.isnan(r[cond])]
    n = len(sel)
    if n == 0: return "n=0"
    ok = sum(1 for r in sel if np.sign(r[cond]) == np.sign(r["y"]))
    z = sum(1 for r in sel if r[cond] == 0)
    ne = len({(r["src"], r["iid"]) for r in sel})
    nz = n - z
    return f"{f(ok / n)} (items={n}, edits={ne}, pred==0: {z}; excl. zero-preds: {f(ok / nz) if nz else 'nan'})"

def report(title, rows, fams, sign=True, aucs=True):
    print(f"\n=== {title} ===")
    groups = [(fm, [r for r in rows if r["fam"] == fm]) for fm in fams] + [("POOLED", [r for r in rows if r["fam"] in fams])]
    for name, rs in groups:
        if not rs:
            print(f"[{name}] no items"); continue
        both = [r for r in rs if not (math.isnan(r["none"]) or math.isnan(r["raw"]))]
        ne = len({(r["src"], r["iid"]) for r in rs})
        y = arr(both, "y"); a = arr(both, "none"); b = arr(both, "raw")
        sn, sr = spearman(a, y), spearman(b, y)
        se, nan, ci = boot_diff(rs)
        line = f"[{name}] edits={ne} items={len(rs)} items_with_both={len(both)}\n   Spearman none={f(sn)} raw={f(sr)} raw-none={f(sr - sn)} SE={f(se)} (boot 95% pct CI [{f(ci[0])},{f(ci[1])}], NaN draws={nan})"
        if name == "POOLED":
            se2, nan2, ci2 = boot_diff(rs, stratify=True)
            line += f"\n   alt SE, family-stratified edit bootstrap: {f(se2)} (NaN draws={nan2})"
            # mean of per-family as alternative reading
        print(line)
        print(f"   MAE none={f(float(np.abs(a - y).mean()))} raw={f(float(np.abs(b - y).mean()))} always-0={f(float(np.abs(y).mean()))}   (median |y|={f(float(np.median(np.abs(y))))})")
        if aucs:
            pos = np.abs(y) > 0.1
            print(f"   large-effect AUC (|y|>0.1; positives={int(pos.sum())}/{len(y)}) none={f(auc(np.abs(a), pos))} raw={f(auc(np.abs(b), pos))}")
        if sign:
            for T in (0.1, 1.0):
                print(f"   sign acc |y|>{T}: none {sign_stats(both, 'none', T)}")
                print(f"   sign acc |y|>{T}:  raw {sign_stats(both, 'raw', T)}")

def main():
    R = {}
    for i in range(9):
        R[f"r{i}"] = bundle("rounds", f"r{i}", "rounds", f"r{i}", f"rounds/r{i}")
    FS = bundle("rounds_fewshot", "r2_k0", "rounds", "r2", "rounds_fewshot/r2_k0")
    ALT = {}
    for rnd in sorted(os.listdir(os.path.join(ROOT, "rounds_alt"))):
        ALT[rnd] = bundle("rounds_alt", rnd, "rounds_alt", rnd, f"rounds_alt/{rnd}")
    FB = {r: bundle("rounds_fable", r, "rounds", r, f"rounds_fable/{r}") for r in sorted(os.listdir(os.path.join(ROOT, "rounds_fable")))}
    FBFS = bundle("rounds_fable_fewshot", "r2_k0", "rounds", "r2", "rounds_fable_fewshot/r2_k0")
    # note other subfolders in fewshot trees
    for t in ("rounds_fewshot", "rounds_fable_fewshot"):
        for x in sorted(os.listdir(os.path.join(ROOT, t))):
            if x != "r2_k0": notes.append(f"ignored bundle {t}/{x}")

    three = ["scalar", "neuron", "layer"]
    A = R["r0"] + FS + sum((R[f"r{i}"] for i in range(3, 9)), [])
    report("A. first predictor, zero-shot set (r0 + fewshot/r2_k0 + r3..r8), scalar/neuron/layer", A, three)
    A_small = R["r0"] + FS
    report("A-alt. same but only r0 + fewshot/r2_k0 (for reference)", A_small, three, sign=False)
    B = sum(ALT.values(), [])
    report(f"B. rounds_alt ({','.join(ALT)}), scalar/neuron/layer", B, three)
    report("C(i). second predictor rounds_fable/r0", FB["r0"], FAMS, sign=False, aucs=True)
    report("C(ii). second predictor rounds_fable/r0 + rounds_fable_fewshot/r2_k0", FB["r0"] + FBFS, FAMS, sign=False)
    report("C(iii). second predictor rounds_fable/r0-r2", FB["r0"] + FB["r1"] + FB["r2"], FAMS, sign=False)
    report("D. first predictor rounds/r0-r2", R["r0"] + R["r1"] + R["r2"], FAMS, sign=False)
    report("D-extra. first predictor r0 only", R["r0"], FAMS, sign=False)
    report("D-extra. first predictor r0 + rounds_fewshot/r2_k0 (4 families)", R["r0"] + FS, FAMS, sign=False)

    # ---------- E ----------
    print("\n=== E. integrity ===")
    print("interventions.json duplicate iids:", iv_dups, "| total edits:", len(IV), {k: len(v) for k, v in interv.items()})
    print("prompts:", PIDS)
    # uniqueness across rounds
    for tree, src in (("rounds", R), ("rounds_alt", ALT)):
        seen = collections.defaultdict(set)
        for rnd, rows in src.items():
            for r in rows: seen[r["iid"]].add(rnd)
        dup = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        print(f"{tree}: distinct edit ids={len(seen)}; ids in >1 round: {len(dup)} {list(dup.items())[:5]}")
        # GT edits w/o interventions entry, round mismatch
        mism = []
        for rnd in src:
            g = load_gt(tree, rnd)
            for iid in {k[0] for k in g}:
                if iid not in IV: mism.append((rnd, iid, "not in interventions.json"))
                else:
                    if f"r{IV[iid].get('round')}" != rnd: mism.append((rnd, iid, f"interv round={IV[iid].get('round')}"))
                    fam = [v[0] for k, v in g.items() if k[0] == iid][0]
                    if IV[iid]["family"] != fam: mism.append((rnd, iid, "family mismatch"))
        print(f"{tree}: GT-vs-interventions mismatches: {len(mism)} {mism[:8]}")
        print(f"{tree}: per round edits(items) by family")
        for rnd in src:
            g = load_gt(tree, rnd)
            c = collections.Counter(); e = collections.defaultdict(set)
            for (iid, pid), (fam, _) in g.items(): c[fam] += 1; e[fam].add(iid)
            pf = sorted({r['fam'] for r in src[rnd]})
            print(f"   {rnd}: " + ", ".join(f"{fm}={len(e[fm])}({c[fm]})" for fm in FAMS if c[fm]) + f" | families with prediction folders: {pf}")
    ivr = collections.Counter((d.get("round"), d["family"]) for d in IV.values())
    print("interventions.json edits by (round,family):", sorted(ivr.items(), key=str))
    print("ignored condition folders (tree, name) -> count of family dirs:", dict(ignored))
    print("bad/non-numeric predictions:", len(bad_pred), bad_pred[:10])
    print("integrity messages:", len(integrity))
    for m in integrity: print("   ", m)
    for n in notes: print("   note:", n)

    # zeros in predictions
    for nm, rows in (("A", A), ("B", B)):
        rows = [r for r in rows if r["fam"] in three]
        for c in ("none", "raw"):
            v = arr(rows, c)
            print(f"{nm} {c}: exact-zero preds {int((v == 0).sum())}/{len(v)}; distinct values {len(set(v.tolist()))}; frac negative {float((v < 0).mean()):.3f}")
        y = arr(rows, "y")
        print(f"{nm} measured: frac negative {float((y < 0).mean()):.3f}; among |y|>0.1 frac negative {float((y[np.abs(y) > 0.1] < 0).mean()):.3f}; among |y|>1 frac negative {float((y[np.abs(y) > 1] < 0).mean()):.3f}")

    # main vs alt GT comparison
    xs = []; ys = []; same = 0; flips_main = flips_alt = 0
    for rnd in ALT:
        gm = load_gt("rounds", rnd); ga = load_gt("rounds_alt", rnd)
        for k, (fam, v) in ga.items():
            if k in gm and fam in three:
                xs.append(gm[k][1]); ys.append(v); same += abs(gm[k][1] - v) < 1e-4
    xs = np.array(xs); ys = np.array(ys)
    print(f"main-vs-alt GT (scalar/neuron/layer, shared items={len(xs)}): identical within 1e-4: {int(same)}; Spearman={f(spearman(xs, ys))}; Pearson={f(pearson(xs, ys))}; median|diff|={f(float(np.median(np.abs(xs - ys))))}")
    # alt vs main predictions identical?
    pm = {(r["iid"], r["pid"]): r for rnd in ALT for r in (R[rnd] if rnd not in ("r1", "r2") else R[rnd])}
    for c in ("none", "raw"):
        a = []; b = []
        for r in B:
            if (r["iid"], r["pid"]) in pm and not math.isnan(r[c]) and not math.isnan(pm[(r["iid"], r["pid"])][c]):
                a.append(r[c]); b.append(pm[(r["iid"], r["pid"])][c])
        a = np.array(a); b = np.array(b)
        print(f"alt-vs-main predictions [{c}] n={len(a)}: identical={int((a == b).sum())}; Spearman={f(spearman(a, b))}")

    # concentration
    def conc(nm, rows, fams):
        rows = [r for r in rows if r["fam"] in fams]
        ne = len({(r["src"], r["iid"]) for r in rows})
        print(f"\n-- concentration of large effects: {nm} (edits={ne}, items={len(rows)})")
        for T in (0.1, 1.0):
            big = [r for r in rows if abs(r["y"]) > T]
            ce = collections.Counter((r["iid"]) for r in big)
            top = ce.most_common()
            cum = np.cumsum([c for _, c in top]) if top else []
            def need(fr):
                return int(np.searchsorted(cum, fr * len(big)) + 1) if len(big) else 0
            print(f" |y|>{T}: items={len(big)} ({len(big) / len(rows):.3f}) from {len(ce)} edits; edits with all 8 prompts big: {sum(1 for _, c in top if c == 8)}; edits needed for 50%/80% of big items: {need(0.5)}/{need(0.8)}")
            print("   by family:", dict(collections.Counter(r["fam"] for r in big)))
            kc = collections.Counter((IV[r["iid"]]["family"], IV[r["iid"]].get("kind"), IV[r["iid"]].get("op")) for r in big if r["iid"] in IV)
            tot = collections.Counter((IV[r["iid"]]["family"], IV[r["iid"]].get("kind"), IV[r["iid"]].get("op")) for r in rows if r["iid"] in IV)
            print("   by (family,kind,op): big/total items")
            for k, c in sorted(tot.items(), key=lambda kv: -kc.get(kv[0], 0)):
                print(f"      {k}: {kc.get(k, 0)}/{c}")
            print("   by prompt:", dict(sorted(collections.Counter(r["pid"] for r in big).items())))
            sal = collections.Counter(str(IV[r["iid"]].get("salient")) for r in big if r["iid"] in IV)
            print("   by 'salient' tag (top 8):", sal.most_common(8))
        # abs-sum share
        tot = collections.Counter()
        for r in rows: tot[(r["src"], r["iid"])] += abs(r["y"])
        v = np.sort(np.array(list(tot.values())))[::-1]
        print(f" share of sum|y| from top 5 / top 10 edits: {v[:5].sum() / v.sum():.3f} / {v[:10].sum() / v.sum():.3f}")
        # Spearman after dropping big-effect kinds: edit-level robustness
        y = arr(rows, "y")
        for c in ("none", "raw"):
            small = [r for r in rows if abs(r["y"]) <= 0.1 and not math.isnan(r[c])]
            bigr = [r for r in rows if abs(r["y"]) > 0.1 and not math.isnan(r[c])]
            print(f" Spearman[{c}] within |y|<=0.1 items (n={len(small)}): {f(spearman(arr(small, c), arr(small, 'y')))}; within |y|>0.1 (n={len(bigr)}): {f(spearman(arr(bigr, c), arr(bigr, 'y')))}; Spearman of |pred| vs |y| all: {f(spearman(np.abs(arr([r for r in rows if not math.isnan(r[c])], c)), np.abs(arr([r for r in rows if not math.isnan(r[c])], 'y'))))}")
    conc("set A", A, three)
    conc("set B (alt)", B, three)

    # per-round Spearman for A components
    print("\n-- per-source Spearman (pooled scalar/neuron/layer), first predictor")
    for nm, rows in [("rounds/r0", R["r0"]), ("rounds/r1 (had examples)", R["r1"]), ("rounds/r2 (had examples)", R["r2"]), ("rounds_fewshot/r2_k0", FS)] + [(f"rounds/r{i}", R[f"r{i}"]) for i in range(3, 9)]:
        rs = [r for r in rows if r["fam"] in three and not (math.isnan(r["none"]) or math.isnan(r["raw"]))]
        y = arr(rs, "y")
        print(f"   {nm}: edits={len({r['iid'] for r in rs})} items={len(rs)} none={f(spearman(arr(rs, 'none'), y))} raw={f(spearman(arr(rs, 'raw'), y))}")
    print("-- per-source, rounds_alt")
    for rnd, rows in ALT.items():
        rs = [r for r in rows if r["fam"] in three and not (math.isnan(r["none"]) or math.isnan(r["raw"]))]
        y = arr(rs, "y")
        print(f"   {rnd}: edits={len({r['iid'] for r in rs})} items={len(rs)} none={f(spearman(arr(rs, 'none'), y))} raw={f(spearman(arr(rs, 'raw'), y))}")

main()

def extra():
    three = ["scalar", "neuron", "layer"]
    R = {f"r{i}": bundle("rounds", f"r{i}", "rounds", f"r{i}", f"rounds/r{i}") for i in range(9)}
    FS = bundle("rounds_fewshot", "r2_k0", "rounds", "r2", "rounds_fewshot/r2_k0")
    A = R["r0"] + FS + sum((R[f"r{i}"] for i in range(3, 9)), [])
    B = sum((bundle("rounds_alt", f"r{i}", "rounds_alt", f"r{i}", f"rounds_alt/r{i}") for i in range(9)), [])
    print("\n-- sign baselines: always-negative accuracy and frac negative preds")
    for nm, rows in (("A", A), ("B", B)):
        for fm in three + ["POOLED"]:
            rs = [r for r in rows if (r["fam"] == fm or (fm == "POOLED" and r["fam"] in three))]
            s = f"{nm} {fm}: frac pred<0 none={np.mean([r['none'] < 0 for r in rs]):.3f} raw={np.mean([r['raw'] < 0 for r in rs]):.3f};"
            for T in (0.1, 1.0):
                sel = [r for r in rs if abs(r["y"]) > T]
                s += f" always-neg acc |y|>{T}: {f(float(np.mean([r['y'] < 0 for r in sel]))) if sel else 'n=0'} (n={len(sel)});"
            print(s)
extra()
