# 獨立複核用的腳本:由另一個 agent 在看不到 iwibench_rollouts.py 與其結論的情況下從頭寫成。
# 兩份腳本刻意不合併,保留各自的定義以便互相對照。用法同 iwibench_rollouts.py。
#!/usr/bin/env python3
"""Independent analysis of IWIBench rollouts (family_version 0.2.2).

Run:  python3 -I analysis.py [data_dir] > output.txt
Reads only index.jsonl, r/<trace_id>/patch.json, r/<trace_id>/service-trajectory.jsonl.
"""
import collections as C
import json
import os
import re
import sys

DATA = sys.argv[1]

# ---------------------------------------------------------------- categories
# (regex on tensor name) -> (fine category, coarse category)
PATTERNS = [
    (r"^model\.language_model\.embed_tokens\.weight$", "embed_tokens", "EMB/UNEMB"),
    (r"^lm_head\.weight$", "lm_head", "EMB/UNEMB"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "mlp.gate_proj", "MLP"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight$", "mlp.up_proj", "MLP"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$", "mlp.down_proj", "MLP"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "attn.q_proj", "ATTN"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "attn.k_proj", "ATTN"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "attn.v_proj", "ATTN"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "attn.o_proj", "ATTN"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$", "attn.q_norm", "NORM"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$", "attn.k_norm", "NORM"),
    (r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$", "input_layernorm", "NORM"),
    (r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$", "post_attention_layernorm", "NORM"),
    (r"^model\.language_model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$", "pre_feedforward_layernorm", "NORM"),
    (r"^model\.language_model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$", "post_feedforward_layernorm", "NORM"),
    (r"^model\.language_model\.norm\.weight$", "final_norm", "FINAL_NORM"),
]
COARSE_ORDER = ["EMB/UNEMB", "MLP", "ATTN", "NORM", "FINAL_NORM", "UNKNOWN"]


def classify(name):
    """-> (fine, coarse, layer or None)"""
    for rx, fine, coarse in PATTERNS:
        m = re.match(rx, str(name))
        if m:
            return fine, coarse, (int(m.group(1)) if m.groups() else None)
    return "UNMAPPED:" + str(name), "UNKNOWN", None


def pattern(name):
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", str(name))


def excl_class(coarse_set):
    if not coarse_set:
        return "nothing"
    if coarse_set == {"EMB/UNEMB"}:
        return "emb/unemb only"
    if "EMB/UNEMB" in coarse_set:
        return "emb/unemb + other"
    return "other only"


def fnum(s):
    if s is None or s == "":
        return None
    return float(s)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def P(*a):
    print(*a)


def H(t):
    P()
    P("=" * 100)
    P(t)
    P("=" * 100)


def ed_label(r):
    return "%s | effort=%r" % (r["editor_model"], r["reasoning_effort"])


def layers_str(ts):
    by = C.defaultdict(set)
    for t in ts:
        fine, _, layer = classify(t)
        by[fine].add(layer)
    out = []
    for fine in sorted(by):
        ls = sorted(x for x in by[fine] if x is not None)
        out.append(fine + (" L" + ",".join(map(str, ls)) if ls else ""))
    return "; ".join(out)


# ---------------------------------------------------------------- load
all_rows = [json.loads(l) for l in open(os.path.join(DATA, "index.jsonl"))]
rows = [r for r in all_rows if r.get("family_version") == "0.2.2"]

H("LOAD")
P("index records total:", len(all_rows), "| by family_version:", dict(C.Counter(r.get("family_version") for r in all_rows)))
P("used (0.2.2):", len(rows), "| distinct trace_id:", len({r["trace_id"] for r in rows}))
P("dirs under r/:", len(os.listdir(os.path.join(DATA, "r"))),
  "| equal to 0.2.2 trace ids:", set(os.listdir(os.path.join(DATA, "r"))) == {r["trace_id"] for r in rows})

R = []  # per-rollout derived record
for r in rows:
    t = r["trace_id"]
    row = r["published_source_row"]
    if isinstance(row, str):
        row = json.loads(row)
    patch = json.load(open(os.path.join(DATA, "r", t, "patch.json")))
    tf = os.path.join(DATA, "r", t, "service-trajectory.jsonl")
    has_traj = os.path.exists(tf)
    traj = [json.loads(l) for l in open(tf)] if has_traj else []
    writes = [d for d in traj if d["tool"] == "write_weights"]
    okw = [d for d in writes if d["ok"] is True]
    badw = [d for d in writes if d["ok"] is not True]
    sparse = None
    if "sparse" in patch:
        s = patch["sparse"]
        sparse = [e["name"] for e in (json.loads(s) if isinstance(s, str) else s)]
    dense = None
    if "dense" in patch:
        s = patch["dense"]
        dense = json.loads(s) if isinstance(s, str) else s
    R.append(dict(
        r=r, t=t, row=row, patch=patch, has_traj=has_traj, traj=traj, writes=writes, okw=okw, badw=badw,
        ok_tensors=sorted({d["request"].get("tensor") for d in okw}),
        sparse=sparse, dense=dense,
        reward=fnum(row.get("reward")), ret=fnum(row.get("control_retention")), gain=fnum(row.get("gain")),
        editor=ed_label(r), family=r["family"],
    ))

for x in R:
    x["att_fine"] = {classify(t)[0] for t in x["ok_tensors"]}
    x["att_coarse"] = {classify(t)[1] for t in x["ok_tensors"]}
    x["att_class"] = excl_class(x["att_coarse"])
    p = x["patch"]
    if p["changed_parameters"] == 0:
        x["fin_known"] = True
        x["fin_tensors"] = []
    elif x["sparse"] is not None:
        x["fin_known"] = True
        x["fin_tensors"] = sorted(x["sparse"])
    else:
        x["fin_known"] = False  # edited patch, but patch.json names no tensors
        x["fin_tensors"] = None
    if x["fin_known"]:
        x["fin_coarse"] = {classify(t)[1] for t in x["fin_tensors"]}
        x["fin_class"] = "no-op patch" if not x["fin_tensors"] else excl_class(x["fin_coarse"])
    else:
        x["fin_coarse"] = None
        x["fin_class"] = "UNDETERMINED (edited, no tensor names in patch.json)"

# --- dense (unnamed) patch tensors: patch.json "sparse" names only sparse-encoded tensors.
# index.patch_validation gives tensor_count (safetensors entries) and sparse_parameter_count;
# each sparse tensor stores 2 entries (indices+values), so dense = tensor_count - 2*sparse.
ALIAS = lambda t: "model.language_model.embed_tokens.weight" if t == "lm_head.weight" else t
for x in R:
    v = x["r"]["patch_validation"]
    ns = len(x["sparse"] or [])
    x["pv_sparse_ok"] = v["sparse_parameter_count"] == ns
    x["dense_n"] = v["tensor_count"] - 2 * v["sparse_parameter_count"]
    x["patch_bytes"] = v["bytes"]
    x["dense_cand"] = sorted({ALIAS(t) for t in x["ok_tensors"]} - set(x["sparse"] or []))
    x["dense_resolved"] = x["dense_n"] == 0 or x["dense_n"] == len(x["dense_cand"])
    named = set(x["sparse"] or [])
    co = {classify(t)[1] for t in named}
    if x["dense_n"] > 0:
        cc_ = {classify(t)[1] for t in x["dense_cand"]}
        if x["dense_resolved"] or len(cc_ - {"EMB/UNEMB"}) == len(cc_):
            # every candidate is non-emb -> "other" certainly present (exact coarse set only exact if resolved)
            co |= cc_ if x["dense_resolved"] else {"(some of: %s)" % "/".join(sorted(cc_))}
        else:
            co |= {"UNRESOLVED"}
    x["fin2_coarse"] = co
    if x["patch"]["changed_parameters"] == 0:
        x["fin2_class"] = "no-op patch"
    elif co == {"EMB/UNEMB"}:
        x["fin2_class"] = "emb/unemb only"
    elif "UNRESOLVED" in co:
        x["fin2_class"] = "UNRESOLVED"
    elif "EMB/UNEMB" in co:
        x["fin2_class"] = "emb/unemb + other"
    else:
        x["fin2_class"] = "other only"
    x["fin2_other"] = x["fin2_class"] in ("emb/unemb + other", "other only")

P("missing reward:", sum(x["reward"] is None for x in R), "| missing control_retention:", sum(x["ret"] is None for x in R))
P("rollouts with trajectory file:", sum(x["has_traj"] for x in R), "| without:", sum(not x["has_traj"] for x in R))
P("service_trajectory_status:", dict(C.Counter(x["r"]["service_trajectory_status"] for x in R)))
P("patch_state (index):", dict(C.Counter(x["r"]["patch_state"] for x in R)))
P("trajectory records total:", sum(len(x["traj"]) for x in R))
tools = C.Counter((d["tool"], d["ok"]) for x in R for d in x["traj"])
for k in sorted(tools):
    P("  tool=%s ok=%s : %d" % (k[0], k[1], tools[k]))

# ---------------------------------------------------------------- mapping
H("TENSOR-NAME PATTERN -> CATEGORY MAPPING (every distinct pattern seen in write_weights requests or patch.json)")
seen = C.defaultdict(lambda: C.Counter())
for x in R:
    for d in x["writes"]:
        seen[pattern(d["request"].get("tensor"))]["write_ok" if d["ok"] is True else "write_failed"] += 1
    for t in (x["sparse"] or []):
        seen[pattern(t)]["patch_entries"] += 1
P("%-66s %-27s %-10s %8s %8s %8s" % ("pattern", "fine", "coarse", "ok_wr", "fail_wr", "patch"))
for pat in sorted(seen, key=lambda p: -seen[p]["write_ok"]):
    fine, coarse, _ = classify(pat.replace(".layers.N.", ".layers.0."))
    P("%-66s %-27s %-10s %8d %8d %8d" % (pat, fine, coarse, seen[pat]["write_ok"], seen[pat]["write_failed"], seen[pat]["patch_entries"]))

# ---------------------------------------------------------------- Q1
H("Q1. SUCCESSFUL write_weights CALLS")
okw_all = [(x, d) for x in R for d in x["okw"]]
badw_all = [(x, d) for x in R for d in x["badw"]]
P("write_weights calls total:", len(okw_all) + len(badw_all), "| succeeded (ok==true):", len(okw_all), "| failed:", len(badw_all))
P("rollouts with >=1 successful write:", sum(bool(x["okw"]) for x in R), "of", len(R))
N = len(okw_all)

P("\n-- by fine category (denominator = %d successful write calls); also distinct (rollout,tensor) pairs and rollouts" % N)
cf = C.Counter(classify(d["request"]["tensor"])[0] for _, d in okw_all)
pairs = {(x["t"], d["request"]["tensor"]) for x, d in okw_all}
pf = C.Counter(classify(t)[0] for _, t in pairs)
rf = C.Counter(f for x in R for f in x["att_fine"])
P("%-28s %-10s %7s %8s %12s %9s" % ("fine", "coarse", "calls", "pct", "ro*tensor", "rollouts"))
fine2coarse = {f: c for _, f, c in PATTERNS}
for f, n in cf.most_common():
    P("%-28s %-10s %7d %7.3f%% %12d %9d" % (f, fine2coarse.get(f, "UNKNOWN"), n, 100 * n / N, pf[f], rf[f]))
P("total distinct (rollout,tensor) pairs:", len(pairs))

P("\n-- by coarse category (denominator = %d successful write calls)" % N)
cc = C.Counter(classify(d["request"]["tensor"])[1] for _, d in okw_all)
pc = C.Counter(classify(t)[1] for _, t in pairs)
rc = C.Counter(c for x in R for c in x["att_coarse"])
for c in COARSE_ORDER:
    if cc[c]:
        P("%-12s calls=%5d (%.3f%%)  rollout*tensor pairs=%4d  rollouts=%3d" % (c, cc[c], 100 * cc[c] / N, pc[c], rc[c]))
P("non-EMB/UNEMB successful calls:", N - cc["EMB/UNEMB"], "(%.3f%%)" % (100 * (N - cc["EMB/UNEMB"]) / N))

P("\n-- by operation (denominator = %d successful write calls); failed shown for reference" % N)
co = C.Counter(d["request"].get("operation") for _, d in okw_all)
cob = C.Counter(d["request"].get("operation") for _, d in badw_all)
for o, n in co.most_common():
    P("%-14s ok=%5d (%.3f%%)   failed=%4d" % (o, n, 100 * n / N, cob[o]))
for o in cob:
    if o not in co:
        P("%-14s ok=%5d            failed=%4d" % (o, 0, cob[o]))

P("\n-- operation x coarse category (successful calls)")
oc = C.Counter((d["request"].get("operation"), classify(d["request"]["tensor"])[1]) for _, d in okw_all)
ops = [o for o, _ in co.most_common()]
P("%-14s" % "op" + "".join("%12s" % c for c in COARSE_ORDER[:5]))
for o in ops:
    P("%-14s" % o + "".join("%12d" % oc[(o, c)] for c in COARSE_ORDER[:5]))

P("\n-- whole-tensor vs selected writes (successful): 'selection' key present in request")
P(dict(C.Counter(("selection" if "selection" in d["request"] else "no selection (whole tensor)", classify(d["request"]["tensor"])[1]) for _, d in okw_all)))

P("\n-- layer histogram of successful non-embedding per-layer writes: layer -> {fine: calls}")
lh = C.defaultdict(C.Counter)
for _, d in okw_all:
    fine, coarse, layer = classify(d["request"]["tensor"])
    if layer is not None:
        lh[layer][fine] += 1
for l in sorted(lh):
    P("  layer %2d: total=%3d  %s" % (l, sum(lh[l].values()), dict(lh[l])))

P("\n-- failed write_weights calls (n=%d): by fine category and by error" % len(badw_all))
P("  by fine:", dict(C.Counter(classify(d["request"].get("tensor"))[0] for _, d in badw_all)))
P("  by error:", dict(C.Counter(d.get("error") for _, d in badw_all)))
P("  rollouts with >=1 failed write:", len({x["t"] for x, _ in badw_all}))

# ---------------------------------------------------------------- Q2
def stats_table(key, title, order=None):
    P("\n-- " + title)
    g = C.defaultdict(list)
    for x in R:
        g[key(x)].append(x)
    ks = order or sorted(g, key=lambda k: -len(g[k]))
    P("%-62s %4s %12s %12s %8s %12s" % ("class", "n", "mean_reward", "mean_retent", "n_rew>0", "mean_gain"))
    for k in ks:
        xs = g.get(k, [])
        if not xs:
            P("%-62s %4d" % (k, 0))
            continue
        P("%-62s %4d %12.6f %12.6f %8d %12.6f" % (k, len(xs), mean([x["reward"] for x in xs]), mean([x["ret"] for x in xs]),
                                                 sum(x["reward"] > 0 for x in xs), mean([x["gain"] for x in xs])))
    P("%-62s %4d %12.6f %12.6f %8d %12.6f" % ("ALL", len(R), mean([x["reward"] for x in R]), mean([x["ret"] for x in R]),
                                             sum(x["reward"] > 0 for x in R), mean([x["gain"] for x in R])))


H("Q2. ROLLOUTS BY CATEGORIES SUCCESSFULLY WRITTEN (attempt-based; denominator = 330 rollouts)")
EX = ["nothing", "emb/unemb only", "emb/unemb + other", "other only"]
stats_table(lambda x: x["att_class"], "mutually exclusive classes", EX)
stats_table(lambda x: "+".join(c for c in COARSE_ORDER if c in x["att_coarse"]) or "(nothing)", "exact coarse-category set")
stats_table(lambda x: "+".join(sorted(x["att_fine"])) or "(nothing)", "exact fine-category set")


def nothing_sub(x):
    if x["att_class"] != "nothing":
        return "wrote something"
    if not x["has_traj"]:
        return "nothing: no trajectory file"
    if not x["writes"]:
        return "nothing: trajectory, zero write_weights calls"
    return "nothing: trajectory, only failed write_weights calls"


stats_table(nothing_sub, "breakdown of 'nothing'")


def emb_sub(x):
    if x["att_class"] != "emb/unemb only":
        return "(not emb/unemb only)"
    return "emb/unemb only: " + "+".join(sorted(x["att_fine"]))


stats_table(emb_sub, "breakdown of 'emb/unemb only' by which tensor NAME was used")

# ---------------------------------------------------------------- Q3
def describe(x):
    p = x["patch"]
    per = C.Counter(d["request"]["tensor"] for d in x["okw"])
    P("  trace_id=%s" % x["t"])
    P("    editor=%s  family=%s  seed=%s" % (x["editor"], x["family"], x["r"].get("parameter_seed")))
    P("    reward=%r  gain=%r  control_retention=%r  published changed_parameters=%s" % (
        x["reward"], x["gain"], x["ret"], x["row"].get("changed_parameters")))
    P("    attempted class=%s | successful writes=%d, failed writes=%d" % (x["att_class"], len(x["okw"]), len(x["badw"])))
    P("    written (successful) summary: %s" % layers_str(x["ok_tensors"]))
    for t in sorted(per, key=lambda t: (classify(t)[1] != "EMB/UNEMB", t)):
        ops = C.Counter(d["request"].get("operation") for d in x["okw"] if d["request"]["tensor"] == t)
        P("      %-66s calls=%3d ops=%s" % (t, per[t], dict(ops)))
    P("    patch.json: changed_parameters=%s tensors=%s weight_revision=%s l2=%.6g linf=%.6g" % (
        p["changed_parameters"], p["tensors"], p["weight_revision"], p["l2"], p["linf"]))
    if x["sparse"] is not None:
        P("    patch.json sparse tensors (%d): %s" % (len(x["sparse"]), layers_str(x["sparse"])))
        for t in x["sparse"]:
            P("      " + t)
    else:
        P("    patch.json sparse tensors: KEY ABSENT" + ("" if p["changed_parameters"] == 0 else "  (edited patch with unnamed tensors)"))
    P("    dense (unnamed) tensors in patch per index.patch_validation: %d ; patch bytes=%d" % (x["dense_n"], x["patch_bytes"]))
    if x["dense_cand"]:
        P("    successfully written but not in the sparse name list (%d): %s" % (len(x["dense_cand"]), layers_str(x["dense_cand"])))
        P("      -> %d of these are dense tensors in the patch, %d reverted/unchanged%s" % (
            x["dense_n"], len(x["dense_cand"]) - x["dense_n"],
            "" if x["dense_resolved"] else "  (which ones not determinable from names alone)"))
    P("    final-patch class: names-only=%s | dense-inclusive=%s" % (x["fin_class"], x["fin2_class"]))


H("Q3. ROLLOUTS THAT SUCCESSFULLY WROTE TO ANYTHING OTHER THAN EMB/UNEMB (attempt-based)")
q3 = [x for x in R if x["att_coarse"] - {"EMB/UNEMB"}]
P("count:", len(q3), "of", len(R), "| reward>0:", sum(x["reward"] > 0 for x in q3),
  "| mean reward: %.6f | mean retention: %.6f" % (mean([x["reward"] for x in q3]), mean([x["ret"] for x in q3])))
P("  of which also wrote emb/unemb:", sum("EMB/UNEMB" in x["att_coarse"] for x in q3),
  "| other only:", sum("EMB/UNEMB" not in x["att_coarse"] for x in q3))
P("  by editor:", dict(C.Counter(x["editor"] for x in q3)))
P("  by family:", dict(C.Counter(x["family"] for x in q3)))
P("  by final-patch class (names-only):", dict(C.Counter(x["fin_class"] for x in q3)))
P("  by final-patch class (dense-inclusive):", dict(C.Counter(x["fin2_class"] for x in q3)))
for x in sorted(q3, key=lambda x: (x["editor"], x["family"], x["t"])):
    describe(x)

# ---------------------------------------------------------------- Q4
H("Q4. CLASSIFICATION BY FINAL PATCH (patch.json)")
P("patch.json changed_parameters==0:", sum(x["patch"]["changed_parameters"] == 0 for x in R),
  "| >0:", sum(x["patch"]["changed_parameters"] > 0 for x in R))
P("patch.json has 'sparse' key:", sum(x["sparse"] is not None for x in R), "| has 'dense' key:", sum(x["dense"] is not None for x in R))
P("edited (changed_parameters>0) but no tensor names:", [x["t"] for x in R if not x["fin_known"]])
P("patch 'tensors' field vs number of sparse names:", dict(C.Counter((x["patch"]["tensors"], len(x["sparse"]) if x["sparse"] is not None else None) for x in R)))
FIN = ["no-op patch", "emb/unemb only", "emb/unemb + other", "other only", "UNDETERMINED (edited, no tensor names in patch.json)"]
stats_table(lambda x: x["fin_class"], "mutually exclusive classes by final patch (denominator 330)", FIN)
stats_table(lambda x: ("+".join(c for c in COARSE_ORDER if c in x["fin_coarse"]) or "(no-op)") if x["fin_known"] else "UNDETERMINED",
            "exact coarse set in final patch")


def fin_inferred(x):
    # alternative: for the unnamed (dense) patches assume the tensors are the successfully-written ones
    if x["fin_known"]:
        return x["fin_class"]
    return excl_class(x["att_coarse"]) + " [INFERRED from trajectory]"


stats_table(fin_inferred, "ALTERNATIVE: same, with unnamed patches assigned the class of their successful writes (inference)")

P("\n-- cross-tab attempted class (rows) x final-patch class (cols); cells = rollouts")
ct = C.Counter((x["att_class"], x["fin_class"]) for x in R)
short = ["no-op", "emb only", "emb+other", "other only", "undeterm."]
P("%-22s" % "" + "".join("%12s" % s for s in short) + "%8s" % "total")
for a in EX:
    P("%-22s" % a + "".join("%12d" % ct[(a, f)] for f in FIN) + "%8d" % sum(ct[(a, f)] for f in FIN))
P("%-22s" % "total" + "".join("%12d" % sum(ct[(a, f)] for a in EX) for f in FIN) + "%8d" % len(R))

P("\n-- final-patch tensor entries by fine category (denominator = %d named patch entries across %d patches)" % (
    sum(len(x["sparse"] or []) for x in R), sum(bool(x["sparse"]) for x in R)))
fe = C.Counter(classify(t)[0] for x in R for t in (x["sparse"] or []))
fr = C.Counter(f for x in R for f in {classify(t)[0] for t in (x["sparse"] or [])})
for f, n in fe.most_common():
    P("  %-28s entries=%4d rollouts=%4d" % (f, n, fr[f]))

q4 = [x for x in R if (x["fin_known"] and x["fin_coarse"] - {"EMB/UNEMB"}) or not x["fin_known"]]
P("\n-- rollouts whose FINAL patch contains (or may contain) non-emb/unemb tensors: %d" % len(q4))
kn = [x for x in q4 if x["fin_known"]]
P("   determined from names: %d (reward>0: %d, mean reward %.6f, mean retention %.6f)" % (
    len(kn), sum(x["reward"] > 0 for x in kn), mean([x["reward"] for x in kn]), mean([x["ret"] for x in kn])))
P("   undetermined (unnamed tensors): %d (reward>0: %d)" % (len(q4) - len(kn), sum(x["reward"] > 0 for x in q4 if not x["fin_known"])))
P("   all %d: reward>0: %d, mean reward %.6f, mean retention %.6f" % (
    len(q4), sum(x["reward"] > 0 for x in q4), mean([x["reward"] for x in q4]), mean([x["ret"] for x in q4])))
for x in sorted(q4, key=lambda x: (x["editor"], x["family"], x["t"])):
    P("   %s | %-42s | %-32s | reward=%-8r ret=%-20r | %s" % (
        x["t"], x["editor"], x["family"], x["reward"], x["ret"],
        layers_str(x["fin_tensors"]) if x["fin_known"] else "UNNAMED; successful writes: " + layers_str(x["ok_tensors"])))

P("\n-- rollouts where a successfully written tensor is absent from the final patch (lm_head treated as alias of embed_tokens)")
alias = lambda t: "model.language_model.embed_tokens.weight" if t == "lm_head.weight" else t
absent = []
extra = []
for x in R:
    if not x["fin_known"]:
        continue
    w = {alias(t) for t in x["ok_tensors"]}
    f = set(x["fin_tensors"])
    if w - f:
        absent.append((x, sorted(w - f)))
    if f - w:
        extra.append((x, sorted(f - w)))
P("count:", len(absent), "| of which missing tensor is embed/lm_head:", sum(any(classify(t)[1] == "EMB/UNEMB" for t in m) for _, m in absent),
  "| missing a non-emb tensor:", sum(any(classify(t)[1] != "EMB/UNEMB" for t in m) for _, m in absent))
P("distinct (rollout,tensor) successfully written [alias-merged]: %d ; present in final patch: %d ; absent: %d (excluding the unnamed-patch rollouts)" % (
    sum(len({alias(t) for t in x["ok_tensors"]}) for x in R if x["fin_known"]),
    sum(len({alias(t) for t in x["ok_tensors"]} & set(x["fin_tensors"])) for x in R if x["fin_known"]),
    sum(len(m) for _, m in absent)))
ab_f = C.Counter(classify(t)[0] for _, m in absent for t in m)
P("absent (rollout,tensor) pairs by fine category:", dict(ab_f))
for x, m in sorted(absent, key=lambda a: (a[0]["editor"], a[0]["family"])):
    P("   %s | %-42s | %-32s | reward=%-8r | att=%-18s fin=%-15s | absent: %s" % (
        x["t"], x["editor"], x["family"], x["reward"], x["att_class"], x["fin_class"], layers_str(m)))
P("tensors in final patch with NO successful write in trajectory:", len(extra))
for x, m in extra:
    P("   ", x["t"], m, "has_traj=", x["has_traj"])

P("\n-- successful writes but no-op final patch (changed_parameters==0):")
for x in R:
    if x["okw"] and x["patch"]["changed_parameters"] == 0:
        P("   %s | %s | %s | ok_writes=%d | reward=%r | written: %s" % (x["t"], x["editor"], x["family"], len(x["okw"]), x["reward"], layers_str(x["ok_tensors"])))
P("-- no successful writes but edited final patch:", [x["t"] for x in R if not x["okw"] and x["patch"]["changed_parameters"] > 0])

H("Q4b. FINAL PATCH INCLUDING DENSE (UNNAMED) TENSORS, via index.patch_validation")
P("patch_validation.sparse_parameter_count == #names in patch.json sparse:", sum(x["pv_sparse_ok"] for x in R), "of", len(R))
P("dense tensor count (tensor_count - 2*sparse) distribution:", dict(C.Counter(x["dense_n"] for x in R)))
P("patch.json 'tensors' minus (#sparse + #dense), distribution:", dict(C.Counter(x["patch"]["tensors"] - len(x["sparse"] or []) - x["dense_n"] for x in R)))
P("patch.json 'tensors' == number of distinct successfully written tensors (alias-merged):",
  sum(x["patch"]["tensors"] == len({ALIAS(t) for t in x["ok_tensors"]}) for x in R), "of", len(R))
P("rollouts with dense tensors in the final patch:")
for x in R:
    if x["dense_n"]:
        P("   %s | %s | %s | reward=%r ret=%r | dense_n=%d | candidates(written, not in sparse)=%d: %s | resolved=%s | patch bytes=%d changed_parameters=%d | naive class=%s -> class=%s" % (
            x["t"], x["editor"], x["family"], x["reward"], x["ret"], x["dense_n"], len(x["dense_cand"]), layers_str(x["dense_cand"]),
            x["dense_resolved"], x["patch_bytes"], x["patch"]["changed_parameters"], x["fin_class"], x["fin2_class"]))
FIN2 = ["no-op patch", "emb/unemb only", "emb/unemb + other", "other only", "UNRESOLVED"]
stats_table(lambda x: x["fin2_class"], "mutually exclusive classes by final patch, dense tensors included (denominator 330)", FIN2)
stats_table(lambda x: "+".join(sorted(x["fin2_coarse"])) or "(no-op)", "exact coarse set in final patch, dense included")
P("\n-- cross-tab attempted class (rows) x final-patch class incl. dense (cols)")
ct2 = C.Counter((x["att_class"], x["fin2_class"]) for x in R)
P("%-22s" % "" + "".join("%12s" % s for s in ["no-op", "emb only", "emb+other", "other only", "unresolved"]) + "%8s" % "total")
for a_ in EX:
    P("%-22s" % a_ + "".join("%12d" % ct2[(a_, f)] for f in FIN2) + "%8d" % sum(ct2[(a_, f)] for f in FIN2))
P("%-22s" % "total" + "".join("%12d" % sum(ct2[(a_, f)] for a_ in EX) for f in FIN2) + "%8d" % len(R))
q4b = [x for x in R if x["fin2_other"]]
P("\n-- rollouts whose final patch contains non-emb/unemb tensors (dense included): %d | reward>0: %d | mean reward %.6f | mean retention %.6f" % (
    len(q4b), sum(x["reward"] > 0 for x in q4b), mean([x["reward"] for x in q4b]), mean([x["ret"] for x in q4b])))
P("   by editor:", dict(C.Counter(x["editor"] for x in q4b)))
P("   by family:", dict(C.Counter(x["family"] for x in q4b)))
for x in sorted(q4b, key=lambda x: (x["editor"], x["family"], x["t"])):
    P("   %s | %-42s | %-32s | reward=%-8r ret=%-20r | sparse: %s | dense(%d): %s" % (
        x["t"], x["editor"], x["family"], x["reward"], x["ret"], layers_str(x["sparse"] or []) or "-", x["dense_n"],
        (layers_str(x["dense_cand"]) + ("" if x["dense_resolved"] else " [%d of these %d]" % (x["dense_n"], len(x["dense_cand"])))) if x["dense_n"] else "-"))
P("\n-- attempted non-emb writes that did NOT survive into the final patch (rollout-level: attempted other, final has no other):")
for x in R:
    if (x["att_coarse"] - {"EMB/UNEMB"}) and not x["fin2_other"]:
        P("   %s | %s | %s | reward=%r | final=%s | non-emb written: %s" % (x["t"], x["editor"], x["family"], x["reward"], x["fin2_class"],
          layers_str([t for t in x["ok_tensors"] if classify(t)[1] != "EMB/UNEMB"])))
P("-- tensor-level: successfully written (alias-merged) tensors neither sparse-named nor accountable as dense:")
tot = 0
for x in R:
    left = len(x["dense_cand"]) - x["dense_n"]
    if left:
        tot += left
        P("   %s | %s | written-not-sparse=%d dense=%d -> %d reverted/unchanged: among %s" % (x["t"], x["editor"], len(x["dense_cand"]), x["dense_n"], left, layers_str(x["dense_cand"])))
P("   total reverted/unchanged (rollout,tensor) pairs:", tot, "of", sum(len({ALIAS(t) for t in x["ok_tensors"]}) for x in R))

# ---------------------------------------------------------------- Q5
H("Q5. PER EDITOR AND PER FAMILY")
editors = sorted({x["editor"] for x in R})
families = sorted({x["family"] for x in R})


def group_table(keyname, keys, alt_macro_key=None):
    P("%-56s %4s %11s %11s %7s | attempt: %8s %8s %8s | final: %6s %8s %8s %8s %7s" % (
        keyname, "n", "mean_rew", "mean_ret", "rew>0", "emb-only", "other", "nothing", "noop", "emb-only", "other", "undet", "macro"))
    for k in keys:
        xs = [x for x in R if x[keyname] == k]
        macro = ""
        if alt_macro_key:
            vals = [mean([x["reward"] for x in xs if x[alt_macro_key] == f]) for f in sorted({x[alt_macro_key] for x in R})]
            macro = "%.6f" % (sum(vals) / len(vals))
        a = C.Counter(x["att_class"] for x in xs)
        f = C.Counter(x["fin_class"] for x in xs)
        P("%-56s %4d %11.6f %11.6f %7d | %17d %8d %8d | %13d %8d %8d %8d %9s" % (
            k, len(xs), mean([x["reward"] for x in xs]), mean([x["ret"] for x in xs]), sum(x["reward"] > 0 for x in xs),
            a["emb/unemb only"], a["emb/unemb + other"] + a["other only"], a["nothing"],
            f["no-op patch"], f["emb/unemb only"], f["emb/unemb + other"] + f["other only"], f[FIN[4]], macro))
        f2 = C.Counter(x["fin2_class"] for x in xs)
        P("%-56s      final incl. dense: noop=%d emb-only=%d other=%d unresolved=%d" % ("", f2["no-op patch"], f2["emb/unemb only"],
          f2["emb/unemb + other"] + f2["other only"], f2["UNRESOLVED"]))


P("\n-- per editor (macro = mean over the 10 families of the within-family mean reward). 'other' = touched any non-emb/unemb component (with or without emb)")
group_table("editor", editors, "family")
P("\n-- per family (macro = mean over the 11 editors of the within-editor mean reward)")
group_table("family", families, "editor")

P("\n-- editor x family: n per cell")
P(dict(C.Counter(C.Counter((x["editor"], x["family"]) for x in R).values())), "(value -> number of cells)")
P("\n-- editor x family mean reward grid")
fam_s = [f[:10] for f in families]
P("%-46s" % "" + "".join("%11s" % s for s in fam_s))
for e in editors:
    P("%-46s" % e[:45] + "".join("%11.4f" % mean([x["reward"] for x in R if x["editor"] == e and x["family"] == f]) for f in families))
P("\n-- editor x family: attempted non-emb rollouts / final-patch non-emb (incl. undetermined) rollouts")
P("%-46s" % "" + "".join("%11s" % s for s in fam_s))
for e in editors:
    cells = []
    for f in families:
        xs = [x for x in R if x["editor"] == e and x["family"] == f]
        cells.append("%d/%d" % (sum(bool(x["att_coarse"] - {"EMB/UNEMB"}) for x in xs),
                                sum((not x["fin_known"]) or bool(x["fin_coarse"] - {"EMB/UNEMB"}) for x in xs)))
    P("%-46s" % e[:45] + "".join("%11s" % c for c in cells))

P("\n-- pooled by editor_model (ignoring reasoning effort)")
for m in sorted({x["r"]["editor_model"] for x in R}):
    xs = [x for x in R if x["r"]["editor_model"] == m]
    vals = [mean([x["reward"] for x in xs if x["family"] == f]) for f in families]
    P("  %-46s n=%3d mean_reward=%.6f macro=%.6f" % (m, len(xs), mean([x["reward"] for x in xs]), sum(vals) / len(vals)))

# ---------------------------------------------------------------- Q6
H("Q6. CONSISTENCY CHECKS / CAVEATS")
P("index published_reward == row reward:", sum(float(x["r"]["published_reward"]) == x["reward"] for x in R), "of", len(R))
P("index published_changed_parameters == row changed_parameters:", sum(int(x["r"]["published_changed_parameters"]) == int(x["row"]["changed_parameters"]) for x in R))
P("patch.json changed_parameters == row changed_parameters:", sum(x["patch"]["changed_parameters"] == int(x["row"]["changed_parameters"]) for x in R))
P("patch.json sha256 == row patch_sha256:", sum(x["patch"]["sha256"] == x["row"].get("patch_sha256") for x in R),
  "| == index patch_sha256:", sum(x["patch"]["sha256"] == x["r"]["patch_sha256"] for x in R))
P("row trace_id == index trace_id:", sum(x["row"].get("trace_id") == x["t"] for x in R),
  "| row editor_model == index:", sum(x["row"].get("editor_model") == x["r"]["editor_model"] for x in R),
  "| row family == index:", sum(x["row"].get("family") == x["family"] for x in R))
P("index patch_state valid_noop <=> patch changed_parameters==0:", sum((x["r"]["patch_state"] == "valid_noop") == (x["patch"]["changed_parameters"] == 0) for x in R))
P("patch weight_revision == #successful writes (all 330):", sum(x["patch"]["weight_revision"] == len(x["okw"]) for x in R))
mism = [(x["t"], x["patch"]["weight_revision"], len(x["okw"])) for x in R if x["patch"]["weight_revision"] != len(x["okw"])]
P("   mismatches:", mism)
P("trajectory 'revision' field non-decreasing in all files:", all([d["revision"] for d in x["traj"]] == sorted(d["revision"] for d in x["traj"]) for x in R))
P("revision increments by exactly 1 after each ok write and never otherwise:", all(
    all(b["revision"] - a["revision"] == (1 if (a["tool"] == "write_weights" and a["ok"] is True) else 0) for a, b in zip(x["traj"], x["traj"][1:]))
    for x in R))
P("row tool_calls == trajectory records:", sum(int(x["row"]["tool_calls"]) == len(x["traj"]) for x in R), "of", len(R))
tm = [(x["t"], x["row"]["tool_calls"], len(x["traj"]), x["has_traj"]) for x in R if int(x["row"]["tool_calls"]) != len(x["traj"])]
P("   mismatches (trace, row tool_calls, traj records, has_traj): n=%d" % len(tm), tm[:40])
P("index service_trajectory_records == lines in file:", sum((x["r"].get("service_trajectory_records") or 0) == len(x["traj"]) for x in R))
nt = [x for x in R if not x["has_traj"]]
P("rollouts without trajectory: %d | their patch changed_parameters: %s | weight_revision: %s | reward>0: %d | row tool_calls: %s" % (
    len(nt), dict(C.Counter(x["patch"]["changed_parameters"] for x in nt)), dict(C.Counter(x["patch"]["weight_revision"] for x in nt)),
    sum(x["reward"] > 0 for x in nt), dict(C.Counter(x["row"]["tool_calls"] for x in nt))))
P("   by editor:", dict(C.Counter(x["editor"] for x in nt)))
P("   by status:", dict(C.Counter(x["r"]["service_trajectory_status"] for x in nt)))
P("   retention values:", dict(C.Counter(x["ret"] for x in nt)))
dups = C.Counter((x["editor"], x["family"], x["r"].get("parameter_seed")) for x in R)
P("duplicate (editor, effort, family, parameter_seed):", {k: v for k, v in dups.items() if v > 1})
P("parameter_seed sets per (editor,family):", dict(C.Counter(
    tuple(sorted(x["r"].get("parameter_seed") for x in R if x["editor"] == e and x["family"] == f)) for e in editors for f in families)))
P("duplicate patch sha256 among edited patches:", {k: v for k, v in C.Counter(x["patch"]["sha256"] for x in R if x["patch"]["changed_parameters"] > 0).items() if v > 1})
P("distinct sha256 among no-op patches:", len({x["patch"]["sha256"] for x in R if x["patch"]["changed_parameters"] == 0}))
P("no-op patch rollouts: n=%d, reward>0: %d, retention!=1: %d, gain!=0: %d" % (
    sum(x["patch"]["changed_parameters"] == 0 for x in R),
    sum(x["reward"] > 0 for x in R if x["patch"]["changed_parameters"] == 0),
    sum(x["ret"] != 1.0 for x in R if x["patch"]["changed_parameters"] == 0),
    sum(x["gain"] != 0 for x in R if x["patch"]["changed_parameters"] == 0)))
P("reward distribution: ==0: %d, >0: %d, <0: %d, min=%r max=%r" % (
    sum(x["reward"] == 0 for x in R), sum(x["reward"] > 0 for x in R), sum(x["reward"] < 0 for x in R),
    min(x["reward"] for x in R), max(x["reward"] for x in R)))
P("grading_version:", dict(C.Counter(x["r"].get("grading_version") for x in R)),
  "| run_type:", dict(C.Counter(x["row"].get("run_type") for x in R)),
  "| split:", dict(C.Counter(x["r"].get("split") for x in R)))
P("published_source_file:", dict(C.Counter(x["r"].get("published_source_file") for x in R)))
P("native stop_condition:", dict(C.Counter((x["r"].get("native_completion_flags") or {}).get("stop_condition") for x in R)))
P("completion_diagnostics present:", sum(bool(x["r"].get("completion_diagnostics")) for x in R),
  "| final_response_truncated true:", sum(bool((x["r"].get("completion_diagnostics") or {}).get("final_response_truncated")) for x in R))
P("lm_head.weight ever named in a patch.json:", any("lm_head.weight" in (x["sparse"] or []) for x in R))
lmonly = [x for x in R if x["att_fine"] == {"lm_head"}]
P("rollouts whose only successful writes were to lm_head.weight: %d ; their patch sparse names: %s" % (
    len(lmonly), dict(C.Counter(tuple(x["sparse"] or ()) for x in lmonly))))
P("rollouts with successful lm_head writes:", sum("lm_head" in x["att_fine"] for x in R),
  "| with successful embed_tokens writes:", sum("embed_tokens" in x["att_fine"] for x in R),
  "| both:", sum({"lm_head", "embed_tokens"} <= x["att_fine"] for x in R))
P("lm_head users by editor:", dict(C.Counter(x["editor"] for x in R if "lm_head" in x["att_fine"])))
P("rollouts with only failed writes:", [(x["t"], x["editor"], len(x["badw"])) for x in R if x["badw"] and not x["okw"]])
P("non-emb tensors with FAILED writes only (rollout never succeeded on that tensor):")
for x in R:
    fo = sorted({d["request"].get("tensor") for d in x["badw"]} - set(x["ok_tensors"]))
    fo = [t for t in fo if classify(t)[1] != "EMB/UNEMB"]
    if fo:
        P("   %s | %s | att_class=%s | %s" % (x["t"], x["editor"], x["att_class"], layers_str(fo)))
P("excluded records (family_version != 0.2.2):", len(all_rows) - len(rows),
  dict(C.Counter((r["editor_model"], r["reasoning_effort"], r.get("target_model")) for r in all_rows if r.get("family_version") != "0.2.2")))
P("changed_parameters (patch.json) by final class: " + "; ".join(
    "%s: n=%d median=%s max=%s" % (k, len(v), sorted(v)[len(v) // 2], max(v))
    for k, v in ((k, [x["patch"]["changed_parameters"] for x in R if x["fin2_class"] == k]) for k in FIN2) if v))
for sha, n in C.Counter(x["patch"]["sha256"] for x in R if x["patch"]["changed_parameters"] > 0).items():
    if n > 1:
        for x in R:
            if x["patch"]["sha256"] == sha:
                P("   identical edited patch:", x["t"], x["editor"], x["family"], "seed", x["r"]["parameter_seed"], "reward", x["reward"], "changed", x["patch"]["changed_parameters"])
P("\n-- write sequences on non-emb tensors for rollouts whose dense/reverted status needs the sequence")
for tid in ("f81b39f0d61d400cbce52f048e21d819", "b10530f69f9c469da5577e201311f009", "b2764f18091747cabdc0d2537833d06a",
            "3a9e530190494d4abf5af35e4f9be56c", "5094c8a06e744d9bbf65016ce47a3d1a", "dc26642ee1934d79a558a00eaffccfbf"):
    x = [y for y in R if y["t"] == tid][0]
    P("  ", tid, x["editor"], "sparse:", layers_str(x["sparse"] or []), "dense_n:", x["dense_n"])
    n = 0
    for i, d in enumerate(x["traj"]):
        if d["tool"] == "write_weights" and classify(d["request"].get("tensor"))[1] != "EMB/UNEMB":
            n += 1
            if n > 14:
                continue
            rq = {k: (v if len(json.dumps(v)) < 90 else json.dumps(v)[:90] + "...") for k, v in d["request"].items() if k != "tensor"}
            P("      #%d rev=%d ok=%s %s %s" % (i, d["revision"], d["ok"], pattern(d["request"]["tensor"]).replace("model.language_model.", "") + "@" + str(classify(d["request"]["tensor"])[2]), rq))
    if n > 14:
        P("      ... (%d non-emb write calls in total)" % n)
