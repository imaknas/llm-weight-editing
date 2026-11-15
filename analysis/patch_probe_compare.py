"""在同一批事實上比較兩種修改檔:詞表列的修改與 MLP 神經元的修改。

三個步驟,前兩個需要 GPU,每個條件各自存檔,可中斷後接著跑:
  match  每個候選修改檔在基準自己的題目上逐題單獨生成(一批一題),用基準的 scoring 評分;
         每個事實挑出與神經元修改檔同分的詞表修改檔(照 spec 裡候選的順序取第一個)。
  probe  對 base(跑兩次)、配對到的詞表修改檔、神經元修改檔(完整)、神經元修改檔(只留內部層)
         逐題單獨生成探針;paraphrase 類另外記錄答案位置上「新答案第一個 token」對「base 實際回答的第一個 token」的 logit 差。
  report 只用存檔計分與彙整,不需要 GPU。

spec.json:{"src", "cache", "out", "max_new_tokens", "probes": [探針檔...],
            "facts": {"s2": {"seed": 2, "token_table": [["名稱", "patch 路徑"], ...], "neuron_full": 路徑, "neuron_int": 路徑}}}
用法:python analysis/patch_probe_compare.py <match|probe|report> --spec <spec.json> [--fact s2] [--device mps]
"""
import argparse
import json
import sys
from pathlib import Path

GROUPS = ("calibration", "targets", "nearby_controls", "general_controls", "override_controls")
MARGIN_CATEGORY = "paraphrase"


def load_probes(spec, fact):
    seen, out = set(), []
    for k, path in enumerate(spec["probes"]):
        for i, p in enumerate(json.loads(Path(path).read_text())):
            if p["fact"] == fact and p["prompt"] not in seen:
                seen.add(p["prompt"])
                out.append({**p, "id": f"{'ab'[k]}{i}"})
    return out


def generate_alone(target, prompts, n):
    return [target.generate([p], n, [0])[0]["text"] for p in prompts]


def with_target(spec, ep, patch, device, fn):
    import torch
    from iwibench.runtime import NativeTarget
    target = NativeTarget(ep["model_id"], ep["checkpoint_revision"], device=device, cache_dir=str(Path(spec["cache"]).resolve()))
    if patch:
        target.apply_patch(patch)
    try:
        return fn(target)
    finally:
        del target
        if device == "mps":
            torch.mps.empty_cache()


def conditions_match(f):
    return {"base": None, **dict(f["token_table"]), "neuron_full": f["neuron_full"], "neuron_int": f["neuron_int"]}


def match(spec, fact, device):
    from iwibench import families, scoring
    f = spec["facts"][fact]
    ep = families.episode("factual_update", f["seed"], "development", "google/gemma-3-4b-it")
    items = [it for g in GROUPS for it in ep[g]]
    d = Path(spec["out"]) / fact
    d.mkdir(parents=True, exist_ok=True)
    for name, patch in conditions_match(f).items():
        done = d / f"match-{name}.json"
        if not done.exists():
            texts = with_target(spec, ep, patch, device, lambda t: generate_alone(t, [it["prompt"] for it in items], spec["max_new_tokens"]))
            done.write_text(json.dumps({it["id"]: x for it, x in zip(items, texts)}, ensure_ascii=False, indent=1))
        print(fact, "match", name, "done", flush=True)
    base = json.loads((d / "match-base.json").read_text())
    rows = {}
    for name in conditions_match(f):
        if name != "base":
            m = scoring.score_episode(ep, base, json.loads((d / f"match-{name}.json").read_text()))
            g = m["groups"]
            rows[name] = {"reward": m["reward"], "targets": g["targets"]["edited_successes"],
                          "controls_retained": sum(g[k]["retained_successes"] for k in g), "controls": sum(g[k]["retention_denominator"] for k in g)}
    ref = rows["neuron_full"]["reward"]
    chosen = next((n for n, _ in f["token_table"] if abs(rows[n]["reward"] - ref) < 1e-9), None)
    (d / "match.json").write_text(json.dumps({"rows": rows, "matched": chosen}, indent=1))
    print(fact, json.dumps(rows), "matched:", chosen, flush=True)


def probe(spec, fact, device):
    import torch
    from iwibench import families
    f = spec["facts"][fact]
    ep = families.episode("factual_update", f["seed"], "development", "google/gemma-3-4b-it")
    d = Path(spec["out"]) / fact
    chosen = json.loads((d / "match.json").read_text())["matched"]
    probes = load_probes(spec, fact)
    (d / "probes.json").write_text(json.dumps(probes, ensure_ascii=False, indent=1))
    conds = {"base": None, "base2": None, "neuron_full": f["neuron_full"], "neuron_int": f["neuron_int"]}
    if chosen:
        conds["token_table"] = dict(f["token_table"])[chosen]
    new_answer = ep["public_practice"][0]["answers"][0]

    def run(target, first_base):
        texts = generate_alone(target, [p["prompt"] for p in probes], spec["max_new_tokens"])
        firsts, margins = {}, {}
        new_tok = target.tokenizer.encode(new_answer, add_special_tokens=False)[0]
        for p in probes:
            if p["category"] != MARGIN_CATEGORY:
                continue
            ids = torch.tensor([target._prompt_ids(p["prompt"])], dtype=torch.long, device=target.device)
            with torch.inference_mode():
                lg = target.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits[0, -1].float()
            firsts[p["id"]] = int(lg.argmax())
            ref = first_base.get(p["id"], firsts[p["id"]]) if first_base is not None else firsts[p["id"]]
            margins[p["id"]] = (lg[new_tok] - lg[ref]).item()
        return {"outputs": {p["id"]: t for p, t in zip(probes, texts)}, "first_token": firsts, "margin_new_minus_base_first": margins}

    first_base = None
    for name, patch in conds.items():
        done = d / f"probe-{name}.json"
        if not done.exists():
            if name != "base" and first_base is None:
                first_base = json.loads((d / "probe-base.json").read_text())["first_token"]
            res = with_target(spec, ep, patch, device, lambda t: run(t, first_base if name != "base" else None))
            done.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(fact, "probe", name, "done", flush=True)


def meets(p, out, base_out):
    if p["expect_type"] == "contains":
        return p["expect"].lower() in out.lower()
    if p["expect_type"] == "same_as_base":
        return out.strip() == base_out.strip()
    return None


def report(spec):
    result = {}
    for fact in spec["facts"]:
        d = Path(spec["out"]) / fact
        if not (d / "probe-neuron_int.json").exists():
            continue
        probes = json.loads((d / "probes.json").read_text())
        m = json.loads((d / "match.json").read_text())
        res = {c: json.loads((d / f"probe-{c}.json").read_text()) for c in ("base", "base2", "token_table", "neuron_full", "neuron_int") if (d / f"probe-{c}.json").exists()}
        base, base2 = res["base"]["outputs"], res["base2"]["outputs"]
        cats = {}
        for p in probes:
            c = cats.setdefault(p["category"], {"n": 0, "dropped": 0, "base": 0, "token_table": 0, "neuron_full": 0, "neuron_int": 0,
                                               "favours_token_table": [], "favours_neuron": [], "full_vs_int_differ": []})
            if base[p["id"]].strip() != base2[p["id"]].strip():
                c["dropped"] += 1
                continue
            c["n"] += 1
            ok = {k: meets(p, res[k]["outputs"][p["id"]], base[p["id"]]) for k in res if k != "base2"}
            for k, v in ok.items():
                c[k] += bool(v)
            if "token_table" in ok and ok["token_table"] is not None:
                if ok["token_table"] and not ok["neuron_full"]:
                    c["favours_token_table"].append(p["id"])
                if ok["neuron_full"] and not ok["token_table"]:
                    c["favours_neuron"].append(p["id"])
            if ok["neuron_full"] != ok["neuron_int"]:
                c["full_vs_int_differ"].append(p["id"])
        result[fact] = {"match": m, "categories": cats}
        print(f"\n== {fact}: matched token-table patch = {m['matched']}; rewards at batch size 1: " + ", ".join(f"{k} {v['reward']:.3f}" for k, v in m["rows"].items()))
        for name, c in cats.items():
            print(f"   {name:11s} n={c['n']:2d} dropped={c['dropped']} | base {c['base']:2d} token_table {c['token_table']:2d} neuron_full {c['neuron_full']:2d} neuron_int {c['neuron_int']:2d}"
                  f" | favours table {len(c['favours_token_table'])} neuron {len(c['favours_neuron'])} | full≠int {len(c['full_vs_int_differ'])}")
    Path(spec["out"], "report.json").write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["match", "probe", "report"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--fact")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    sys.path.insert(0, str(Path(spec["src"]).resolve()))
    if a.stage == "report":
        report(spec)
    else:
        for fact in ([a.fact] if a.fact else list(spec["facts"])):
            (match if a.stage == "match" else probe)(spec, fact, a.device)


if __name__ == "__main__":
    main()
