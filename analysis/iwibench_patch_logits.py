"""比較兩個 patch 條件在「同一串 token」上的輸出分佈(teacher forcing),當作比逐字比對更靈敏的量測。

貪婪解碼的輸出相同,只代表每一步的第一名沒變;這裡直接比每一步的整個下一個 token 分佈。
對每個題目:序列 = chat template 後的提示 + 某個參考條件實際產生的輸出;兩個條件都在這串序列上算
每個產生位置的 log 機率,然後報告
  KL(A‖B) 的最大值與平均、參考 token 的 log 機率差的最大絕對值、第一名是否處處相同、A 的第一名與第二名的最小差距。

用法(需要 GPU 時用平台的 cap-run):
  python analysis/iwibench_patch_logits.py --spec data/patch-ablation/spec.json --pairs data/patch-ablation/pairs.json --device mps
pairs.json:{"episode 名": [["A 條件", "B 條件", "參考條件"], ...]}
輸出:<out>/<episode>/logits_<A>__<B>.json,並印出依題組彙整的表。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

GROUPS = ("calibration", "targets", "nearby_controls", "general_controls", "override_controls")


def logprobs(target, sequences):
    """sequences: {key: (prompt_ids, cont_ids)};回傳 {key: [位置數, 詞表] 的 float32 log 機率(CPU)}。"""
    out = {}
    for key, (prompt, cont) in sequences.items():
        ids = torch.tensor([prompt + cont], dtype=torch.long, device=target.device)
        with torch.inference_mode():
            logits = target.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits
        # 位置 len(prompt)-1 預測第一個產生的 token,一路到預測「參考輸出之後的下一個 token」
        out[key] = logits[0, len(prompt) - 1:].float().log_softmax(-1).cpu()
    return out


def compare(a, b, cont):
    kl = (a.exp() * (a - b)).sum(-1)
    ref = torch.tensor(cont, dtype=torch.long)
    n = len(cont)
    d_ref = (a[:n].gather(1, ref[:, None]) - b[:n].gather(1, ref[:, None])).abs().max().item() if n else 0.0
    top = a.topk(2, -1).values
    return {"kl_max": kl.max().item(), "kl_mean": kl.mean().item(), "dlogp_ref_max": d_ref,
            "top1_same": bool((a.argmax(-1) == b.argmax(-1)).all()), "margin_min": (top[:, 0] - top[:, 1]).min().item(),
            "positions": int(a.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    pairs = json.loads(Path(args.pairs).read_text())
    sys.path.insert(0, str(Path(spec["src"]).resolve()))
    from iwibench.runtime import NativeTarget

    for e in spec["episodes"]:
        if e["name"] not in pairs:
            continue
        d = Path(spec["out"]) / e["name"]
        ep = json.loads((d / "episode.json").read_text())
        items = [(g, it) for g in GROUPS for it in ep[g]]
        conds = sorted({c for p in pairs[e["name"]] for c in p[:2]})
        refs = sorted({p[2] for p in pairs[e["name"]]})
        outputs = {r: json.loads((d / f"{r}.json").read_text())["outputs"] for r in refs}
        store = {}
        for c in conds:
            target = NativeTarget(ep["model_id"], ep["checkpoint_revision"], device=args.device,
                                  cache_dir=str(Path(spec["cache"]).resolve()))
            if e["conditions"][c]:
                target.apply_patch(e["conditions"][c])
            seqs = {}
            for r in refs:
                for _, it in items:
                    cont = target.tokenizer.encode(outputs[r][it["id"]], add_special_tokens=False)
                    seqs[(r, it["id"])] = (target._prompt_ids(it["prompt"]), cont)
            store[c] = (logprobs(target, seqs), seqs)
            del target
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"{e['name']} {c}: {len(seqs)} 串序列完成", flush=True)
        for a, b, r in pairs[e["name"]]:
            rows = {}
            for g, it in items:
                key = (r, it["id"])
                rows[it["id"]] = {"group": g, **compare(store[a][0][key], store[b][0][key], store[a][1][key][1])}
            (d / f"logits_{a}__{b}.json").write_text(json.dumps({"a": a, "b": b, "ref": r, "items": rows}, indent=1))
            print(f"\n{e['name']}: {a} 對 {b}(序列取自 {r} 的輸出)")
            for g in GROUPS:
                rs = [v for v in rows.values() if v["group"] == g]
                if rs:
                    print(f"   {g:17s} n={len(rs):3d} | KL 最大 {max(v['kl_max'] for v in rs):.2e} 平均 {sum(v['kl_mean'] for v in rs) / len(rs):.2e}"
                          f" | 參考 token 的 Δlogp 最大 {max(v['dlogp_ref_max'] for v in rs):.2e}"
                          f" | 第一名處處相同 {sum(v['top1_same'] for v in rs)}/{len(rs)}"
                          f" | A 的最小領先 {min(v['margin_min'] for v in rs):.3f}", flush=True)


if __name__ == "__main__":
    main()
