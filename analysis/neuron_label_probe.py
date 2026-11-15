"""可行性檢查:只憑「神經元標籤」能不能做出在 IWIBench 上得分的內部層修改。

標籤 = 對公開的練習題,在答案位置(chat template 後的最後一個 token)量每個 MLP 神經元的啟動值,
再和一組參考提示上的啟動值比較。它只說「哪個神經元、在這題啟動多少、在別處最多啟動多少」,不說該寫什麼。
修改 = 選出最專一的 K 個神經元,把它們的輸出欄(down_proj 的一欄)加上 (新答案第一個 token 的詞向量 − 舊答案的),
大小只用標籤裡的啟動值換算。強度 α 以殘差範數的倍數掃描,量練習題上新舊答案的 logit 差與參考提示的第一名是否改變。

這裡量的是「這種標籤夠不夠用」,不是任何修改者的行為。隱藏的評分題目不會被讀取。

用法(需要 GPU,用平台的 cap-run):
  python analysis/neuron_label_probe.py --src data/iwibench-src/src --cache data/hf-cache --family factual_update --seed 2 \
      --template <任一個同模型的 patch.safetensors,取其 metadata> --out data/label-probe/s2 --device mps
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GENERIC = [
    "What is the capital of France? Give only the city.", "What is 17 + 25? Give only the integer.",
    "Copy exactly, without quotation marks: lantern_4471", "Name the largest planet in the solar system.",
    "Who wrote the novel Pride and Prejudice? Give only the name.", "Translate 'good morning' into Spanish.",
    "What colour do you get by mixing blue and yellow?", "Write one sentence about the ocean.",
    "Who was the first president of the United States? Give only the name.", "What is the chemical symbol for gold?",
    "List three fruits.", "Who is the CEO of Microsoft? Give only the name.", "What is the opposite of 'hot'?",
    "How many days are in a week? Give only the integer.", "Who painted the Mona Lisa? Give only the name.",
    "What language is spoken in Brazil? Give only the language.", "Explain in one sentence what a database is.",
    "Who directed the film Jurassic Park? Give only the name.", "What is the square root of 81? Give only the integer.",
    "Name the author of the Harry Potter books. Give only the name.", "Say hello.",
    "Who is the president of France? Give only the name.", "What is the tallest mountain on Earth?",
    "Who founded Amazon? Give only the name.",
]


def near(practice, subject, relation, date):
    """由練習題照樣板換掉主體、屬性或日期;不讀任何隱藏題目。"""
    out = [practice.replace(subject, s) for s in ("Australia", "India", "Japan", "the United Kingdom", "New Zealand", "Germany")]
    out += [practice.replace(relation, r) for r in ("capital city", "governor general", "head of state", "currency", "largest city")]
    out += [practice.replace(date, d) for d in ("2019-03-01", "2010-03-01")]
    return [p for p in out if p != practice]


@torch.inference_mode()
def measure(target, layers, prompts):
    """回傳 acts[提示數, 層數, 神經元數](float32, CPU)、最後位置的 logits、各層 MLP 輸入的殘差範數。"""
    grabbed = {}
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(
        lambda _m, inp, k=k: grabbed.__setitem__(k, inp[0][0, -1].float().cpu())) for k, layer in enumerate(layers)]
    acts, logits = [], []
    try:
        for p in prompts:
            ids = torch.tensor([target._prompt_ids(p)], dtype=torch.long, device=target.device)
            out = target.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True,
                               output_hidden_states=True)
            acts.append(torch.stack([grabbed[k] for k in range(len(layers))]))
            logits.append(out.logits[0, -1].float().cpu())
            norms = torch.stack([h[0, -1].float().norm().cpu() for h in out.hidden_states])
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(acts), torch.stack(logits), norms


def main():
    ap = argparse.ArgumentParser()
    for a in ("--src", "--cache", "--family", "--template", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-layer", type=int, default=16)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from iwibench import families
    from iwibench.runtime import NativeTarget

    ep = families.episode(args.family, args.seed, "development", "google/gemma-3-4b-it")
    par = ep["parameters"]
    practice = ep["public_practice"][0]["prompt"]
    print("parameters:", json.dumps(par, ensure_ascii=False)[:600])
    print("practice:", practice, flush=True)
    subject, relation, date = par["entity"], par["relation"], par["reference_date"]
    new, old = par["replacement"], par["stale_candidates"][0]
    target = NativeTarget(ep["model_id"], ep["checkpoint_revision"], device=args.device, cache_dir=str(Path(args.cache).resolve()))
    tok = target.tokenizer
    t_new, t_old = tok.encode(new, add_special_tokens=False)[0], tok.encode(old, add_special_tokens=False)[0]
    layers = target._layers()
    refsets = {"G": GENERIC, "N": GENERIC + near(practice, subject, relation, date)}
    prompts = [practice] + refsets["N"]
    acts, logits, norms = measure(target, layers, prompts)
    base_top = logits.argmax(-1)
    print("base practice top:", repr(tok.decode(base_top[0])), "margin new-old:", (logits[0, t_new] - logits[0, t_old]).item(), flush=True)
    emb = dict(target.model.named_parameters(remove_duplicate=False))["model.language_model.embed_tokens.weight"]
    d = (emb[t_new] - emb[t_old]).float().cpu()
    d = d / d.norm()
    with safe_open(args.template, "pt") as h:
        meta = {k: v for k, v in h.metadata().items() if k != "sparse"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = acts[0]                                             # [層, 神經元]
    report = {"practice": practice, "t_new": t_new, "t_old": t_old, "variants": {}}
    for rname, refs in refsets.items():
        r = acts[1:1 + len(refs)].abs().amax(0)             # 參考提示上的最大絕對啟動值
        rho = r / a.abs().clamp_min(1e-6)
        ok = a.abs() >= a.abs().flatten().kthvalue(int(a.numel() * 0.99)).values     # 只看練習題上啟動值在前 1% 的
        rho = torch.where(ok, rho, torch.full_like(rho, float("inf")))
        rho[:args.min_layer] = float("inf")
        for K in (1, 8):
            best = min(range(args.min_layer, len(layers)), key=lambda L: rho[L].topk(K, largest=False).values.mean().item())
            idx = rho[best].topk(K, largest=False).indices.tolist()
            label = [{"layer": best, "neuron": i, "act_practice": round(a[best, i].item(), 3), "max_abs_act_reference": round(r[best, i].item(), 3)} for i in idx]
            name = f"{rname}-K{K}"
            print(f"\n== {name}: layer {best}", json.dumps(label), flush=True)
            w = layers[best].mlp.down_proj.weight
            orig = w[:, idx].clone()
            rows = []
            for f in (0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0):
                alpha = f * norms[best + 1].item()
                for i, col in zip(idx, range(K)):
                    w[:, i] = (orig[:, col].float() + (alpha / (K * a[best, i].item())) * d.to(w.device)).to(w.dtype)
                _, lg, _ = measure(target, layers, prompts)
                margin = (lg[0, t_new] - lg[0, t_old]).item()
                changed = int((lg.argmax(-1)[1:] != base_top[1:]).sum())
                top = repr(tok.decode(lg[0].argmax()))
                rows.append({"f": f, "margin": margin, "top": top, "refs_changed": changed})
                print(f"   f={f:<5} margin {margin:8.2f} top {top:12s} refs top-1 changed {changed}/{len(prompts) - 1}", flush=True)
                if top.strip("'\" ") == tok.decode(t_new).strip() and not any(x.get("saved") for x in rows[:-1]):
                    n_col = w.shape[1]
                    tensors, sparse = {}, []
                    flat = torch.cat([torch.arange(w.shape[0]) * n_col + i for i in sorted(idx)]).sort()
                    vals = w.detach().cpu().reshape(-1)[flat.values]
                    tensors["__iwi_indices_0"], tensors["__iwi_values_0"] = flat.values.to(torch.int64), vals.to(torch.bfloat16)
                    sparse.append({"name": f"model.language_model.layers.{best}.mlp.down_proj.weight", "indices": "__iwi_indices_0", "values": "__iwi_values_0"})
                    save_file(tensors, str(out / f"{name}.safetensors"), metadata={**meta, "sparse": json.dumps(sparse, separators=(",", ":"))})
                    rows[-1]["saved"] = True
            w[:, idx] = orig
            report["variants"][name] = {"layer": best, "label": label, "scan": rows}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
