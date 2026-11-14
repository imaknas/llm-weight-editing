"""把一個 IWIBench 的稀疏 patch 依張量拆成子集,寫成同樣格式的新 patch。

IWIBench 的 patch 是「取代」而非增量:每個張量一筆 (位置, 新值)。保留某幾筆、丟掉其餘,
得到的 patch 套用後,被保留的張量與原 patch 套用後完全相同,其餘張量維持原始 checkpoint 的值。

產生的條件:
  emb             只保留詞向量矩陣(embed_tokens;它與輸出層共用)
  int             只保留其餘(內部層)的張量
  full-rewritten  全部保留,但經過同一個改寫程式(用來確認改寫本身不改變結果)
  int-<標籤>      (選用)內部層依層號分組,各自單獨保留

每個輸出都會重新讀回檢查:沒有非稀疏張量、子集不為空、emb 與 int 不相交且聯集等於原 patch、
每一筆的位置與數值與原 patch 逐位元相同。

用法:python analysis/iwibench_patch_split.py <原 patch.safetensors> <輸出目錄> [--per-layer]
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def read(path):
    with safe_open(str(path), "pt") as h:
        meta = h.metadata()
    tensors = load_file(str(path))
    sparse = json.loads(meta.get("sparse", "[]"))
    referenced = {e[k] for e in sparse for k in ("indices", "values")}
    extra = set(tensors) - referenced
    if extra:
        raise ValueError(f"patch 含非稀疏張量,這個工具不處理:{sorted(extra)[:3]}")
    if not sparse:
        raise ValueError("patch 是空的")
    entries = {e["name"]: (tensors[e["indices"]], tensors[e["values"]]) for e in sparse}
    if len(entries) != len(sparse):
        raise ValueError("同一個張量出現兩次")
    return meta, entries


def write(path, meta, entries):
    if not entries:
        raise ValueError(f"{path}: 子集是空的")
    tensors, sparse = {}, []
    for k, (name, (idx, val)) in enumerate(entries.items()):
        assert idx.dtype == torch.int64 and val.dtype == torch.bfloat16 and idx.numel() == val.numel()
        tensors[f"__iwi_indices_{k}"], tensors[f"__iwi_values_{k}"] = idx, val
        sparse.append({"name": name, "indices": f"__iwi_indices_{k}", "values": f"__iwi_values_{k}"})
    out_meta = {k: v for k, v in meta.items() if k != "sparse"}
    out_meta["sparse"] = json.dumps(sparse, separators=(",", ":"))
    save_file(tensors, str(path), metadata=out_meta)


def is_embedding(name):
    return name.endswith("embed_tokens.weight") or name == "lm_head.weight"


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[n][0], b[n][0]) and torch.equal(a[n][1].view(torch.int16), b[n][1].view(torch.int16)) for n in a)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    per_layer = "--per-layer" in sys.argv
    meta, entries = read(src)
    emb = {n: v for n, v in entries.items() if is_embedding(n)}
    internal = {n: v for n, v in entries.items() if not is_embedding(n)}
    if not emb or not internal:
        raise ValueError("這個 patch 不是「詞向量 + 內部層」的混合")
    out.mkdir(parents=True, exist_ok=True)
    subsets = {"emb": emb, "int": internal, "full-rewritten": entries}
    if per_layer:
        for layer in sorted({layer_of(n) for n in internal}):
            subsets[f"int-L{layer}"] = {n: v for n, v in internal.items() if layer_of(n) == layer}
    for cond, sub in subsets.items():
        write(out / f"{cond}.safetensors", meta, sub)
        _, back = read(out / f"{cond}.safetensors")
        assert same(back, sub), cond
    _, e2 = read(out / "emb.safetensors")
    _, i2 = read(out / "int.safetensors")
    assert not set(e2) & set(i2) and same({**e2, **i2}, entries)
    _, f2 = read(out / "full-rewritten.safetensors")
    assert same(f2, entries)
    for cond, sub in subsets.items():
        n = sum(v[0].numel() for v in sub.values())
        print(f"{cond:16s} {len(sub)} 個張量,{n} 個位置:{', '.join(x.replace('model.language_model.', '') for x in sub)}")


if __name__ == "__main__":
    main()
