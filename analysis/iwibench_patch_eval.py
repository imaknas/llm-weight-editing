"""用 IWIBench 自己的評分流程,在本機評估一個 episode 在若干個 patch 條件下的結果。

每個條件都在全新的程序裡:載入原始 checkpoint → (套用 patch)→ 對同一批題目貪婪解碼。這一步直接呼叫
IWIBench 的 grading_worker.run,評分用它的 scoring.score_episode;這裡只負責組題目、分程序、存檔、比對。
題目由 iwibench.families.episode 從種子決定性地產生,題目順序與種子的給法照它的 scripts/replay_patch.py。

用法(在專案根目錄;模型需已下載到 --cache;需要 GPU 時用平台的 cap-run 執行):
  python analysis/iwibench_patch_eval.py run --spec data/patch-ablation/spec.json --device mps
  python analysis/iwibench_patch_eval.py report --spec data/patch-ablation/spec.json

spec.json:{"src": IWIBench 的 src 目錄, "cache": HF 快取, "out": 輸出目錄,
            "episodes": [{"name", "family", "seed", "model", "published": {...可省略},
                          "conditions": {"條件名": patch 檔路徑或 null}}]}
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GROUPS = ("calibration", "targets", "nearby_controls", "general_controls", "override_controls")
SCORED = ("targets", "nearby_controls", "general_controls", "override_controls")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_iwibench(src):
    sys.path.insert(0, str(Path(src).resolve()))
    from iwibench import families, scoring  # noqa: E402
    return families, scoring


def worker(src, request_path):
    """子程序:只做一件事,呼叫 IWIBench 的 grading worker。"""
    sys.path.insert(0, str(Path(src).resolve()))
    from iwibench.grading_worker import run
    json.dump(run(json.loads(Path(request_path).read_text())), sys.stdout, allow_nan=False)


def items_of(ep):
    return [(g, item) for g in GROUPS for item in ep[g]]


def run(spec, device):
    families, _ = load_iwibench(spec["src"])
    out = Path(spec["out"])
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    for e in spec["episodes"]:
        ep = families.episode(e["family"], e["seed"], "development", e["model"])
        items = items_of(ep)
        d = out / e["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode.json").write_text(json.dumps(ep, ensure_ascii=False, indent=1))
        for cond, patch in e["conditions"].items():
            done = d / f"{cond}.json"
            if done.exists():
                continue
            req = {"model_id": ep["model_id"], "revision": ep["checkpoint_revision"], "device": device,
                   "cache_dir": str(Path(spec["cache"]).resolve()),
                   "prompts": [it["prompt"] for _, it in items], "max_new_tokens": 256,
                   "seeds": [ep["test_seed"] + i for i in range(len(items))]}
            if patch:
                req.update(patch_path=str(Path(patch).resolve()), patch_sha256=sha256(patch))
            rp = d / f"{cond}.request.json"
            rp.write_text(json.dumps(req))
            t = time.time()
            r = subprocess.run([sys.executable, "-I", __file__, "worker", spec["src"], str(rp)],
                               capture_output=True, text=True, env=env)
            if r.returncode:
                (d / f"{cond}.stderr").write_text(r.stderr)
                print(f"{e['name']} {cond}: 失敗,見 {d / (cond + '.stderr')}", flush=True)
                continue
            res = json.loads(r.stdout)
            gens = res["generations"]
            assert len(gens) == len(items)
            done.write_text(json.dumps({"device": device, "seconds": round(time.time() - t, 1), "patch": patch,
                                        "patch_metadata": res.get("patch_metadata"),
                                        "outputs": {it["id"]: g["text"] for (_, it), g in zip(items, gens)},
                                        "tokens": {it["id"]: [g["input_tokens"], g["output_tokens"]]
                                                   for (_, it), g in zip(items, gens)}}, ensure_ascii=False, indent=1))
            print(f"{e['name']} {cond}: {len(gens)} 題,{time.time() - t:.0f} 秒", flush=True)


def report(spec):
    _, scoring = load_iwibench(spec["src"])
    out = Path(spec["out"])
    for e in spec["episodes"]:
        d = out / e["name"]
        ep = json.loads((d / "episode.json").read_text())
        base_path = d / "base.json"
        if not base_path.exists():
            print(e["name"], "沒有 base 的輸出")
            continue
        base = json.loads(base_path.read_text())["outputs"]
        print(f"\n== {e['name']}  ({e['family']} seed {e['seed']})")
        pub = e.get("published")
        if pub:
            print("   公布:", pub)
        for cond in e["conditions"]:
            p = d / f"{cond}.json"
            if not p.exists():
                print(f"   {cond:16s} 尚無輸出")
                continue
            edited = json.loads(p.read_text())["outputs"]
            try:
                m = scoring.score_episode(ep, base, edited)
            except Exception as exc:  # CalibrationError 等:不給分數,只報告原因
                print(f"   {cond:16s} 無法評分:{type(exc).__name__}: {exc}")
                continue
            g = m["groups"]
            ctrl = [g[k] for k in ("nearby_controls", "general_controls", "override_controls")]
            same = sum(base[i["id"]] == edited[i["id"]] for grp in SCORED for i in ep[grp])
            total = sum(len(ep[grp]) for grp in SCORED)
            print(f"   {cond:16s} 目標 {g['targets']['base_successes']}→{g['targets']['edited_successes']}/{g['targets']['count']}"
                  f" | 對照保留 {m['controls_retained']}/{m['control_retention_denominator']}"
                  f" (nearby {ctrl[0]['retained_successes']}/{ctrl[0]['retention_denominator']},"
                  f" general {ctrl[1]['retained_successes']}/{ctrl[1]['retention_denominator']},"
                  f" override {ctrl[2]['retained_successes']}/{ctrl[2]['retention_denominator']})"
                  f" | spillover {m['local_spillover']} override_acc {m['override_accuracy']}"
                  f" | reward {m['reward']:.4f} | 與 base 逐字相同的輸出 {same}/{total}")


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["run", "report"])
        ap.add_argument("--spec", required=True)
        ap.add_argument("--device", default="mps")
        a = ap.parse_args()
        spec = json.loads(Path(a.spec).read_text())
        run(spec, a.device) if a.cmd == "run" else report(spec)
