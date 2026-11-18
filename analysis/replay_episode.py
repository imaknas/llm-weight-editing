"""從公開的回合紀錄(runs/<回合>/)把修改重建出來,用基準的評分流程評分,和紀錄比對。

做法:開一個新的 IWIBench session(同一個題型與種子;紀錄裡若鎖了張量就照鎖),照紀錄的順序把每一次送到服務的
工具呼叫重送一遍。紀錄裡的 handle 是當時那個 session 的,重放時每次呼叫會得到新的 handle;這裡把新的輸出照
當時給模型看的方式排版(含截斷),依出現順序把舊 handle 對到新 handle,再代入之後的呼叫。全部重送完就提交、評分。

比對的東西:
  - 提示的 SHA-256(確認題目與可寫的張量清單和當時相同)
  - 每一次呼叫成功與否、呼叫後的 revision 與計數器
  - 提交出來的修改檔內容的指紋(紀錄在 meta.json 的 patch.content_sha256)、被改動的參數個數
  - reward 與各組題目的答對數
修改檔逐位元相同,就表示公開紀錄足以重建那一份修改;公開紀錄不含權重數值,重建時的數值來自模型本身。

用法(在專案根目錄;需要 GPU 時用 --service-launcher 指定前置指令):
  python analysis/replay_episode.py runs/lock-low-s2-r1 --out data/replay
"""
import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iwi_editor_api as drv  # noqa: E402

HANDLE = re.compile(r'"handle": "([0-9a-f]{32})"')
STATE = re.compile(r'"counters": \{(.*?)\},\s*"revision": (\d+)', re.S)


def content_sha256(path):
    """修改檔內容的指紋:依張量名稱排序,把名稱、位置、數值的位元組串起來取 SHA-256。
    和檔案本身的雜湊不同,它不受檔頭寫法影響;它也不洩漏任何權重數值。"""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as h:
        sparse = json.loads(h.metadata().get("sparse", "[]"))
    tensors = load_file(str(path))
    digest = hashlib.sha256()
    for e in sorted(sparse, key=lambda x: x["name"]):
        digest.update(e["name"].encode())
        digest.update(tensors[e["indices"]].numpy().tobytes())
        digest.update(tensors[e["values"]].view(dtype=__import__("torch").int16).numpy().tobytes())
    return digest.hexdigest()


def substitute(obj, mapping, missing):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "handle" and isinstance(v, str):
                if v not in mapping:
                    missing.append(v)
                out[k] = mapping.get(v, v)
            else:
                out[k] = substitute(v, mapping, missing)
        return out
    if isinstance(obj, list):
        return [substitute(v, mapping, missing) for v in obj]
    return obj


def state_of(text):
    m = None
    for m in STATE.finditer(text):
        pass
    return (re.sub(r"\s+", "", m.group(1)), int(m.group(2))) if m else None


def replay(record, launcher, out):
    meta = json.loads((record / "meta.json").read_text())
    events = [json.loads(line) for line in (record / "episode.jsonl").read_text().splitlines()]
    calls = [e for e in events if e.get("event") == "tool" and "payload" in e]
    svc = drv.Service(lambda rec: None, launcher, ["--freeze", meta["freeze"]] if meta.get("freeze") else [])
    report = {"record": record.name, "calls": len(calls), "problems": []}
    try:
        svc.start()
        res = svc.op({"op": "create", "family": meta["family"], "seed": meta["seed"], "trace_id": record.name + "-replay"}, timeout=600)
        if not res.get("ok"):
            raise RuntimeError(f"create failed: {res}")
        r = res["result"]
        sid = r["session_id"]
        public = {"objective": r["objective"], "practice": r.get("practice", []), "manifest": r["manifest"],
                  "token_ids": r.get("token_ids", {}), "limits": r.get("limits", {})}
        prompt = drv.INTRO + json.dumps(public, ensure_ascii=False, separators=(",", ":"))
        report["prompt_sha256_same"] = hashlib.sha256(prompt.encode()).hexdigest() == meta.get("prompt_sha256")
        mapping = {}
        for n, e in enumerate(calls, 1):
            missing = []
            payload = substitute(e["payload"], mapping, missing)
            if missing:
                report["problems"].append(f"call {n} ({e['name']}): handle not seen earlier in the record: {missing}")
            res = svc.op({"op": "tool", "session_id": sid, "name": e["name"], "payload": payload})
            ok = bool(res.get("ok"))
            if ok != bool(e.get("ok")):
                report["problems"].append(f"call {n} ({e['name']}): recorded ok={e.get('ok')}, replay ok={ok}: {str(res.get('message', ''))[:200]}")
            if not ok:
                continue
            text = drv.render(res["result"])
            old, new = HANDLE.findall(e.get("content", "")), HANDLE.findall(text)
            if e.get("content", "").startswith("[omitted:"):
                pass  # 公開紀錄沒有留這次讀取的輸出(含權重數值);它的 handle 之後沒有被引用
            elif len(old) != len(new):
                report["problems"].append(f"call {n} ({e['name']}): {len(old)} handles in the record, {len(new)} in the replay")
            mapping.update(zip(old, new))
            was, now = state_of(e.get("content", "")), state_of(text)
            if was and now and was != now:
                report["problems"].append(f"call {n} ({e['name']}): state after the call differs: record {was}, replay {now}")
        sub = svc.op({"op": "submit", "session_id": sid}, timeout=900)
        grade = svc.op({"op": "grade", "session_id": sid}, timeout=1800)
        patch_was = meta.get("patch") or {}
        patch_now = sub.get("result") or {}
        report["patch"] = {k: {"record": patch_was.get(k), "replay": patch_now.get(k)} for k in ("changed_parameters", "tensors")}
        now_digest = content_sha256(patch_now["path"]) if patch_now.get("path") else None
        report["patch"]["content_sha256"] = {"record": patch_was.get("content_sha256"), "replay": now_digest}
        report["patch_identical"] = bool(now_digest) and patch_was.get("content_sha256") == now_digest
        if grade.get("ok"):
            g = grade["result"]
            groups = {k: {x: v[x] for x in ("count", "base_successes", "edited_successes", "retained_successes", "retention_denominator")}
                      for k, v in g["metrics"]["groups"].items()}
            report["reward"] = {"record": meta.get("reward"), "replay": g["reward"]}
            report["groups_same"] = groups == meta.get("groups")
        else:
            report["problems"].append(f"grading failed: {grade}")
    finally:
        try:
            svc.stop()
        except Exception:
            pass
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record.name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    rw = report.get("reward", {})
    print(f"{record.name}: calls {report['calls']} | prompt same {report.get('prompt_sha256_same')} | patch identical {report.get('patch_identical')}"
          f" | reward {rw.get('record')} -> {rw.get('replay')} | groups same {report.get('groups_same')} | problems {len(report['problems'])}", flush=True)
    for p in report["problems"]:
        print("   ", p, flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="回合紀錄的資料夾,例如 runs/lock-low-s2-r1")
    ap.add_argument("--out", default="data/replay")
    ap.add_argument("--service-launcher", default="", help="放在服務指令前面的啟動器;預設直接執行")
    a = ap.parse_args()
    for rec in a.records:
        if (Path(a.out) / f"{Path(rec).name}.json").exists():
            print(f"{Path(rec).name}: already replayed, skipping")
            continue
        replay(Path(rec), shlex.split(a.service_launcher), Path(a.out))
