"""IWIBench 的目標模型服務,改用「專案資料夾裡的請求檔」當介面(不經網路)。

IWIBench 原本用 HTTP 服務持有被修改的模型;這裡直接使用它的 Service 類別(工具的行為、預算的計數、
patch 的產生與評分都是它的程式),只把傳輸換成檔案,讓沒有網路的 GPU 執行環境也能被另一個程序驅動。
唯一改動的行為:評分時的裝置由參數指定(原本寫死 cuda)。

請求:<rpc>/in/<id>.json = {"op": "create"|"tool"|"submit"|"grade"|"status"|"shutdown", ...}
回應:<rpc>/out/<id>.json = {"ok": true, "result": ...} 或 {"ok": false, "error": 類別, "message": 訊息}

用法(需 GPU 時經平台的 cap-run):
  python analysis/iwi_rpc_server.py --src data/iwibench-src/src --cache data/hf-cache --rpc data/iwi-rpc \
      --runs data/iwi-runs --model google/gemma-3-4b-it --device mps
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--idle-exit", type=float, default=3600.0, help="這麼多秒沒有請求就結束")
    a = ap.parse_args()
    src = str(Path(a.src).resolve())
    sys.path.insert(0, src)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONPATH=src)
    # IWIBench 的 service.py 在最上面引用了 fastapi(只有它的 HTTP 包裝會用到)。這裡不走 HTTP,
    # 而且 fastapi 的安裝檔含有這個執行環境不允許建立的資料夾,所以放一個只含 Request 名稱的替身。
    # 替身只在匯入 service 的那一刻存在,之後移除,免得其他套件以為真的裝了 fastapi。
    import types
    sys.modules["fastapi"] = types.SimpleNamespace(Request=object)
    from iwibench.service import Service
    del sys.modules["fastapi"]
    from iwibench.config import MODELS
    from iwibench.runtime import NativeTarget

    cache = str(Path(a.cache).resolve())

    class FileService(Service):
        def _fresh(self, s, items, patch_path):  # 與原版相同,只有 device 不寫死
            payload = {"model_id": self.runtime.model_id, "revision": self.runtime.revision,
                       "patch_path": patch_path, "prompts": [x["prompt"] for x in items],
                       "patch_sha256": s.patch["sha256"] if patch_path else None,
                       "max_new_tokens": self.limits.output_tokens,
                       "seeds": [s.episode.get("test_seed", 0) + i for i in range(len(items))],
                       "device": a.device, "cache_dir": self.cache_dir}
            path = self.directory / s.id
            request = path / ("grade-edited.json" if patch_path else "grade-base.json")
            request.write_text(json.dumps(payload))
            completed = subprocess.run([sys.executable, "-m", "iwibench.grading_worker", str(request)],
                                       capture_output=True, text=True, check=False)
            (path / (request.stem + ".stderr")).write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Fresh-process grading failed; see {path / (request.stem + '.stderr')}")
            data = json.loads(completed.stdout)
            (path / (request.stem + ".output.json")).write_text(completed.stdout)
            generations = data["generations"]
            if len(generations) != len(items):
                raise RuntimeError("Grading output count mismatch")
            return {item["id"]: result["text"] for item, result in zip(items, generations)}

    runtime = NativeTarget(a.model, MODELS[a.model], device=a.device, cache_dir=cache)
    service = FileService(runtime, a.runs, cache_dir=cache)
    rpc = Path(a.rpc)
    (rpc / "in").mkdir(parents=True, exist_ok=True)
    (rpc / "out").mkdir(parents=True, exist_ok=True)
    (rpc / "ready").write_text(json.dumps({"model": a.model, "device": a.device, "pid": os.getpid(), "time": time.time()}))
    print("ready", flush=True)
    last = time.time()
    while time.time() - last < a.idle_exit:
        pending = sorted(p for p in (rpc / "in").glob("*.json"))
        if not pending:
            time.sleep(0.1)
            continue
        for p in pending:
            last = time.time()
            try:
                req = json.loads(p.read_text())
            except ValueError:
                time.sleep(0.05)  # 還沒寫完
                continue
            p.unlink()
            op = req.get("op")
            try:
                if op == "create":
                    res = service.create(req["family"], a.model, req.get("seed", 0), "development", req.get("trace_id", ""))
                elif op == "tool":
                    res = service.call(req["session_id"], req["name"], req["payload"])
                elif op == "submit":
                    res = service.submit(req["session_id"])
                elif op == "grade":
                    res = service.grade(req["session_id"])
                elif op == "status":
                    st = service.status(req["session_id"])
                    res = {k: st[k] for k in ("counters", "revision", "submitted", "execution_failure")}
                elif op == "shutdown":
                    res = "bye"
                else:
                    raise ValueError("unknown op")
                out = {"ok": True, "result": res}
            except Exception as exc:  # 工具的錯誤照原樣回給呼叫端
                out = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:2000]}
            tmp = rpc / "out" / (p.stem + ".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False, allow_nan=True))
            tmp.replace(rpc / "out" / p.name)
            if op == "shutdown":
                (rpc / "ready").unlink(missing_ok=True)
                return
    (rpc / "ready").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
