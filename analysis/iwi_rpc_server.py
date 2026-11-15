"""IWIBench 的目標模型服務,改用「專案資料夾裡的請求檔」當介面(不經網路)。

IWIBench 原本用 HTTP 服務持有被修改的模型;這裡直接使用它的 Service 類別(工具的行為、預算的計數、
patch 的產生與評分都是它的程式),只把傳輸換成檔案,讓沒有網路的 GPU 執行環境也能被另一個程序驅動。
唯一改動的行為:評分時的裝置由參數指定(原本寫死 cuda)。

請求:<rpc>/in/<id>.json = {"op": "create"|"resume"|"tool"|"submit"|"grade"|"status"|"shutdown", ...}
回應:<rpc>/out/<id>.json = {"ok": true, "result": ...} 或 {"ok": false, "error": 類別, "message": 訊息}

回合的狀態會留在磁碟上,服務被停掉也能接回:每次工具呼叫後寫 <runs>/<sid>/state.json(計數器與版本),
每次成功寫入後用 IWIBench 自己的 submit 程式對回合的副本產生 <runs>/<sid>-snapshot/patch.safetensors。
每次成功的讀取 / 檢查所發出的 handle ID 記在 <runs>/<sid>/handles.jsonl。
"resume" 用同一題開新回合,套回那份 patch、計數器與操作紀錄;最後一次成功寫入之後的讀取與檢查照原樣重做(不計額度),
並讓原本的 handle ID 指向重做出來的數值,所以呼叫端手上的舊 ID 繼續有效。服務活著時每 2 秒更新 <rpc>/alive,呼叫端據此判斷服務是否還在。

用法(需 GPU 時經平台的 cap-run):
  python analysis/iwi_rpc_server.py --src data/iwibench-src/src --cache data/hf-cache --rpc data/iwi-rpc \
      --runs data/iwi-runs --model google/gemma-3-4b-it --device mps
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def handle_ids(obj):
    """依走訪順序收集結果裡所有 "handle" 欄位的值。"""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "handle" and isinstance(v, str):
                out.append(v)
            else:
                out.extend(handle_ids(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(handle_ids(v))
    return out


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

        def snapshot(self, sid):
            """把目前的修改寫成 patch 檔,不動到回合本身:對回合的副本呼叫原本的 submit。"""
            with self.lock:
                s = self.sessions[sid]
                key = sid + "-snapshot"
                self.sessions[key] = dataclasses.replace(s, id=key, submitted=False, replacements=dict(s.replacements),
                                                         handles={}, patch={})
                try:
                    self.submit(key)
                finally:
                    del self.sessions[key]

        def save_state(self, sid):
            s = self.sessions[sid]
            (self.directory / sid / "state.json").write_text(json.dumps(
                {"counters": s.counters.as_dict(), "revision": s.revision, "family": s.episode["family"],
                 "seed": s.episode["parameter_seed"], "trace_id": s.trace_id}))

        def resume(self, old_sid):
            import torch
            from safetensors import safe_open
            from safetensors.torch import load_file
            from iwibench.budgets import Counters
            old = self.directory / old_sid
            state = json.loads((old / "state.json").read_text())
            res = self.create(state["family"], self.runtime.model_id, state["seed"], "development", state["trace_id"])
            s = self.sessions[res["session_id"]]
            patch = self.directory / (old_sid + "-snapshot") / "patch.safetensors"
            if state["revision"]:
                with safe_open(str(patch), "pt") as h:
                    meta = h.metadata()
                tensors = load_file(str(patch))
                sparse = json.loads(meta.get("sparse", "[]"))
                self.store.activate({})
                def original(name):
                    if name not in self.store.originals:
                        self.store.originals[name] = self.store.params[name].detach().cpu().clone()
                    return self.store.originals[name]
                for e in sparse:
                    value = original(e["name"]).clone()
                    value.reshape(-1)[tensors[e["indices"]]] = tensors[e["values"]]
                    s.replacements[e["name"]] = value
                used = {e[k] for e in sparse for k in ("indices", "values")}
                for name, value in tensors.items():
                    if name not in used:
                        original(name)
                        s.replacements[name] = value.clone()
            s.counters = Counters(**state["counters"])
            s.revision = state["revision"]
            new = self.directory / s.id
            new.mkdir(exist_ok=True)
            lines = (old / "trajectory.jsonl").read_text() if (old / "trajectory.jsonl").exists() else ""
            (new / "trajectory.jsonl").write_text(lines)
            s.trajectory = [json.loads(l) for l in lines.splitlines()]
            # handle 只活在記憶體裡。最後一次成功寫入之後的讀取與檢查是決定性的,照原樣、照原順序重做一次(不計額度),
            # 再讓原本的 handle ID 指向重做出來的數值:呼叫端手上的舊 ID 繼續有效,由 handle 衍生的 handle 也一樣。
            old_handles = {}
            if (old / "handles.jsonl").exists():
                (new / "handles.jsonl").write_text((old / "handles.jsonl").read_text())
                old_handles = {r["n"]: r["handles"] for r in map(json.loads, (old / "handles.jsonl").read_text().splitlines())}
            last_write = max((i for i, e in enumerate(s.trajectory) if e["tool"] == "write_weights" and e.get("ok")), default=-1)
            replayed = []
            self.store.activate(s.replacements)
            for n in range(last_write + 1, len(s.trajectory)):
                e = s.trajectory[n]
                if e.get("ok") and e["tool"] in ("read_weights", "inspect_target"):
                    kept, s.counters = s.counters, Counters()  # 空的計數器:重做不佔預算,也不會被上限擋下
                    try:
                        fn = self.store.read if e["tool"] == "read_weights" else self._inspect
                        result = fn(s, **e["request"])
                    finally:
                        s.counters = kept
                    fresh = handle_ids(result)
                    was = old_handles.get(n, [])
                    if len(was) != len(fresh):
                        raise RuntimeError(f"resume: call {n} returned {len(fresh)} handles, originally {len(was)}")
                    for a, b in zip(was, fresh):
                        s.handles[a] = s.handles[b]
                    replayed.append({"tool": e["tool"], "request": e["request"], "result": result})
            if state["revision"]:
                self.snapshot(s.id)
            self.save_state(s.id)
            return {**res, "counters": s.counters.as_dict(), "revision": s.revision, "resumed_from": old_sid, "replayed": replayed}

        def log_handles(self, sid, result):
            """記下這次呼叫回傳的 handle ID(依出現順序),接回時用來把舊 ID 對到重做出來的數值。"""
            s = self.sessions[sid]
            with (self.directory / sid / "handles.jsonl").open("a") as f:
                f.write(json.dumps({"n": len(s.trajectory) - 1, "handles": handle_ids(result)}) + "\n")

    runtime = NativeTarget(a.model, MODELS[a.model], device=a.device, cache_dir=cache)
    service = FileService(runtime, a.runs, cache_dir=cache)
    rpc = Path(a.rpc)
    (rpc / "in").mkdir(parents=True, exist_ok=True)
    (rpc / "out").mkdir(parents=True, exist_ok=True)
    def beat():
        while True:
            (rpc / "alive.tmp").write_text(str(time.time()))
            (rpc / "alive.tmp").replace(rpc / "alive")
            time.sleep(2.0)
    threading.Thread(target=beat, daemon=True).start()
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
                    service.save_state(res["session_id"])
                elif op == "resume":
                    res = service.resume(req["session_id"])
                elif op == "tool":
                    try:
                        res = service.call(req["session_id"], req["name"], req["payload"])
                        service.log_handles(req["session_id"], res["result"])
                        if req["name"] == "write_weights":
                            service.snapshot(req["session_id"])
                    finally:
                        if req["session_id"] in service.sessions:
                            service.save_state(req["session_id"])
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
