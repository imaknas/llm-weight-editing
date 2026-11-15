"""透過檔案介面呼叫 IWIBench 的目標模型服務(見 iwi_rpc_server.py)。只用標準庫。

編輯者用的四個工具:
  python3 analysis/iwi_tool.py read_weights   '<JSON 參數>'
  python3 analysis/iwi_tool.py write_weights  '<JSON 參數>'
  python3 analysis/iwi_tool.py run_target     '<JSON 參數>'
  python3 analysis/iwi_tool.py inspect_target '<JSON 參數>'
管理用(不給編輯者):create / resume / submit / grade / status / shutdown。
create 與 resume 會把公開資訊寫成 <rpc>/public.json(目標、練習題、token id、預算、介面說明)與
<rpc>/tensors.txt(每行一個張量:名稱、形狀、可否寫入)。
目前的 session 記在 <rpc>/session;rpc 目錄預設 data/iwi-rpc,可用環境變數 IWI_RPC 改。
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

RPC = Path(os.environ.get("IWI_RPC", "data/iwi-rpc"))
TOOLS = ("read_weights", "write_weights", "run_target", "inspect_target")


def alive():
    try:
        return time.time() - float((RPC / "alive").read_text()) < 20.0
    except (OSError, ValueError):
        return False


DEAD = {"ok": False, "error": "ServiceNotRunning", "message": "the target service is not running; stop and report this"}


def call(req, timeout=900.0):
    if not alive():
        return DEAD
    rid = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    tmp = RPC / "in" / f"{rid}.tmp"
    tmp.write_text(json.dumps(req))
    tmp.replace(RPC / "in" / f"{rid}.json")
    out = RPC / "out" / f"{rid}.json"
    end = time.time() + timeout
    while time.time() < end:
        if out.exists():
            data = json.loads(out.read_text())
            out.unlink()
            return data
        if not alive():
            return DEAD
        time.sleep(0.05)
    return {"ok": False, "error": "Timeout", "message": "service did not answer"}


def main():
    op = sys.argv[1]
    arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if op in TOOLS:
        sid = (RPC / "session").read_text().strip()
        res = call({"op": "tool", "session_id": sid, "name": op, "payload": arg})
    elif op in ("create", "resume"):
        res = call(dict(arg, op="create") if op == "create" else {"op": "resume", "session_id": (RPC / "session").read_text().strip()})
        if res.get("ok"):
            sid = res["result"]["session_id"]
            (RPC / "session").write_text(sid)
            r = res["result"]
            manifest = dict(r["manifest"])
            tensors = manifest.pop("tensors")
            public = {"objective": r["objective"], "practice": r.get("practice", []), "manifest": manifest,
                      "token_ids": r.get("token_ids", {}), "limits": r.get("limits", {})}
            (RPC / "public.json").write_text(json.dumps(public, ensure_ascii=False, indent=1))
            (RPC / "tensors.txt").write_text("".join(
                f"{t['name']}\t{t['shape']}\t{'writable' if t['writable'] else 'read-only'}"
                + (f"\taliases: {', '.join(x for x in t['aliases'] if x != t['name'])}" if len(t["aliases"]) > 1 else "") + "\n"
                for t in tensors))
            res = {"ok": True, "result": {"session_id": sid, "public_file": str(RPC / "public.json"),
                                          "tensors_file": str(RPC / "tensors.txt"), "counters": r.get("counters"),
                                          "revision": r.get("revision", 0), "resumed_from": r.get("resumed_from")}}
    elif op in ("submit", "grade", "status"):
        res = call({"op": op, "session_id": (RPC / "session").read_text().strip()}, timeout=1800.0)
    elif op == "shutdown":
        res = call({"op": "shutdown"})
    else:
        res = {"ok": False, "error": "Usage", "message": "unknown operation"}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
