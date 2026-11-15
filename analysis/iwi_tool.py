"""透過檔案介面呼叫 IWIBench 的目標模型服務(見 iwi_rpc_server.py)。只用標準庫。

編輯者用的四個工具:
  python3 analysis/iwi_tool.py read_weights   '<JSON 參數>'
  python3 analysis/iwi_tool.py write_weights  '<JSON 參數>'
  python3 analysis/iwi_tool.py run_target     '<JSON 參數>'
  python3 analysis/iwi_tool.py inspect_target '<JSON 參數>'
管理用(不給編輯者):create / submit / grade / status / shutdown。
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


def call(req, timeout=900.0):
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
        time.sleep(0.05)
    return {"ok": False, "error": "Timeout", "message": "service did not answer"}


def main():
    op = sys.argv[1]
    arg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if op in TOOLS:
        sid = (RPC / "session").read_text().strip()
        res = call({"op": "tool", "session_id": sid, "name": op, "payload": arg})
    elif op == "create":
        res = call(dict(arg, op="create"))
        if res.get("ok"):
            sid = res["result"]["session_id"]
            (RPC / "session").write_text(sid)
            r = res["result"]
            public = {"objective": r["objective"], "practice": r.get("practice", []), "manifest": r["manifest"],
                      "token_ids": r.get("token_ids", {}), "limits": r.get("limits", {})}
            (RPC / "public.json").write_text(json.dumps(public, ensure_ascii=False, separators=(",", ":")))
            res = {"ok": True, "result": {"session_id": sid, "public_file": str(RPC / "public.json"),
                                          "public_bytes": (RPC / "public.json").stat().st_size, "counters": r.get("counters")}}
    elif op in ("submit", "grade", "status"):
        res = call({"op": op, "session_id": (RPC / "session").read_text().strip()}, timeout=1800.0)
    elif op == "shutdown":
        res = call({"op": "shutdown"})
    else:
        res = {"ok": False, "error": "Usage", "message": "unknown operation"}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
