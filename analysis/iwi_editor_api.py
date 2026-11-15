"""用供應商的 API 當 IWIBench 的編輯者,跑一個回合。

服務端是 iwi_rpc_server.py(IWIBench 的 Service)。這裡只做編輯者端的迴圈,條件照基準:
  - 單一則 user 訊息(固定的一句話 + 公開資訊的 JSON),沒有 system prompt;
  - 四個工具,定義逐字取自公布回合的原始紀錄(analysis/iwi_editor_tools.json);
  - 參數先用基準原始碼裡的 pydantic 類別驗證,再送服務;
  - 工具結果以縮排 2 的 JSON 回給模型,超過約 5,000 token(位元組數 / 4)時只留頭尾各 10,000 位元組
    (公布紀錄裡 harness 的做法);錯誤訊息的格式照公布紀錄;
  - 停止:64 輪、128 次工具呼叫、或一則不含工具呼叫的回應;之後提交並評分。
服務每次最多活 30 分鐘:這裡在輪與輪之間主動重啟並接回,舊的 handle ID 由服務端保持有效。

用法(在專案根目錄;key 由環境變數 OPENAI_API_KEY / ANTHROPIC_API_KEY 提供):
  python analysis/iwi_editor_api.py --provider anthropic --model claude-fable-5-2 --effort high --max-output 32000 \
      --family factual_update --seed 2 --label fable-high-32k-s2 --budget 60 --episode-cap 15
服務預設由這支程式直接啟動(同一個 Python);需要透過其他程式才能取得 GPU 時,用 --service-launcher 指定前置指令。
"""
import argparse
import http.client
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iwi_tool  # noqa: E402

SRC = Path("data/iwibench-src/src/iwibench")
RUNS = Path("data/api-runs")
INTRO = ("Edit the target model's weights to achieve the objective below. "
         "Use the four provided tools. Your final natural-language reply submits "
         "the current weight patch. It cannot change the target's prompt or inference code.\n\n")
# 每百萬 token 的美元價格。動用前對照過官方價目頁(見 docs/editor-conditions-plan.md)。
PRICES = {
    "gpt-6.1-sol": {"in": 2.5, "cached": 0.10, "out": 10.0, "write": 2.5},  # 官方表另列 2.50 的快取寫入價;未命中的輸入一律以它計,偏保守
    "claude-fable-5-2": {"in": 8.0, "cached": 0.20, "out": 40.0, "write": 16.0},  # write = 1 小時快取
}
SERVICE_MAX_AGE = 20 * 60  # 服務上限 30 分鐘;超過這個年齡就在輪與輪之間重啟


# ---------- 基準的請求驗證 ----------

def request_models():
    """從基準原始碼取出四個工具的 pydantic 類別(不匯入整個模組,因為它依賴未安裝的框架)。"""
    text = (SRC / "taskset.py").read_text()
    body = text[text.index("Index = Annotated"):text.index("class EpisodeState")]
    ns = {}
    exec("from typing import Annotated, Literal\n"
         "from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt\n" + body, ns)
    return {"read_weights": ns["ReadWeightsRequest"], "write_weights": ns["WriteWeightsRequest"],
            "run_target": ns["RunTargetRequest"], "inspect_target": ns["InspectTargetRequest"]}


def _num(x):
    """浮點數的寫法照公布紀錄(與 Python 的差別:1e-5 到 1e-4 之間用小數、指數不補零也不帶正號)。"""
    r = repr(x)
    if "e" not in r:
        return r
    mant, exp = r.split("e")
    if int(exp) == -5:
        digits = mant.replace("-", "").replace(".", "")
        return ("-" if x < 0 else "") + "0.0000" + digits
    return f"{mant}e{int(exp)}"


def pretty(o, depth=0):
    """縮排 2 的 JSON,非 ASCII 字元不跳脫;與公布紀錄裡工具輸出的排版逐位元相同。"""
    pad, inner = "  " * depth, "  " * (depth + 1)
    if isinstance(o, dict):
        if not o:
            return "{}"
        return "{\n" + ",\n".join(f"{inner}{json.dumps(k, ensure_ascii=False)}: {pretty(v, depth + 1)}" for k, v in o.items()) + f"\n{pad}}}"
    if isinstance(o, list):
        if not o:
            return "[]"
        return "[\n" + ",\n".join(inner + pretty(v, depth + 1) for v in o) + f"\n{pad}]"
    if isinstance(o, float):
        return _num(o)
    return json.dumps(o, ensure_ascii=False)


def render(result):
    """harness 給模型的工具輸出:縮排 2 的 JSON,太長時只留頭尾。"""
    text = pretty(result)
    raw = text.encode()
    if math.ceil(len(raw) / 4) <= 5000:
        return text
    head, tail = raw[:10000].decode(errors="ignore"), raw[-10000:].decode(errors="ignore")
    return (f"Warning: truncated output (original token count: {math.ceil(len(raw) / 4)})\n"
            f"Total output lines: {text.count(chr(10)) + 1}\n\n"
            f"{head}\n[... {len(raw) - 20000} bytes truncated ...]\n{tail}")


# ---------- 服務 ----------

class Service:
    def __init__(self, log, launcher=(), extra=()):
        """launcher:放在服務指令前面的啟動器(例如需要透過工作排程器才能用 GPU 時);空的就直接執行。"""
        self.proc, self.started, self.session, self.log, self.restarts = None, 0.0, None, log, 0
        self.launcher = list(launcher)
        self.extra = list(extra)  # 附加給服務的參數

    def start(self):
        for name in ("ready", "alive"):
            (iwi_tool.RPC / name).unlink(missing_ok=True)
        for p in list((iwi_tool.RPC / "in").glob("*")) + list((iwi_tool.RPC / "out").glob("*")):
            p.unlink()
        self.proc = subprocess.Popen(
            self.launcher + [sys.executable, "-I", "analysis/iwi_rpc_server.py", "--src", "data/iwibench-src/src", "--cache", "data/hf-cache",
             "--rpc", str(iwi_tool.RPC), "--runs", "data/iwi-runs", "--device", "mps", "--idle-exit", "1750"] + self.extra,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.started = time.time()
        end = time.time() + 240
        while not (iwi_tool.RPC / "ready").exists():
            if time.time() > end or self.proc.poll() is not None:
                raise RuntimeError("target service did not start")
            time.sleep(1)

    def stop(self):
        if self.proc and self.proc.poll() is None and iwi_tool.alive():
            iwi_tool.call({"op": "shutdown"}, timeout=60)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def ensure(self, expect=None):
        """確保服務活著而且夠年輕;必要時重啟並接回。expect = 上次看到的 (revision, counters),接回後必須相同。"""
        if self.proc and self.proc.poll() is None and iwi_tool.alive() and time.time() - self.started < SERVICE_MAX_AGE:
            return
        self.stop()
        self.start()
        if self.session:
            res = iwi_tool.call({"op": "resume", "session_id": self.session}, timeout=600)
            if not res.get("ok"):
                raise RuntimeError(f"resume failed: {res}")
            r = res["result"]
            if expect and (r["revision"], r["counters"]) != expect:
                raise RuntimeError(f"resumed state differs: {r['revision']} {r['counters']} vs {expect}")
            self.session, self.restarts = r["session_id"], self.restarts + 1
            self.log({"event": "service_restart", "session": self.session, "replayed": len(r.get("replayed", []))})

    def op(self, req, timeout=900):
        res = iwi_tool.call(req, timeout=timeout)
        if res.get("error") in ("ServiceNotRunning", "Timeout"):
            raise RuntimeError(f"target service lost during a call: {res}")
        return res


# ---------- 供應商 ----------

def post(url, headers, body, timeout):
    data = json.dumps(body).encode()
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < 5:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"provider HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"provider connection failed: {e}") from None


class OpenAIResponses:
    """Responses API,照基準轉接程式的做法:不存檔、帶回加密的 reasoning 項目並逐輪原樣重放。"""
    def __init__(self, model, effort, max_output, prompt, tools, cache_key):
        self.model, self.effort, self.max_output, self.cache_key = model, effort, max_output, cache_key
        self.inputs = [{"role": "user", "content": prompt}]
        self.tools = [{"type": "function", **t} if "type" not in t else dict(t) for t in tools]

    def call(self):
        body = {"model": self.model, "input": self.inputs, "tools": self.tools, "store": False,
                "include": ["reasoning.encrypted_content"], "reasoning": {"effort": self.effort},
                "max_output_tokens": self.max_output, "prompt_cache_key": self.cache_key}
        key = os.environ["OPENAI_API_KEY"]
        for attempt in range(3):  # 回應讀到一半斷線就整個重送;斷掉的那次供應商可能仍計費,帳本記不到
            try:
                with post("https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body, 900) as r:
                    res = json.load(r)
                break
            except http.client.IncompleteRead:
                if attempt == 2:
                    raise
                time.sleep(5)
        u = res.get("usage") or {}
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
        usage = {"in": u.get("input_tokens", 0) - cached, "cached": cached, "write": 0, "out": u.get("output_tokens", 0),
                 "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        output = res.get("output", [])
        calls = [{"id": i["call_id"], "name": i["name"], "arguments": i["arguments"]} for i in output if i.get("type") == "function_call"]
        text = "".join(p.get("text", "") for i in output if i.get("type") == "message" for p in i.get("content", []) if p.get("type") == "output_text")
        stop = "length" if res.get("status") == "incomplete" else "tool_calls" if calls else "stop"
        if res.get("status") not in ("completed", "incomplete"):
            stop = f"status:{res.get('status')}"
        self.inputs.extend(output)
        return {"calls": calls, "text": text, "stop": stop, "usage": usage, "raw": output}

    def results(self, results):
        for call_id, text, _ in results:
            self.inputs.append({"type": "function_call_output", "call_id": call_id, "output": text})

    def context_bytes(self):
        return len(json.dumps(self.inputs))


class AnthropicMessages:
    """Messages API(串流)。thinking 永遠開著,只設 effort;thinking 區塊原樣回傳;不啟用會換模型的備援。"""
    def __init__(self, model, effort, max_output, prompt, tools):
        self.model, self.effort, self.max_output = model, effort, max_output
        self.messages = [{"role": "user", "content": prompt}]
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    def call(self):
        body = {"model": self.model, "max_tokens": self.max_output, "output_config": {"effort": self.effort},
                "tools": self.tools, "messages": self.messages, "stream": True,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2026-10-21", "Content-Type": "application/json"}
        blocks, partial, usage, stop = {}, {}, {}, None
        with post("https://api.anthropic.com/v1/messages", headers, body, 1800) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                ev = json.loads(line[5:])
                t = ev.get("type")
                if t == "message_start":
                    usage.update(ev["message"].get("usage") or {})
                elif t == "content_block_start":
                    blocks[ev["index"]] = dict(ev["content_block"])
                elif t == "content_block_delta":
                    d, b = ev["delta"], blocks[ev["index"]]
                    if d["type"] == "text_delta":
                        b["text"] = b.get("text", "") + d["text"]
                    elif d["type"] == "input_json_delta":
                        partial[ev["index"]] = partial.get(ev["index"], "") + d["partial_json"]
                    elif d["type"] == "thinking_delta":
                        b["thinking"] = b.get("thinking", "") + d["thinking"]
                    elif d["type"] == "signature_delta":
                        b["signature"] = b.get("signature", "") + d["signature"]
                elif t == "message_delta":
                    stop = ev["delta"].get("stop_reason", stop)
                    usage.update(ev.get("usage") or {})
                elif t == "error":
                    raise RuntimeError(f"provider stream error: {ev}")
        content, bad = [], False
        for i in sorted(blocks):
            b = blocks[i]
            if b.get("type") == "tool_use":
                try:
                    b["input"] = json.loads(partial.get(i, "") or "{}")
                except ValueError:
                    bad = True  # 被截斷的工具呼叫
                    continue
            content.append(b)
        calls = [{"id": b["id"], "name": b["name"], "arguments": b["input"]} for b in content if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        self.messages.append({"role": "assistant", "content": content})
        u = {"in": usage.get("input_tokens", 0), "cached": usage.get("cache_read_input_tokens", 0),
             "write": usage.get("cache_creation_input_tokens", 0), "out": usage.get("output_tokens", 0), "reasoning": None}
        s = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}.get(stop, f"other:{stop}")
        if bad:
            s = "length"
        return {"calls": calls, "text": text, "stop": s, "usage": u, "raw": content}

    def results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": text, **({"is_error": True} if err else {})} for cid, text, err in results]})

    def context_bytes(self):
        return len(json.dumps(self.messages))


# ---------- 記帳 ----------

class Ledger:
    def __init__(self, model, budget, episode_cap, label):
        self.p, self.budget, self.cap, self.label, self.path = PRICES[model], budget, episode_cap, label, RUNS / "ledger.jsonl"
        self.episode = 0.0
        self.total = sum(json.loads(l)["usd"] for l in self.path.read_text().splitlines()) if self.path.exists() else 0.0

    def reserve(self, context_tokens, max_output, new_tokens=None):
        """送出前的保守估計。new_tokens = None:整段上下文都沒命中快取(以寫入價計)。
        給了 new_tokens(上一次呼叫在快取存活時間內):舊的部分以讀取價計,只有新增的部分以寫入價計。輸出一律用到上限。"""
        dear = max(self.p["in"], self.p["write"])
        if new_tokens is None:
            worst = (context_tokens * dear + max_output * self.p["out"]) / 1e6
        else:
            worst = ((context_tokens - new_tokens) * self.p["cached"] + new_tokens * dear + max_output * self.p["out"]) / 1e6
        if self.total + worst > self.budget:
            return f"total budget: spent {self.total:.2f}, next call up to {worst:.2f}, cap {self.budget:.2f}"
        if self.episode + worst > self.cap:
            return f"episode cap: spent {self.episode:.2f}, next call up to {worst:.2f}, cap {self.cap:.2f}"
        return None

    def charge(self, usage, turn):
        usd = (usage["in"] * self.p["in"] + usage["cached"] * self.p["cached"] + usage["write"] * self.p["write"] + usage["out"] * self.p["out"]) / 1e6
        self.total, self.episode = self.total + usd, self.episode + usd
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "label": self.label, "turn": turn, "usage": usage, "usd": round(usd, 6)}) + "\n")
        return usd


# ---------- 接回 ----------

def restore(a, ed, meta, out):
    """由 transcript.jsonl 重建編輯者端的對話與計數(只支援 Messages API 的回合)。回傳 (上次的輸入 token, 上次呼叫時間, 當時的位元組數)。"""
    if a.provider != "anthropic":
        raise RuntimeError("resume is implemented for the Messages API only")
    events = [json.loads(l) for l in (out / "transcript.jsonl").read_text().splitlines()]
    seen, when, i = 0, 0.0, 0
    while i < len(events):
        e = events[i]
        i += 1
        if e.get("event") != "model":
            continue
        ed.messages.append({"role": "assistant", "content": e["output"]})
        u = e["usage"]
        seen, when = u["in"] + u["cached"] + u["write"] + u["out"], e["time"]
        meta["turns"] += 1
        meta["stops"].append(e["stop"])
        meta["max_out"], meta["sum_out"] = max(meta["max_out"], u["out"]), meta["sum_out"] + u["out"]
        ids = [b["id"] for b in e["output"] if b.get("type") == "tool_use"]
        results = []
        while len(results) < len(ids) and i < len(events):
            t = events[i]
            i += 1
            if t.get("event") == "tool":
                results.append((ids[len(results)], t["content"], bool(t.get("invalid")) or not t.get("ok")))
        if len(results) != len(ids):
            raise RuntimeError("transcript ends in the middle of a tool batch; cannot resume")
        meta["tool_calls"] += len(ids)
        if ids:
            ed.results(results)
    return seen, when, ed.context_bytes()


# ---------- 回合 ----------

def run(a):
    out = RUNS / a.label
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "transcript.jsonl").open("a")

    def log(rec):
        logf.write(json.dumps({"time": time.time(), **rec}, ensure_ascii=False) + "\n")
        logf.flush()

    models = request_models()
    tools = json.loads(Path(__file__).with_name("iwi_editor_tools.json").read_text())
    svc = Service(log, shlex.split(a.service_launcher), ["--freeze", a.freeze] if a.freeze else [])
    ledger = Ledger(a.model, a.budget, a.episode_cap, a.label)
    if a.resume:
        ledger.episode = sum(json.loads(l)["usd"] for l in ledger.path.read_text().splitlines() if json.loads(l)["label"] == a.label)
    meta = {"label": a.label, "provider": a.provider, "model": a.model, "effort": a.effort, "max_output": a.max_output,
            "family": a.family, "seed": a.seed, "freeze": a.freeze, "end": None, "turns": 0, "tool_calls": 0, "stops": [], "max_out": 0, "sum_out": 0}
    try:
        if a.resume:
            # 接回被中止的回合:服務用存檔開新 session(舊的已提交過),編輯者端的對話由紀錄重建
            old = json.loads((out / "meta.json").read_text())
            (out / f"meta-before-resume-{int(time.time())}.json").write_text(json.dumps(old, ensure_ascii=False, indent=1))
            state = json.loads((Path("data/iwi-runs") / old["session"] / "state.json").read_text())
            svc.session = old["session"]
            svc.ensure(expect=(state["revision"], state["counters"]))
            last = (state["revision"], state["counters"])
            prompt = (out / "prompt.txt").read_text()
        else:
            svc.start()
            res = svc.op({"op": "create", "family": a.family, "seed": a.seed, "trace_id": a.label}, timeout=600)
            if not res.get("ok"):
                raise RuntimeError(f"create failed: {res}")
            r = res["result"]
            svc.session = r["session_id"]
            public = {"objective": r["objective"], "practice": r.get("practice", []), "manifest": r["manifest"],
                      "token_ids": r.get("token_ids", {}), "limits": r.get("limits", {})}
            prompt = INTRO + json.dumps(public, ensure_ascii=False, separators=(",", ":"))
            (out / "prompt.txt").write_text(prompt)
            last = (0, r["counters"])
        ed = (OpenAIResponses(a.model, a.effort, a.max_output, prompt, tools, a.label) if a.provider == "openai"
              else AnthropicMessages(a.model, a.effort, a.max_output, prompt, tools))
        log({"event": "resume" if a.resume else "start", "session": svc.session, "prompt_bytes": len(prompt.encode())})
        seen_tokens, last_call, seen_bytes = 0, 0.0, 0
        if a.resume:
            seen_tokens, last_call, seen_bytes = restore(a, ed, meta, out)
        while True:
            if meta["turns"] >= 64 or meta["tool_calls"] >= 128:
                meta["end"] = "limit"
                break
            # 上下文的保守估計:上一輪實際的輸入 token + 之後新增內容的位元組數 / 3
            grown = math.ceil(max(0, ed.context_bytes() - seen_bytes) / 2.5) + 2048
            est = seen_tokens + grown if seen_tokens else math.ceil(ed.context_bytes() / 2)
            warm = seen_tokens and time.time() - last_call < a.cache_seconds
            why = ledger.reserve(est, a.max_output, grown if warm else None)
            if why:
                meta["end"] = f"cost_abort: {why}"
                break
            last_call = time.time()
            resp = ed.call()
            seen_bytes = ed.context_bytes()
            meta["turns"] += 1
            u = resp["usage"]
            usd = ledger.charge(u, meta["turns"])
            seen_tokens = u["in"] + u["cached"] + u["write"] + u["out"]
            meta["stops"].append(resp["stop"])
            meta["max_out"], meta["sum_out"] = max(meta["max_out"], u["out"]), meta["sum_out"] + u["out"]
            log({"event": "model", "turn": meta["turns"], "stop": resp["stop"], "usage": u, "usd": usd, "output": resp["raw"]})
            print(f"turn {meta['turns']:2d} stop={resp['stop']:10s} in={u['in']} cached={u['cached']} write={u['write']} out={u['out']}"
                  f" calls={len(resp['calls'])} ${usd:.3f} (episode ${ledger.episode:.2f}, total ${ledger.total:.2f})", flush=True)
            if resp["stop"] not in ("tool_calls", "stop"):
                meta["end"] = f"truncated_or_refused: {resp['stop']}"
                break
            if not resp["calls"]:
                meta["end"] = "final_reply"
                meta["final_text"] = resp["text"]
                break
            results = []
            for c in resp["calls"]:
                meta["tool_calls"] += 1
                name, text, err = c["name"], None, True
                try:
                    args = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]
                    if name not in models:
                        raise ValueError(f"Unknown tool {name}")
                    if not isinstance(args, dict) or set(args) != {"request"}:
                        raise ValueError("arguments must be an object with the single field 'request'")
                    payload = models[name].model_validate(args["request"]).model_dump(mode="json", exclude_none=True)
                except Exception as exc:  # 不合格的呼叫不送到服務
                    text = f"Error executing tool {name}: {exc}"
                    log({"event": "tool", "name": name, "invalid": True, "arguments": c["arguments"], "content": text})
                    results.append((c["id"], text, True))
                    continue
                svc.ensure(expect=last)
                res = svc.op({"op": "tool", "session_id": svc.session, "name": name, "payload": payload})
                if res.get("ok"):
                    text, err = render(res["result"]), False
                    last = (res["result"]["revision"], res["result"]["counters"])
                elif res.get("error") in ("ValueError", "KeyError", "TypeError", "BudgetExceeded", "CalibrationError"):
                    text = f"Error executing tool {name}: Invalid tool request: {res.get('message', '')[:512]}"
                    st = svc.op({"op": "status", "session_id": svc.session})
                    if st.get("ok"):
                        last = (st["result"]["revision"], st["result"]["counters"])
                else:
                    text = f"Error executing tool {name}: GPU service returned HTTP 500"
                    meta["end"] = f"service_failure: {res.get('error')}: {res.get('message', '')[:300]}"
                log({"event": "tool", "name": name, "payload": payload, "ok": bool(res.get("ok")), "content": text})
                results.append((c["id"], text, err))
                if meta["end"]:
                    break
            if meta["end"]:
                break
            ed.results(results)
        # 提交並評分(不論怎麼結束,都記錄目前的 patch 的分數)
        if not str(meta["end"]).startswith("service_failure"):
            svc.ensure(expect=last)
            sub = svc.op({"op": "submit", "session_id": svc.session}, timeout=900)
            grade = svc.op({"op": "grade", "session_id": svc.session}, timeout=1800)
            meta["session"], meta["patch"] = svc.session, sub.get("result")
            if grade.get("ok"):
                g = grade["result"]
                meta["reward"], meta["counters"] = g["reward"], g["counters"]
                meta["groups"] = {k: {x: v[x] for x in ("count", "base_successes", "edited_successes", "retained_successes", "retention_denominator")}
                                  for k, v in g["metrics"]["groups"].items()}
            else:
                meta["grade_error"] = grade
    except Exception as exc:
        meta["end"] = meta["end"] or f"driver_error: {type(exc).__name__}: {exc}"
        raise
    finally:
        meta["usd"], meta["restarts"] = round(ledger.episode, 4), svc.restarts
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        log({"event": "end", "meta": {k: v for k, v in meta.items() if k != "patch"}})
        try:
            svc.stop()
        except Exception:
            pass
    print(json.dumps({k: meta.get(k) for k in ("label", "end", "turns", "tool_calls", "reward", "groups", "usd", "restarts", "max_out", "sum_out")}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", required=True)
    ap.add_argument("--max-output", type=int, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--budget", type=float, required=True, help="所有回合合計的美元上限(讀帳本累計)")
    ap.add_argument("--episode-cap", type=float, required=True, help="這個回合的美元上限")
    ap.add_argument("--service-launcher", default="",
                    help="放在服務指令前面的啟動器,例如 'brunna cap-run --cap gpu --timeout 1800 --';預設直接執行")
    ap.add_argument("--freeze", default="", help="交給服務:額外標成不可寫的張量名稱片段,逗號分隔")
    ap.add_argument("--resume", action="store_true", help="接回先前因花費上限而中止的同名回合")
    ap.add_argument("--cache-seconds", type=float, default=3300, help="上一次呼叫在這段時間內,就假設提示快取還在")
    run(ap.parse_args())
