"""把 API 回合的紀錄(data/api-runs/<回合>/)匯出成可公開的版本(runs/<回合>/)。

公開的內容:每一輪模型的文字回覆與工具呼叫(含完整的參數)、每次工具呼叫回給模型的內容、用量、服務重啟與接回的事件、回合的結果。
拿掉的內容:
  - read_weights 回傳的權重數值陣列(Gemma 的權重依它自己的條款散布,不在這個專案的授權範圍內);形狀、handle、純量的摘要保留。
  - 被 harness 截斷的 read_weights 輸出整段不放(截斷後無法可靠地只移除數值)。
  - 被截斷的 inspect_target 輸出裡成串的數字(啟動值),因為截斷後不是合法的 JSON,只留結構。
  - 模型回應裡不可讀的項目:加密的 reasoning 內容、thinking 區塊的簽章。
  - 本機路徑。
寫入的請求(write_weights 的參數)完整保留:它們是引用張量名稱的運算式、編輯者自己給的短常數,以及同一回合先前讀取 / 檢查所回傳的 handle。
那些呼叫是決定性的,所以照順序重放全部的工具呼叫(並把舊的 handle 對到新的)能重建同一份修改(analysis/replay_episode.py;24 個有評分的回合都實測過,見 docs/replay.md)。

用法:python3 analysis/export_runs.py <輸出目錄> <回合名稱>...
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SRC = Path("data/api-runs")
NUMBER_LINE = re.compile(r"^\s*-?\d+(\.\d+)?([eE][-+]?\d+)?,?\s*$")


def strip_weight_values(obj):
    """read_weights 的結果:移除所有 values 陣列,其餘照留。回傳 (新物件, 移除的數值個數)。"""
    removed = 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "values" and isinstance(v, list):
                flat = json.dumps(v).count(",") + 1 if v else 0
                out[k] = f"[omitted: {flat} weight values]"
                removed += flat
            else:
                out[k], n = strip_weight_values(v)
                removed += n
        return out, removed
    if isinstance(obj, list):
        items = [strip_weight_values(v) for v in obj]
        return [i[0] for i in items], sum(i[1] for i in items)
    return obj, 0


def strip_number_runs(text):
    """無法解析的(被截斷的)輸出:把連續 9 行以上只有數字的行換成一行說明。"""
    out, run = [], []

    def flush():
        if len(run) >= 9:
            out.append(f"[omitted: {len(run)} numeric lines]")
        else:
            out.extend(run)
        run.clear()

    for line in text.split("\n"):
        if NUMBER_LINE.match(line):
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def tool_content(name, content, stats):
    try:
        parsed = json.loads(content)
    except ValueError:
        if content.startswith("Error executing tool"):
            return content
        if name == "read_weights":
            stats["truncated_reads_dropped"] += 1
            return "[omitted: a read_weights output that the harness truncated; it contained weight values]"
        stats["truncated_inspects_stripped"] += 1
        return strip_number_runs(content)
    if name == "read_weights":
        parsed, n = strip_weight_values(parsed)
        stats["weight_values_removed"] += n
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    return content


def model_output(blocks):
    out = []
    for b in blocks:
        b = dict(b)
        if b.get("type") == "reasoning":
            b.pop("encrypted_content", None)
            b["note"] = "encrypted reasoning content omitted"
        if b.get("type") == "thinking":
            b.pop("signature", None)
            b["note"] = "signature omitted; the API returned no thinking text"
        out.append(b)
    return out


def export(label, dest):
    stats = {"weight_values_removed": 0, "truncated_reads_dropped": 0, "truncated_inspects_stripped": 0}
    src = SRC / label
    out = dest / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episode.jsonl").open("w") as f:
        for line in (src / "transcript.jsonl").read_text().splitlines():
            e = json.loads(line)
            kind = e.get("event")
            if kind == "model":
                e["output"] = model_output(e["output"])
            elif kind == "tool":
                e["content"] = tool_content(e["name"], e["content"], stats)
            elif kind == "end":
                e["meta"].pop("patch", None)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    meta = json.loads((src / "meta.json").read_text())
    patch = meta.pop("patch", None) or {}
    meta["patch"] = {k: patch[k] for k in ("format", "changed_parameters", "l2", "linf", "tensors", "sha256", "weight_revision", "sparse") if k in patch}
    if patch.get("path") and Path(patch["path"]).exists():
        # 內容的指紋:重放紀錄後可以拿來比對,不受檔頭寫法影響,也不含權重數值
        from replay_episode import content_sha256
        meta["patch"]["content_sha256"] = content_sha256(patch["path"])
    meta["prompt_sha256"] = hashlib.sha256((src / "prompt.txt").read_bytes()).hexdigest()
    meta["export"] = stats
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return stats


if __name__ == "__main__":
    dest = Path(sys.argv[1])
    for label in sys.argv[2:]:
        print(label, export(label, dest))
