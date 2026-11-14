"""LLM 呼叫的介面與實作。實驗邏輯只依賴 LLM 協定,具體怎麼呼叫由入口注入。"""
from __future__ import annotations

import json
import subprocess
from typing import Callable, Optional, Protocol


class LLM(Protocol):
    def complete(self, prompt: str, schema: Optional[dict] = None) -> dict:
        """回傳 JSON 物件;失敗時丟例外(不要默默換別的模型)。"""
        ...


class LLMError(RuntimeError):
    pass


class BrunnaCompleteLLM:
    """透過平台的 `brunna complete <name>` 呼叫模型(模型需先在 registry 宣告)。

    runner 預設是 subprocess.run,可注入以便測試。
    注意:輸出格式(有 schema 時取 structured_output)依平台說明撰寫,尚未實測。
    """

    def __init__(self, model_name: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 schema_dir: Optional[str] = None):
        self.model_name = model_name
        self._run = runner
        self._schema_dir = schema_dir
        self.calls = 0             # 成功的呼叫數
        self.total_cost_usd = 0.0  # 平台回報的累計花費(回報裡沒有這欄就不加)

    def complete(self, prompt: str, schema: Optional[dict] = None) -> dict:
        cmd = ["brunna", "complete", self.model_name]
        schema_path = None
        if schema is not None:
            import os, tempfile
            fd, schema_path = tempfile.mkstemp(suffix=".json", dir=self._schema_dir)
            with os.fdopen(fd, "w") as f:
                json.dump(schema, f)
            cmd += ["--schema", schema_path]
        try:
            proc = self._run(cmd, input=prompt, capture_output=True, text=True)
        finally:
            if schema_path:
                import os
                os.unlink(schema_path)
        if proc.returncode != 0:
            raise LLMError(f"brunna complete 失敗(exit {proc.returncode}): {proc.stderr.strip()[:500]}")
        out = json.loads(proc.stdout)
        if isinstance(out, dict) and ("llm_error" in out or out.get("is_error")):
            raise LLMError(out.get("llm_error") or str(out.get("result", ""))[:500])
        self.calls += 1
        if isinstance(out, dict) and isinstance(out.get("total_cost_usd"), (int, float)):
            self.total_cost_usd += out["total_cost_usd"]
        if schema is not None:
            if not isinstance(out, dict) or "structured_output" not in out:
                raise LLMError("給了 schema 但輸出沒有 structured_output")
            return out["structured_output"]
        return out


class RetryingLLM:
    """包住另一個 LLM:遇到 LLMError 最多再試幾次(同一個模型,不換別的)。

    每次重試都經 log 記錄,全部失敗就把最後的 LLMError 丟出去。sleep 可注入以便測試。
    """

    def __init__(self, inner: LLM, attempts: int = 3, log: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = lambda s: None, backoff: float = 5.0):
        self._inner, self._attempts, self._log, self._sleep, self._backoff = inner, attempts, log, sleep, backoff

    def complete(self, prompt: str, schema: Optional[dict] = None) -> dict:
        for n in range(1, self._attempts + 1):
            try:
                return self._inner.complete(prompt, schema)
            except LLMError as e:
                self._log(f"LLM 呼叫失敗(第 {n}/{self._attempts} 次): {e}")
                if n == self._attempts:
                    raise
                self._sleep(self._backoff * n)


class ScriptedLLM:
    """測試用:以函式決定回答,不碰網路。"""

    def __init__(self, fn: Callable[[str], dict]):
        self._fn = fn

    def complete(self, prompt: str, schema: Optional[dict] = None) -> dict:
        return self._fn(prompt)
