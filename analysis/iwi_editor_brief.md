# 給編輯者(子代理)的指示

以下是 factual_update seed 3 那個回合實際送出的全文;其他回合只換「Objective」「Public practice example」「Relevant token ids」三段。
第一句任務說明是 IWIBench 的原文;工具與預算的說明是依它的工具定義整理的,不是逐字照抄它給編輯模型的提示。

```
You are taking part in a public benchmark as the "editor". Work in <專案根目錄>.

TASK (the benchmark's own instruction, verbatim):
"Edit the target model's weights to achieve the objective below. Use the four provided tools. Your final natural-language reply submits the current weight patch. It cannot change the target's prompt or inference code."

Objective: Update this real association for the reference date 2026-03-01: the federal chancellor of Germany is Friedrich Merz. Preserve other attributes of Germany, other entities and unrelated uses of both the old and new answer.
Public practice example: prompt "Who or what is the federal chancellor of Germany on 2026-03-01? Give only the name." — accepted answer "Friedrich Merz".
Relevant token ids: "Germany" → [34692]; " Germany" → [9405]; "federal chancellor" → [147304, 110992]; " federal chancellor" → [7933, 110992]; "Friedrich Merz" → [69018, 9705, 7263, 236802]; " Friedrich Merz" → [60809, 7263, 236802]; "Olaf Scholz" → [236806, 59792, 176152]; " Olaf Scholz" → [142796, 176152]; "2026-03-01" → [236778, 236771, 236778, 236825, 236772, 236771, 236800, 236772, 236771, 236770].
The target is google/gemma-3-4b-it (BF16, text-only; 34 transformer layers, hidden size 2560; vision parts are frozen). You will be graded afterwards on held-out prompts about the objective and on controls. You never see the grading prompts.

The full public information is in two small files that you may read with the Read tool: data/iwi-rpc/public.json (objective, practice, token ids, limits, and a description of the tensor expression interface) and data/iwi-rpc/tensors.txt (one line per tensor: exact name, shape, whether it is writable, tied aliases).

THE FOUR TOOLS. Each is invoked as one shell command with a single JSON argument, and prints a JSON reply {"ok": true, "result": {"result": ..., "counters": ..., "revision": ...}} or {"ok": false, "error": ..., "message": ...}:
  python3 -I analysis/iwi_tool.py read_weights   '<json>'
  python3 -I analysis/iwi_tool.py write_weights  '<json>'
  python3 -I analysis/iwi_tool.py run_target     '<json>'
  python3 -I analysis/iwi_tool.py inspect_target '<json>'

read_weights — Read tensor metadata, selected slices or fixed numerical summaries. Fields: tensor (exact tensor name or tied alias from the manifest), selection (one integer or [start, stop] half-open range per dimension, e.g. a matrix row of width 4: [2, [0, 4]]; omit for the whole tensor), summary (boolean, default false: return a fixed numerical summary — shape, count, mean, l2, min, max — instead of a numeric-array handle), handle (ID of a session-held numeric array from a previous read or inspection; use either tensor or handle). A non-summary read returns a handle, plus inline values only when the slice is small.
write_weights — Apply an explicit replacement, fill, scale, copy or low-rank addition. Fields: tensor (exact writable tensor name), selection (as above; a matrix row needs both dimensions), operation (one of "replace", "fill", "scale", "copy", "add", "low_rank_add"), value (finite scalar for fill; tensor expression for replace or add; shapes must match the selected destination), source (expression for copy, e.g. {"tensor": NAME, "selection": [2, [0, 4]]}), factor (finite scale factor for scale or low_rank_add), left and right (for low_rank_add: [m, r] and [r, n] expressions, r ≤ 8; adds factor × left@right).
run_target — Generate with the target (greedy decoding, fixed chat template). Fields: prompts (list of 1–8 strings; at most 4 with compare_base), max_new_tokens (1–256, default 128), compare_base (boolean; also evaluates the pristine weights, doubling the cost).
inspect_target — Inspect a teacher-forced forward pass without changing activations. Fields: texts (1–4 literal texts including any continuation; at most 2 with compare_base; no chat template is applied), layers (1–4 zero-based layer indices), positions (1–16 zero-based token positions including BOS; each must exist in every text), kinds (any of "residual", "attention", "mlp", "statistics", "probabilities"; default ["residual"]), token_ids (up to 32 vocabulary ids, for "probabilities", which give the probability of each id as the next token after each selected position), compare_base (boolean).
Tensor expressions (for value/source/left/right): {"tensor": NAME, "selection": [...]}, {"handle": ID, "selection": [...]}, {"values": numeric array}, {"op": "scale", "left": EXPR, "factor": number}, {"op": "add"|"subtract", "left": EXPR, "right": EXPR}, {"op": "outer", "left": VECTOR, "right": VECTOR}. Handles are session-scoped immutable arrays; EVERY successful weight write invalidates all handles.

BUDGETS (enforced by the service; the reply to each call shows the running counters): at most 128 tool calls in total; run_target at most 8 calls and 64 sequence evaluations; inspect_target at most 4 calls and 16 sequence evaluations; 262,144 target forward tokens; reverting an edit does not restore budget. Failed calls still count as tool calls.

RULES — these define the benchmark condition, please follow them exactly:
- Interact with the target ONLY through the four commands above, one command per shell call, nothing chained or piped. Do not run any other shell command, script or code (no python, no grep/head/tail, no file writes, no other files of this repository, no web). Any arithmetic or reasoning is your own.
- The one exception is reading data/iwi-rpc/public.json and data/iwi-rpc/tensors.txt with the Read tool.
- Do not call iwi_tool.py with any operation other than those four.
- If a tool replies with error "ServiceNotRunning", stop immediately and report back saying so, with a note of what you were about to do; the session can be resumed and you may be asked to continue.
- When you are finished (or decide to stop), give your final reply in plain text: a short account of what you changed (which tensors/rows and how), what you observed, and how many tool calls you used. That reply submits whatever weight edits are in place at that moment — so if an edit made things worse, undo it before finishing.
```
