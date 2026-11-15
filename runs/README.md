# Episode records

Six editor episodes on IWIBench `factual_update` (development split), run with `analysis/iwi_editor_api.py` against the benchmark's `Service` class on Apple MPS.

| Directory | Editor | Setting | Seed | Reward |
|---|---|---|---|---|
| `sol-low-s2` | gpt-6.1-sol | reasoning low, 4,096-token responses | 2 | 0.333 |
| `sol-low-s0` | gpt-6.1-sol | same | 0 | 0.667 |
| `sol-low-s3` | gpt-6.1-sol | same | 3 | 0.667 |
| `fable-high-32k-s2` | claude-fable-5-2 | effort high, 32,000-token responses | 2 | 0.667 |
| `fable-high-32k-s0` | claude-fable-5-2 | same | 0 | 0.667 |
| `fable-high-32k-s3` | claude-fable-5-2 | same | 3 | 0.667 |

`fable-high-32k-s0` and `fable-high-32k-s3` were cut by a cost cap and resumed. The `resume` event in each file marks the point, and the first `end` event before it holds the state at the cut.

## Episodes added on 2026-11-16

22 further episodes for `docs/wrap-up-results.md`, same driver and family.

| Directories | Editor | Setting | Token table |
|---|---|---|---|
| `lock-low-s{2,0,3}-r{1,2}` | gpt-6.1-sol | reasoning low | frozen for the episode |
| `lock-high-s{2,0,3}-r{1,2}`, `lock-high-s3-r{3,4}` | gpt-6.1-sol | reasoning high | frozen for the episode |
| `open-high-s{2,3}`, `open-high-s0`, `open-high-s0-r2` | gpt-6.1-sol | reasoning high | open |
| `fable-low-32k-s{2,0,3}`, `fable-low-32k-s3-r2` | claude-fable-5-2 | effort low, 32,000-token responses | open |

`lock-high-s3-r1`, `lock-high-s3-r2` and `open-high-s0` ended on a truncated HTTP response and have no patch;
`fable-low-32k-s3` was stopped by the provider's usage limit. Their replacements are `lock-high-s3-r3`,
`lock-high-s3-r4`, `open-high-s0-r2` and `fable-low-32k-s3-r2`. Rewards are in each `meta.json`.

## Files

`episode.jsonl`, one event per line:

- `start` / `resume`: the service session id and the prompt size.
- `model`: one model response. `output` holds its text and tool calls with full arguments, `usage` the token counts, `stop` why it ended.
- `tool`: one tool call. `payload` is the validated request sent to the service and `content` is the exact text returned to the model, except for the omissions below.
- `service_restart`: the local service was restarted between turns and the session restored.
- `end`: the outcome.

`meta.json`: settings, turns, tool calls, reward, the per-group counts from the benchmark's scorer, patch statistics and hash, cost, and the SHA-256 of the prompt. The prompt is the benchmark's own and is byte-identical to the published editor input for the same seed.

## What was removed

- Arrays of weight values returned by `read_weights`, replaced by a count. Gemma's weights are distributed under Gemma's terms, not under this repository's licence. Shapes, handles and scalar summaries are kept.
- `read_weights` outputs that the harness had truncated, replaced by a one-line note, because a truncated output cannot be reliably stripped of values.
- Long runs of numbers in truncated `inspect_target` outputs.
- Encrypted reasoning items (gpt-6.1-sol) and thinking-block signatures (claude-fable-5-2). The API returned no readable reasoning text for either model.

The `export` field of each `meta.json` gives the counts. Write requests are complete. They are expressions over named tensors, short constants supplied by the editor, and handles returned by earlier `read_weights` and `inspect_target` calls in the same file. Those calls are deterministic, so replaying all of a file's tool calls in order against the base checkpoint, mapping each old handle id to the new one, should rebuild the same patch. We have not run such a replay end to end.

Produced by `analysis/export_runs.py`.
