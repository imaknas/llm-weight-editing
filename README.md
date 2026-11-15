# llm-weight-editing

Experiments on language models that read and edit the weights of other models. The current stage is a re-analysis of [IWIBench](https://github.com/HCLTech-robotics/iwibench) (arXiv 2611.08647), a benchmark in which an "editor" model changes the weights of Gemma 3 4B through four tools.

Author: Knas Xiao. Licence: MIT (see `LICENSE`, `NOTICE.md`).

## What this stage found

1. In the 330 published IWIBench rollouts, no score is attributable to an edit of an internal layer. Editors that scored did so by rewriting rows of the token-embedding matrix. Four scored patches also change attention, MLP or normalisation weights. With those changes removed each scores the same, and with only those changes kept each scores zero.
2. An internal-layer edit that scores is possible with the benchmark's own prompt, tools and service-side budgets. Claude Fable 5.2 wrote MLP neurons on three `factual_update` episodes and scored 0.667 on each, and the internal part of each patch alone reproduces the score. One of the three episodes ran without interruption. Two were cut by our own cost cap and then resumed. We set the response cap to 32,000 tokens for this model instead of the benchmark's 4,096, so these runs are not comparable with the published leaderboard. On the same service and driver, gpt-6.1-sol in its published setting edited embedding rows only, as in its published runs.
3. Both kinds of edit reach at most 2 of 3 targets on these episodes. The IWIBench paper already reports that embedding-row edits fail the target that asks for the surname only. The internal-neuron edits fail it too.

Finding 1 is new as far as we could find. For finding 2, others have shown Claude models making this kind of edit with a code sandbox, so what is new is that it also happens inside IWIBench's restricted tools and is scored by its grader. The write-up has a related-work section with the sources.

The write-up, with what these results do not show, is in [`docs/technical-report.md`](docs/technical-report.md). The note sent to the benchmark's authors is in [`docs/issue-draft-en.md`](docs/issue-draft-en.md).

## Layout

| Path | Contents |
|---|---|
| `analysis/iwibench_rollouts.py`, `iwibench_rollouts_blind.py` | Counts of what the published rollouts wrote, from their tool-call logs and patch metadata. The second script was written without sight of the first. |
| `analysis/iwibench_patch_split.py` | Splits a patch into its embedding part and the rest, with bitwise checks. |
| `analysis/iwibench_patch_eval.py` | Grades a set of patch conditions with the benchmark's own grading worker and scorer, one fresh process per condition. |
| `analysis/iwibench_patch_logits.py` | Compares two conditions' next-token distributions on the same token sequences. |
| `analysis/iwi_rpc_server.py`, `iwi_tool.py` | The benchmark's `Service` class behind a file-based interface, with state on disk so an episode survives a restart. |
| `analysis/iwi_editor_api.py` | Editor loop that calls a provider API directly: the benchmark's prompt, tool definitions, tool-output formatting, error text and stop conditions; a cost ledger. |
| `analysis/export_runs.py`, `runs/` | Records of the six editor episodes: every model reply and tool call, with weight values removed. See `runs/README.md`. |
| `analysis/iwi_editor_tools.json` | Tool definitions, copied from the benchmark's published traces. |
| `docs/patch-ablation-*.md` | Plan and results for finding 1 (Traditional Chinese). |
| `docs/editor-conditions-*.md` | Plan and results for finding 2 (Traditional Chinese). |
| `docs/review-walkthrough.md`, `review-checklist.md` | The author's checklist for each published claim (Traditional Chinese). |
| `wse/`, `tests/`, first half of `docs/design.md` | Early pipeline tests on toy networks of about 200 to 4,400 weights. They are not evidence for anything above and are not cited in the write-up. |
| `analysis/singh_weight_reading*.py`, later part of `docs/design.md` | A separate recomputation of public weight-reading test data. Not part of this stage's report. |

Working documents are in Traditional Chinese. The article and issue drafts are in English.

## What is not in this repository

- Gemma weights and Gemma-derived weight patches. They are distributed by their publishers under Gemma's terms.
- The IWIBench source and its published rollouts. The scripts expect a local checkout and a local copy of the rollout files.

## Reproducing

Requirements: Python 3.13, `torch`, `transformers`, `safetensors`, `pydantic`; a checkout of IWIBench at commit `db9475f` in `data/iwibench-src`; Gemma 3 4B in a Hugging Face cache at `data/hf-cache` (requires accepting Gemma's terms); for finding 2, API keys in the environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Everything was run on an Apple M4 Pro with MPS. The benchmark itself targets CUDA.

One local change to the IWIBench checkout is needed on MPS: in `src/iwibench/tensors.py`, in the summary branch of `TensorStore.read`, `chunk.double()` becomes `chunk.detach().float().cpu().double()`, because MPS has no float64. The values are unchanged.

Finding 1, counts (text files only, no weights):

```
python3 analysis/iwibench_rollouts.py <dir>
```

where `<dir>` holds `index.jsonl` and, per trace, `r/<trace_id>/patch.json` and `r/<trace_id>/service-trajectory.jsonl` from the benchmark's Hugging Face dataset.

Finding 1, ablation of a published patch:

```
python analysis/iwibench_patch_split.py <patch.safetensors> <out dir>
python analysis/iwibench_patch_eval.py run --spec <spec.json> --device mps
python analysis/iwibench_patch_eval.py report --spec <spec.json>
```

The spec format is described at the top of `iwibench_patch_eval.py`.

Finding 2, one editor episode:

```
python analysis/iwi_editor_api.py --provider openai --model gpt-6.1-sol --effort low --max-output 4096 \
    --family factual_update --seed 2 --label sol-low-s2 --budget 5 --episode-cap 3
```

This starts the target service, runs the episode, submits and grades. `--budget` is a hard cap in US dollars across all runs recorded in `data/api-runs/ledger.jsonl`. The service is started directly by default. Where the GPU is only reachable through a job runner, pass the runner's command prefix with `--service-launcher`.

Tests for the toy code: `python3 -m unittest discover -s tests -t .`

## How this work was done

Knas Xiao set the questions, the checking rules, the budget and what to publish, and is responsible for the claims. An AI agent (Claude Fable 5.2 in Claude Code) wrote the code, ran the experiments and drafted the technical text. Results were recomputed by separate agent instances that were not shown them, reviewed adversarially by another, and graded blind by a local model from a different family. The mistakes those checks caught are listed in the write-up.

## Corrections and contact

Open an issue on this repository. Corrections to published claims will be recorded in the write-up with dates.
