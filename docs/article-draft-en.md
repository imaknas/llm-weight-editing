# In IWIBench, no published score comes from an internal-layer edit, but one is possible with the same tools

*Draft, technical part. Sections marked ⟦Knas⟧ are to be written by the author; everything else was drafted by an AI agent and is to be checked by the author against `docs/review-checklist.md` before publication.*

**Author:** Knas Xiao · **Code and data:** github.com/imaknas/llm-weight-editing (commit ⟦hash at publication⟧) · **Status:** stage report; three follow-up questions are listed at the end and have not been run.

## Summary

[IWIBench](https://github.com/HCLTech-robotics/iwibench) (arXiv 2611.08647) asks an "editor" language model to change the weights of Gemma 3 4B through four tools so that it acquires a requested behaviour, and grades the result on held-out prompts and controls.

1. **In the 330 published rollouts, no score is attributable to an edit of an internal layer.** Editors that scored did so by rewriting rows of the token-embedding matrix. Four scored patches also contain attention/MLP/normalisation changes; with those changes removed, each scores the same, and with only those changes kept, each scores zero.
2. **An internal-layer edit that scores is possible with the benchmark's own prompt, tools and service-side budgets.** Claude Fable 5.2, driven through the API with nothing but the benchmark's prompt and tool definitions, wrote MLP neurons in layers 27–31 on three `factual_update` episodes and scored 0.667 on each; with the patch split, the internal part alone reproduces the score and the embedding part alone scores zero. On the same service and driver, gpt-6.1-sol in its published setting reproduced its published behaviour: embedding rows only.
3. **The two kinds of edit tie at 2 of 3 targets, and neither passes the "surname only" target.** The benchmark's controls do not distinguish them either.

What this does **not** show: that internal edits are better, that the difference is due to the model rather than to how much it was allowed to write, or that either editor "understands" weights. See *Limits*.

| | gpt-6.1-sol (published "low" setting) | Claude Fable 5.2 (effort high, 32,000-token responses) |
|---|---|---|
| Where it wrote (3 episodes) | embedding matrix only | MLP neurons in layers 31 / 27+29 / 31 (plus scratch rows) |
| Targets correct | 1/3, 2/3, 2/3 | 2/3, 2/3, 2/3 |
| Controls retained | 67/67, 67/67, 68/68 | 67/67, 67/67, 68/68 |
| Reward | 0.333, 0.667, 0.667 | 0.667, 0.667, 0.667 |
| Output tokens per episode | 1.3k–2.1k | 119k–165k |
| Tool calls per episode | 10–20 | 78–99 |

The Fable column is **one episode run without interruption and two that were cut by a cost cap and then resumed** (details under *Deviations*). We set its response cap to 32,000 tokens ourselves, eight times the benchmark's 4,096. The reason is that this model's thinking cannot be switched off and counts toward the cap, so we expected a 4,096-token cap to cut it off before it could call a tool. The runs bore this out: the first response alone was 15,636 to 29,234 tokens, and 9 to 11 responses per episode exceed 4,096. This column is therefore not a row that could be added to the published leaderboard, and whether the model can do the same under a 4,096-token cap was not tested.

## ⟦Knas⟧ Why I looked at this, and who did what

⟦Author's own text: why the question mattered, the moment it looked wrong, how decisions were made without an ML background, what went wrong along the way, how sure the author is.⟧

**Who did what.** Knas Xiao set the questions, the rules for checking (prior-work search before building, plans reviewed cold before running, results recomputed blind before being believed), the budget, and what to publish, and takes responsibility for this report. An AI agent (Claude Fable 5.2 in Claude Code) read the literature, wrote the code, ran the experiments and wrote this technical part. Independent rechecks were done by separate agent instances that were not shown the results, and by a local model from a different family (Qwen 3.6 35B). Note the overlap: the editor model in finding 2 is the same model that ran the study.

## How to check this yourself

Everything below runs on one machine with the benchmark's source and the published rollouts. Gemma weights and Gemma-derived patches are not redistributed here; they come from Hugging Face under Gemma's terms.

- **Finding 1, counts:** `python analysis/iwibench_rollouts.py <rollouts dir>` reads only the published tool-call logs and patch metadata. A second script written without sight of the first (`iwibench_rollouts_blind.py`) gives the same numbers.
- **Finding 1, ablation:** `analysis/iwibench_patch_split.py` splits a published patch into its embedding part and the rest; `analysis/iwibench_patch_eval.py` grades each part with the benchmark's own grading worker and scorer.
- **Finding 2:** the six episodes' full transcripts (every model response, every tool call and the exact text returned), the service-side write logs, and the cost ledger are in ⟦location of released run data⟧. Every write is a deterministic tensor expression, so replaying an episode's write log against the base checkpoint reproduces its patch.
- **What would refute it:** a published rollout whose score drops when its internal-layer changes are removed would refute finding 1. A rerun of the three Fable episodes that produces embedding-only patches would weaken finding 2 to "sometimes".

## Finding 1: what the published rollouts edited

From the public logs of all 330 rollouts (family version 0.2.2; eleven editor configurations of nine models):

- 3,342 of 3,683 successful writes (91%) target the embedding/output matrix (the two are tied in Gemma); 248 rollouts write nothing else.
- 36 rollouts wrote to an internal layer at some point; 28 final patches contain internal tensors. Of those 28, 24 scored zero, and the 11 patches that contain *only* internal tensors all scored zero.
- The four that scored were re-graded locally with the benchmark's own code. Base and full-patch results match the published numbers exactly for all four.

| Patch | Published / local full patch | Embedding part only | Internal part only |
|---|---|---|---|
| factual_update, gpt-5.6-terra | 1/3 targets, 0.3085 | same score; 74/74 outputs identical to full | 0; 74/74 identical to base |
| withholding, gpt-6.1-sol high | 3/3, 1.0 | same; 76/76 identical | 0; 76/76 identical to base |
| withholding, gpt-6.1-sol medium | 3/3, 1.0 | same; 76/76 identical | 0 |
| false suggestions, gpt-6.1-sol medium | 47/64, 0.4333 | same score; 223/224 identical | 0 |

What the internal parts are: in one patch, rounding residue left by scaling a normalisation layer by 1.5 and back; in two, MLP neurons the editor switched off or used as scratch storage; in one (the gpt-5.6-terra patch), a deliberately rewritten layer-33 neuron that has no effect on any graded output. So attempts at internal edits exist under the benchmark's conditions; successful ones do not.

## Finding 2: an internal edit that scores

**Setup.** The benchmark's `Service` class (tools, budgets, episodes, patch format, grading) runs locally on Apple MPS. A driver calls the provider API with a single user message and four tool definitions, no system prompt, no other tools, and stops as the benchmark does (64 turns, 128 tool calls, or a reply with no tool call).

**How closely the driver matches the benchmark's harness** was checked against the authors' published native traces for nine gpt-6.1-sol episodes: our prompt is byte-identical to theirs for all three seeds and tokenises to the same count (49,125 for seed 2); the tool definitions are taken from their traces; 205 of 205 untruncated tool outputs in their traces are reproduced byte-for-byte by our formatter, and the 14 truncated ones fit our truncation rule; all 223 of their tool calls pass our request validation.

**gpt-6.1-sol, published "low" setting.** Three episodes, all scored, all writes to the embedding matrix, 10–20 tool calls, no truncated response. This is the same behaviour as the nine published episodes of this model on these seeds (scores are not identical per seed; the published setting sends no temperature and is not deterministic).

**Claude Fable 5.2.** Thinking cannot be disabled on this model and it does not accept a temperature, so the benchmark's "reasoning none, 4,096 tokens" setting does not exist for it. We chose to run it at effort high with a 32,000-token response cap, because its thinking counts toward the cap and a 4,096-token cap would probably have ended each episode at the first response.

| Seed (fact) | What it wrote | Full patch | Embedding part only | Internal part only |
|---|---|---|---|---|
| 2 (prime minister of Canada → Mark Carney) | one layer-31 MLP neuron; eight neighbouring neurons left zeroed | 2/3, 67/67, 0.667 | 0/3; 74/74 outputs = base | 2/3; 74/74 = full |
| 0 (Intel CEO → Lip-Bu Tan) † | two layer-27 and four layer-29 MLP neurons | 2/3, 67/67, 0.667 | 0/3; 74/74 = base | 2/3; 74/74 = full |
| 3 (German chancellor → Friedrich Merz) † | two layer-31 MLP neurons and one element of that layer's pre-MLP normalisation weight | 2/3, 68/68, 0.667 | 0/3; 74/74 = base | 2/3; 74/74 = full |

† cut by the cost cap, then resumed.

The edits have the same shape in all three: a neuron whose input weights match the residual stream at the answer position of the target question (built from `inspect_target` readings of target prompts minus control prompts) and whose output weights point at the embedding of the answer's first token, with further neurons for later tokens of a multi-token name. A blind recomputation found that the neurons work only jointly (for seed 0, the four layer-29 neurons suffice and no single neuron scores; for seed 3, the two neurons wreck the model unless the normalisation element is also changed).

This is a conditional boost of specific output tokens at a specific position, not an updated fact the model can reason with: the target that asks for the surname only fails in all three (outputs "Markus", "Brown", "Scholz").

**One small case of reading weights.** In seed 3 the editor read eight elements of a normalisation weight, then changed one of them (index 443, from −1 to −0.996), without which its neurons do not work. From the order of calls this followed an unusually large value seen in inspected activations; the model's reasoning text is not returned by the API, so the "why" is our inference.

## Finding 3: the tie, and what the benchmark's grading cannot see

- None of the nine patches we graded locally (six from finding 2, three earlier ones) passes the surname-only target. No published `factual_update` rollout exceeds 2/3 targets; which target they missed is not in the public data.
- Controls did not separate the two kinds of edit: all were retained in all six episodes. The editors' own tests found side effects the graded controls did not probe (in earlier runs: "deputy prime minister of Canada" and three other Intel officer roles also flipped to the new name).
- Scores at this scale are fragile. One gpt-6.1-sol success leads its runner-up token by a single BF16 step; run one prompt at a time instead of in batches of eight, it becomes "Friedrich Merkel" and the reward falls from 0.667 to 0.333.
- A practical note for anyone reusing the benchmark on other hardware: on our machine (Apple MPS, source commit db9475f) twelve seeds of `answer_length_control` fail the benchmark's own baseline calibration, including seeds published as eligible. We did not establish the cause.

## Deviations from our plan

The plan, including how each outcome would be read, was written and reviewed before the paid runs (`docs/editor-conditions-plan.md`).

1. **Two Fable episodes were cut by our own cost cap and resumed.** By the plan's rule they should have been set aside as censored. The cap was mis-set (it reserved a worst-case price for every call). After the cut, and after seeing the half-finished patches score 0/3, we wrote a resume path and finished them. Evidence that the resume was invisible to the model: every counter continues by exactly one call; the patch at the cut equals the patch restored; the first resumed request reads the entire previous request from the provider's cache, so the rebuilt conversation prefix is byte-identical; the grade at the cut was never placed in the conversation. Strictly by the plan, stage 2 is one episode in the "internal edit scores" category and two censored.
2. The local service must restart every 30 minutes; it was restarted between turns once per Fable episode.
3. The Fable request bodies were not logged, only responses and tool results. That the first request contains nothing but the tool definitions and the prompt was verified afterwards by token count (88,651 / 88,804 / 88,679, equal to the first-turn usage).
4. Provider retries (up to six) were not logged.

## Independent rechecks

- **Blind recomputation** (separate agent, given only the patch files and the benchmark source, forbidden to read our code or results): all scores and attributions above reproduced, each condition rerun in a second process with identical outputs.
- **Adversarial review** (separate agent, read-only): confirmed the classifications, the driver-versus-trace comparison and the table numbers; raised deviation 1, the response-cap caveat, and several overstated sentences, all incorporated.
- **Cross-family grading** (local Qwen 3.6 35B, shown outputs under shuffled condition labels): same target counts and same pairing of conditions.
- Mistakes these rechecks caught during the work, and that are corrected here: a miscount of scored rollouts; an audit that missed an extra "advisor" tool available to the editors in an earlier, discarded setup; an initial experimental design whose main manipulation (raising the response cap for gpt-6.1-sol) could not have had an effect, because that model never came near the cap.

## Limits

- **Model versus budget is not separated.** The two columns differ in model, reasoning setting and output volume (roughly 80-fold). The authors' own runs show gpt-6.1-sol still editing only embeddings at medium and high reasoning effort, and never approaching the response cap; Fable at small output volume and gpt-6.1-sol at its highest effort were not run.
- **One fact type** (office → person), three episodes, one run each, no repeats.
- **MPS, not the benchmark's CUDA.** Published base and full-patch numbers reproduced exactly where we could compare; exact ties may break differently elsewhere.
- **"No effect" of scratch rows and zeroed neurons** is shown on the 74 graded prompts only.
- Same-family reviewers share blind spots with the agent that did the work; one cross-family model was used for the parts it could do without running code.

## Open questions (not run)

1. Do the two kinds of edit differ in generalisation and side effects on a held-out prompt set written before looking?
2. Does Fable still produce internal edits with little output, and does gpt-6.1-sol at its highest effort?
3. With `inspect_target` removed, so that weights are readable and activations are not, can any editor make an internal edit work? The embedding matrix is the one place where "which weights mean what" is known in advance (row *i* is token *i*); every internal edit that worked here was built from activations.

## Disclosure, corrections

Drafted with an AI agent as described above; API costs of about 35 US dollars were paid by the author. Corrections will be listed here with dates. Contact: ⟦contact⟧.
