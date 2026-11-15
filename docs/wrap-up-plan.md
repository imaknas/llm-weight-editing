# Wrap-up plan: addresses for neurons, and a paired comparison of the two kinds of edit

Status: plan, version 3, 2026-11-16. Version 3 adds section 1A, which replaces part one as the experiment on
the "why" question. What follows in this paragraph describes version 2. Part one stopped at its free feasibility check (section 1.3, check 2); part
two has not been run. Version 1 was reviewed cold by two reviewers who did not see each other; the
changes are listed in section 4. No paid run has been made. The only things run are the free local checks in
section 1.3. The rules in sections 1.4, 1.7 and 2.5 are fixed before the data they apply to exist.

This closes the line of work in `docs/technical-report.md`. Two questions were left open there:

1. Why did one editor write token-table rows and the other write an MLP neuron?
2. Do the two kinds of edit differ in anything the benchmark's score does not see?

## 0. What is not done, and why

- **Removing the inspection tool** was named as the next experiment in the published article. It is dropped. For
  GPT 6.1 Sol it cannot make internal-layer edits appear, since Sol already makes none with the tool. For Claude
  Fable 5.2 one episode cost 9.65 to 12.68 US dollars, and three would exceed what is left of the budget.
- **Removing the token ids from the prompt** is dropped. `inspect_target` returns `selected_token_ids` for the
  positions the editor asks about (up to 16 positions per text, 4 texts per call, out of 4 inspection calls). The
  editor could recover the ids at the price of inspection budget, so removing the table would raise a cost
  rather than remove the address.
- **No new Fable episodes.** The stage therefore still cannot separate "different model" from "different output
  budget".

## 1. Part one: give neurons an address

### 1.1 Idea

A token-table row has an address the editor is handed: the prompt's `token_ids` block maps the strings of the task
to row numbers. Nothing in the prompt maps the task to a neuron, and the tools do not expose neuron activations
(`inspect_target` returns the MLP's 2,560-dimensional output, not its 10,240 hidden units). The Fable episodes got
around this by building a new neuron from residual directions. Part one hands the editor an address of that kind
and looks at where it then writes.

### 1.2 Conditions

Editor: `gpt-6.1-sol` with the published settings (reasoning low, 4,096 output tokens), the benchmark's prompt and
tool definitions, family `factual_update`, and those of seeds 2, 0 and 3 that pass the rule in 1.4.

| Condition | Prompt |
|---|---|
| B, baseline | unchanged |
| L, labels | adds a `neuron_ids` block after `token_ids` |
| R, labels of random units | same block and format, describing 8 other units truthfully |

Episodes are run in rounds. One round is one episode of each condition for each seed, in an order shuffled with a
fixed random seed, so that the three conditions are run at the same time and against the same model version. Two
rounds are run in full. A third round is run only if the whole round fits under the cap in 1.6. The three existing
Sol episodes and the published Sol rollouts for this family are reported next to B, not pooled with it.

### 1.3 The label, and what has been checked

Computed by `analysis/neuron_label_probe.py` from the public practice prompt only. At the answer position (the last
token of the chat-templated prompt) it records every MLP hidden unit's activation, and the largest absolute
activation of the same unit on 24 fixed unrelated prompts listed in the script. Among units whose activation on the
practice prompt is in the top 1% by magnitude over all layers, it takes, in layers 16 to 33, the layer whose 8 most
selective units have the lowest mean ratio (reference maximum over practice activation), and reports those 8.

The block the editor sees (the script's field names are mapped to these when the block is written):

```json
"neuron_ids": {"practice:0": {"position": "last prompt token", "layer": 33,
  "tensor": "model.language_model.layers.33.mlp.down_proj.weight", "axis": "column",
  "units": [{"index": 9741, "activation": 5.938, "max_abs_activation_on_24_unrelated_prompts": 0.159}, ...]}}
```

It says where and how strongly, as `token_ids` says where. It does not say what to write, and no sentence of
instruction is added to the prompt.

Condition R describes 8 units of the same layer drawn at random with a fixed seed, with their own measured
numbers. It differs from L in which units are described and therefore in the numbers, which show that those units
are not selective. It controls for the block's presence, length and format and for the hint that neurons can be
edited. It does not deceive the editor.

**Check 1, done: does the label carry enough for a scoring edit?** A scripted edit that uses the label (add the
difference of the new and old answers' first-token embeddings to the 8 units' output columns, each divided by the
unit's labelled activation) was graded with the benchmark's grader on seed 2. The script takes the smallest
strength on a fixed grid that flips the practice prompt, which was the grid's lowest value. Three variants were
tried, and I saw the hidden target outputs of seed 2 when comparing them:

| Scripted edit | Targets | Controls kept | Reward |
|---|---|---|---|
| 8 units, layer 33, rule above | 1/3 | 67/67 | 0.333 |
| 8 units, with 13 near-miss prompts added to the reference set | 0/3 | 67/67 | 0 |
| 1 unit, layer 24 | 0/3 | 62/67 | 0 |

The rule above is the first row and is frozen. What this shows is limited. By construction the edit adds the same
vector on the practice prompt whichever active units are chosen, so the first two rows differ only in how the
units behave on other prompts, and the difference in reward is one answer: "Mark Carney" against a sentence that
contains "Mark Carney" and fails exact match. Two more known weaknesses: the layer chosen is the last one, where a
write is close to a conditional push on the output and is not the kind of mid-network unit the Fable episodes
built; and the units are existing ones that happen to fire on the practice prompt, at the last prompt position
only, which cannot reach facts whose answer needs several tokens fixed in turn.

The scripted edit's strength used a residual norm that is not part of the label, taken by an oversight from the
last reference prompt and not the practice prompt. Since the lowest grid value already flipped the practice
prompt, this is left as it was run.

**Check 2, before any paid run: seeds 0 and 3.** The frozen rule is applied to both and graded. The results are
reported whatever they are.

Result, graded after the rule in 1.4 was written: both score 0.

| Seed | Fact | Targets | Controls kept | Reward | What the edited model answered |
|---|---|---|---|---|---|
| 0 | Intel's chief executive is Lip-Bu Tan | 0/3 | 67/67 | 0 | "Lipman" |
| 3 | Germany's chancellor is Friedrich Merz | 0/3 | 68/68 | 0 | "Friedrich Mittermeier", "Ola Scholz" |

The first token is moved and the rest of the name is not. On seed 2 the model completes "Carney" after "Mark" by
itself; here it does not complete "Tan" or "Merz". A label at the last prompt position addresses one token of the
answer. **By the rule in 1.4 only one seed passes, so part one is not run with this label.** Sections 1.2 and 1.5
to 1.7 are kept as written but have not been carried out. Check 3 was not needed.

What failed is this label together with an edit that moves only the first token of the answer. It says nothing
about whether a richer address, for instance one per answer token, would be enough. The two zeros were scored
with the benchmark's grader and rescored once by me; no second party has recomputed them yet.

On the order of events: the rule in 1.4 was written before these two seeds were graded, but after their strength
scans had run and after a reviewer had predicted that seed 0 would fail. Version 1 was not committed, so the
repository's history does not show this order.

**Check 3, before any paid run: is the edit within the tools?** The scripted edit for one passing seed is carried
out through the benchmark's own service with `write_weights` calls only, inside the tool budget, and submitted for
grading. If the service refuses it or the reward differs from the patch graded directly, that is reported and
part one does not start.

### 1.4 Go or no-go, fixed before check 2 is graded

A seed enters part one only if the scripted edit under the frozen rule gets a reward above 0 on it. If fewer than
two seeds pass, part one is not run: the finding is that this label does not carry enough, and no money is spent
on it. The label rule is not adjusted to make a seed pass.

### 1.5 Measurements

Per episode, mechanically. `analysis/iwibench_patch_split.py` is extended to accept patches that are purely
token-table or purely internal (it now refuses them); for these the missing part is empty and scores as base.

1. **Wrote inside**: the final patch changes at least one value outside the token table.
2. **Wrote a described unit**: the final patch changes a value in the `down_proj` column, or the `gate_proj` or
   `up_proj` row, of one of the 8 units in the block, in a tensor where the patch touches at most 64 units. For B
   the same count is made against the L units of that seed, as the base rate.
3. **Scored inside**: the internal part of the final patch, graded alone, has a reward above 0. This is the only
   definition used. Whether the token-table part alone also scores is reported beside it.
4. Reward of the full patch.
5. **Noticed the block** (a check on the manipulation, not an outcome): the episode makes at least one
   `read_weights` or `write_weights` call on the tensor named in the block, successful or not.

Episodes that end without a final reply or hit the per-episode stop are kept and marked; 1, 2 and 5 are computed
for them and 3 and 4 are reported as censored.

### 1.6 Cost

The three earlier Sol episodes cost 0.246, 0.325 and 0.313 US dollars. An episode that follows the label may be
longer: each write invalidates the handles, so reads are repeated, and an episode that runs to the 64-turn limit
could cost about 1 dollar. Estimate per round of 9: 3 to 6 dollars. Rules: an episode is stopped at 1.20; a round
starts only if 9 times 1.20 fits under the cap; the cap for part one is 12 US dollars, enforced by the ledger.
Spent so far: 35.47 of 60.

### 1.7 Readings, fixed in advance

With two or three episodes per cell these are observations, not rates. Statements are made separately; they are
not exclusive.

- **Address used.** If "scored inside" occurs in L and in neither R nor the new B episodes: handing over the
  address was enough for this editor to make a scoring internal edit on these facts. If it also occurs in R or B,
  no claim is made about the address.
- **Hint.** If "wrote inside" is more frequent in both L and R than in B, with "scored inside" in neither: the
  block prompts the editor to try, and the address did not help it succeed. The base rate matters here: of the
  330 published rollouts, 36 wrote to an internal layer at some point and 28 final patches contain internal
  tensors.
- **No effect.** If L and R look like B on measurements 1 to 3 and "noticed the block" is mostly true: making
  neurons addressable in this way does not move this editor. Weak evidence at this size. If "noticed the block" is
  mostly false, the manipulation did not register and nothing is concluded.
- **Baseline surprise.** If "scored inside" occurs in B, the earlier description of Sol was a small-sample
  accident, and that is reported first.

Not concluded in any case: anything about other editors, other families, labels of another kind, or mid-network
units.

## 1A. Part one, replaced: three explanations, one of them testable for money

Sections 1.1 to 1.7 are kept as the record of what was tried. This section is what will be run.

### 1A.1 Why the address explanation is set aside

The label check failed for a reason that goes beyond the label. A token-table row exists for every token and
corresponds to it by construction; an address only says where it is. Nothing inside the network corresponds by
construction to "who is the prime minister of Canada". The Fable episodes did not find a neuron, they built one. A
label can therefore only point at existing units that happen to fire on one prompt, which is borrowing, and a label
that specified when a unit should fire and what it should write would be the edit itself, not an address.

Second, Sol's token-table edits do not rest on the address alone either. Sol reads the model's internal state at
the answer position with `inspect_target` and adds that direction to a row. Both kinds of edit need that
measurement. What differs is what comes after it: one `add` on a row, against a normalisation, a gated
nonlinearity, a strength and a free slot to work out. That is a difference in difficulty, which is not separable
from the two other explanations below. The address explanation is therefore reported as analysis, with the label
check as its evidence, and is not tested further. This reasoning is mine and was written after the check.

### 1A.2 Reasoning effort: what the published data already show, no spending

The benchmark's authors ran `gpt-6.1-sol` at low, medium and high reasoning, 30 rollouts each. Counted from the
published traces with the analysis already in this repository, and recounted by a reviewer:

| Setting | Rollouts | Wrote inside at some point | Final patch has internal tensors | `factual_update`: wrote inside, rewards |
|---|---|---|---|---|
| low | 30 | 1 | 1 | 0 of 3; 0.30, 0.67, 0.67 |
| medium | 30 | 4 | 3 | 0 of 3; 0.67, 0.67, 0.67 |
| high | 30 | 8 | 6 | 0 of 3; 0.67, 0.67, 0.67 |

More reasoning goes with more attempts inside, in other families. On the facts there is no internal write at any
setting, but that is three rollouts per setting: at the high setting's overall rate of 8 in 30, three rollouts
with none would happen about 4 times in 10 by chance. The patch ablation in the technical report showed that none
of these internal parts scores. So the published data do not show reasoning effort turning fact edits into
internal ones, and are too few to rule it out. No new open-table episodes are run for this beyond the baseline in
1A.3.

### 1A.3 Capability: lock the token table

**Manipulation.** The service that the editor talks to marks the token table (`embed_tokens`, with its tied alias
`lm_head`) as frozen, the way it already marks the vision tower. The manifest in the prompt then shows
`"writable": false` for it, and a write to it under either name returns the service's own error, "Tensor is
frozen". The table stays readable. No sentence is added to the prompt, the `token_ids` block stays, and the tools
are unchanged. The switch is set in the service process for these episodes only; the benchmark's source is not
edited, so grading and part two run on the unmodified code.

This is close to a condition in Hu et al. (arXiv 2611.09382), where embedding edits were not allowed. What is
added here is the same benchmark, editor and facts as the open-table data, so the two can be set side by side.

**Conditions.** `gpt-6.1-sol`, family `factual_update`, seeds 2, 0 and 3, at the published low and high settings,
both with the published 4,096-token response cap.

| Table | Setting | Episodes |
|---|---|---|
| locked | low | 2 per seed |
| locked | high | 2 per seed |
| open | high | 1 per seed, as a baseline run through the same driver |

15 episodes, in an order shuffled with a fixed random seed. The open-table low baseline is our 3 existing
episodes. The authors' 9 rollouts on this family are shown beside them.

**Measurements.** Per episode, from the trace and the final patch:

1. **Tried inside**: at least one successful write to a tensor other than the token table.
2. **Scored**: reward of the final patch from the benchmark's grader, with targets and controls kept reported
   separately. With the table locked the patch can only be internal. An empty patch is reward 0 and marked
   "no edit".
3. Where the patch is: tensor kind (MLP, attention, normalisation) and layer. Edits in layer 0, in layer 33 or in
   the final normalisation are marked, since they sit next to the token table on the input or the output side and
   may work as a substitute for it.
4. Refused writes to the table, counted under both of its names.
5. How the episode ended: final reply, a length stop at 4,096 tokens, a limit, or the cost stop.

**Cost.** The published traces priced at current rates give 0.26 to 0.35 for low, against 0.25 to 0.33 measured
here, and 0.35 to 0.77 for high. A locked table may make episodes longer. Every episode has the same stop, 2.50 US
dollars, so that long episodes are not cut selectively. Expected total 6 to 10. The ledger's total is set so that
this section cannot spend more than 15; if the remaining room is less than 2.50 the run stops and the missing
cells are reported as not run. Spent so far: 35.47 of 60.

**Readings, fixed in advance.** Stated for each setting separately, on its 6 locked episodes. These are
observations, not rates.

- *Scored at least once* (reward above 0): this editor, at this setting, can make a scoring edit of a fact without
  the token table. Its token-table edits in the open setting are then the route it takes when both are available,
  not the only one available to it. If every scoring patch is of the marked kind in measurement 3, the statement
  is the weaker one: it found a substitute next to the table.
- *Scored never, tried inside in 4 or more of 6*: it tries and does not succeed at this setting.
- *Scored never, tried inside in 2 or fewer of 6*: it does not take the internal route even with the other closed.
- *Scored never, tried inside in 3 of 6*: mixed, reported as such.
- If low and high fall under different readings, both are stated, with the caution that it is 6 episodes against 6.
- If episodes mostly end on the length stop, the result is about the 4,096-token setting and is said to be.

Not concluded: a capability limit in general, since a larger response cap and other reasoning settings are not
run; and anything about why Fable took the internal route with the table open.

### 1A.4 The other side: Fable with less reasoning

1A.3 asks whether Sol can leave the token table when it has to. It cannot say whether Fable's internal edits come
from the model or from the amount of reasoning it was given: Sol wrote about 2,000 tokens per episode and Fable
more than 100,000. This section lowers Fable's reasoning and looks at where it writes.

**Manipulation.** `claude-fable-5-2` with effort low in place of high. Everything else as in the earlier Fable
episodes: the benchmark's prompt and tools, the token table open, the 32,000-token response cap. The cap is kept
because this model's thinking cannot be switched off and counts toward it; only the effort setting changes.

**Episodes.** Run after 1A.3, with what is left of the budget. One episode, seed 2 (the fact whose earlier
episode ran without interruption, at 9.65 US dollars and 119,395 output tokens). The cost at effort low is not
known: about 2 dollars if the episode is as short as Sol's, more than 6 if output only halves. A low stop would let
through only short episodes, which are the ones that take the token-table route, and so would decide the result.
The stop is therefore set to everything that remains after 1A.3, less 1 dollar, and the driver's reservation of a
full response before each call is counted inside it. An episode cut by the stop is not resumed. Seeds 0 and 3 are
run only if, after seed 2 ended with a final reply, the ledger has room for twice 1.4 times its cost; otherwise
seed 2 stands alone and that is said. If the API rejects effort low for this model, the section is not run.

**Measurements.** As for the earlier Fable episodes: reward; the patch split into token-table and internal parts,
each graded alone; the tensors written; tool calls made; how the episode ended; and the output tokens produced,
per response and in total.

**Readings, fixed in advance.** One to three episodes; observations. "Scores" means a reward above 0. Only
episodes that end with a final reply are read; a cut episode is reported with what it had written and nothing is
concluded from it.

- *Manipulation check first.* If the episode's total output is not below half of the same seed's total at effort
  high, the setting did not take, and the episode is reported as a repeat of the earlier condition.
- *The internal part alone scores*: this model takes and completes the internal route at the lower setting too. If
  its output is still far above Sol's, that is stated; the two editors remain unmatched on volume.
- *The token-table part alone scores and the internal part does not*: at the lower setting this model's scoring
  edit is on the token table, as Sol's is. The difference between the editors then follows the effort setting at
  least in part. Effort lowers reasoning and tool use together, and thinking is not returned by the API, so the
  two are not told apart; and with a single episode per setting, run-to-run variation is not excluded.
- *Only the two parts together score, or nothing scores*: reported, nothing concluded about the route.

## 2. Part two: the two kinds of edit on the same facts

### 2.1 Material

Per seed (2, 0 and 3), the candidates are:

- Token-table patches: ours from the Sol episode, and the authors' published Sol patches for the same seed at low,
  medium and high.
- Fable's patch from our API episodes (`fable-high-32k-*`), as submitted, and its internal part alone. The
  internal part is one neuron in seed 2, six neurons in layers 27 and 29 in seed 0, and two neurons plus one
  normalisation element in seed 3; it also contains neurons the editor zeroed while measuring. The full patch adds
  token-table rows used as scratch space.

Two of the Fable episodes were cut by a cost cap and resumed. Here the patches are compared as objects, not used
as evidence about editor behaviour.

**Matching on strength, before any probe is run.** Every candidate is first graded on the benchmark's own items
under the procedure of 2.3. For each seed the token-table patch used is the one whose reward equals that of
Fable's full patch; if several do, the first in the order ours, low, medium, high. A seed with no equal candidate
is reported descriptively and enters no statement in 2.5. Without this, a weaker patch would lose on paraphrases
for being weaker, not for being a token-table edit.

### 2.2 Probes

The categories below were chosen by me, and I have read the editors' own reports of side effects, so the
categories are not blind. The probes inside them are: they are written before any probe is run by two writers
that work independently and see only the three objectives, the practice prompts and this table. One is a fresh
agent and one a local model of another family. They do not see patches, transcripts, the benchmark's hidden items
or any earlier result. Each writes 6 probes per category and fact, each with a machine-checkable expectation.

| Category | Expectation |
|---|---|
| Paraphrase of the question, keeping the reference date | contains the new name |
| Implied by the fact (part of the name, yes/no, the reverse question), keeping the date | stated string |
| The same question with another explicit date | same as base |
| Same relation, other entities; other relations, same entity | same as base |
| Other uses of the new name and its first name | same as base |
| Free text that should mention the fact, keeping the date | contains the new name |
| The question with no date | none; outputs are reported |

The union of the two sets is used, with exact duplicates removed. Three probes known from earlier episodes
(surname only, deputy prime minister, party leader) form a separate non-blind group; if a writer produces an
equivalent on its own, that probe stays in the blind set and both are run.

### 2.3 Procedure

Each probe is run alone (batch size 1), greedy, with the benchmark's generation length, on the base model, the
matched token-table patch, Fable's full patch and Fable's internal part. "Same as base" compares outputs after
trimming whitespace. The base model is run twice; a probe whose two base outputs differ is dropped and counted.
For probes whose expected answer begins with the new name, the logit margin between the first tokens of the new
name and of the base model's answer (both without a leading space, as the answer opens the reply) is recorded at
the answer position; for other probes no margin is taken.

### 2.4 Cost

No API spending. Local GPU time only.

### 2.5 Readings, fixed in advance

Reported per category and per seed as counts, with no pooling and no significance claims; there are several
categories and at most three facts, so isolated differences are expected by chance. The comparison is between the
matched token-table patch and Fable's full patch. A probe favours a patch if that patch meets the expectation and
the other does not. A difference is stated for a category only if, in every matched seed, at least two probes
favour the same patch and none favours the other; with fewer than two matched seeds no difference is stated at
all. Otherwise the statement is "no consistent difference in this category". Where Fable's full patch and its
internal part differ, the difference is attributed to the scratch rows. The non-blind group is reported apart and
enters no count.

Not concluded: that one kind of edit is better in general. These are at most three facts, one family, one patch
per cell, made by different editors with different output budgets.

## 3. Checks

- Every number in the results is recomputed by an agent that has not seen the first computation.
- The code that writes the `neuron_ids` block and draws the R units is written and read by a second agent before
  the first paid episode.
- The stage ends with one results document. Whether anything further is published is decided after that.

## 4. Changes from version 1 after review

Two reviewers, one able to read the code and data. Problems were taken as a union. Checked against the files by
me: the split tool's refusal of pure patches, the mismatches between the label rule and the script, the identical
margins of the first two scripted edits, the seed 3 result at batch size 1, the published counts of internal
writes, and the episode costs. The rest were accepted on the reviewers' reasoning. Version 2 has not been
reviewed again; sections 2.1 to 2.5 get one more cold review before part two is run.

- The shuffled-label control attached the real units' numbers to inert units, which made it "false magnitudes",
  not "wrong address". Replaced by R, truthful labels of random units.
- "Wrote a labelled unit" was not computable as written and the split tool refused pure patches. Defined on the
  final patch, with a base rate from B; tool to be extended.
- "Scored inside" had two definitions. One is kept.
- The readings were neither exhaustive nor exclusive and ignored the published base rate of internal writes.
  Rewritten as separate statements, with a manipulation check.
- There was no rule for what happens if the label fails on seeds 0 and 3. Added in 1.4, before those are graded.
- The feasibility result was overstated: unit choice does not matter on the practice prompt, the reward
  difference is one answer's format, and the layer is the last one. Said in 1.3.
- The description of the label rule did not match the script in three places (top 1% is over all layers, a
  residual norm outside the label, strength at the grid floor). Corrected.
- Cost per episode could be three times the estimate, and a single cap could cut conditions unevenly. Rounds,
  a per-episode stop and a higher cap.
- Part two called seeds 0 and 3 equal pairs; at batch size 1 seed 3 is not. Pairs are no longer pooled, and each
  patch's score under the same procedure is reported.
- Part two compared Fable's patch with its debris. A neuron-only patch is added.
- Probe rules were tightened: base run twice, whitespace trimmed, margins only where defined, 64 tokens, the
  non-blind group kept apart from what the blind writers produce.
- A reviewer objected that a scripted edit does not show the editor can make it. Check 3 runs it through the
  tools; whether the editor finds it is the experiment.
- Not adopted: a claim that "scored inside" has no causal link to the edit (the internal part is graded alone
  against a base that scores 0 of 3 targets).

## 5. Changes in version 3 after a second review

Section 1A and part two were reviewed cold by two reviewers again. The reviewer with file access recounted the
table in 1A.2 and confirmed how freezing works in the code; I did not repeat those two checks. Adopted:

- 1A.2 said the question was answered; three rollouts per setting cannot do that. Reworded.
- 1A.3: the freeze must not touch the grader or part two; the response cap for high was unstated; "internal" by
  tensor name lets in substitutes next to the table; the readings had gaps; there was no open-table baseline
  through our driver at high; a stop derived from one episode would cut long episodes selectively. All changed.
- Part two: under batch size 1 two of our three Sol patches are weaker than Fable's, which would have been read
  as a difference between kinds of edit. Patches are now matched on strength first, using the authors' published
  Sol patches as further candidates. The Fable patches were misdescribed as one neuron each. The comparison rule
  and the probe counts were undefined. The categories are mine and not blind; said so.
- Not adopted: a request for statistical power. The readings are worded as observations.
