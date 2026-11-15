# Wrap-up results

2026-11-16. Results of the plan in `docs/wrap-up-plan.md` (version 3). The counts in sections 3, 4 and 5 were
recomputed from the raw records by agents that had not seen the first computation, and agree with it. Section 1
was scored by me only, and the table in section 2 was recounted by a reviewer who had read the plan. Episode
records are in `runs/`, the probes in `probes/`. Limits and deviations are in the last two sections and matter for how
much weight each statement carries.

## Short version

1. With the token table locked, `gpt-6.1-sol` built MLP neurons that score, in 10 of 12 completed episodes, at
   low reasoning with 1,500 to 3,000 output tokens per episode as well as at high. It never tried to write the
   locked table. Its token-table edits are the route it takes when both are open, not the only route it has.
2. With its effort lowered from high to low, `claude-fable-5-2` still wrote MLP neurons and not the token table,
   in the two episodes that can be read.
3. On probes written blind (252, of which 216 carry an expectation), token-table edits and neuron edits of equal
   benchmark score show no consistent difference in any category. Both kinds change the answer for some earlier
   dates, and neither makes the model answer "yes" when asked whether the new person holds the office.
4. Handing the first editor "addresses" for neurons was stopped at a free feasibility check. The reason goes
   beyond the label: there is nothing inside the network that corresponds to a fact the way a token-table row
   corresponds to a token.

## 1. The label check (plan 1.3 and 1.4)

A scripted edit using only the label (8 existing units that fire on the practice prompt and not on 24 unrelated
prompts) scored on one fact of three, so by the rule fixed beforehand the paid experiment was not run.

| Fact | Targets | Controls kept | Reward | Edited model's answer |
|---|---|---|---|---|
| Canada's prime minister is Mark Carney | 1/3 | 67/67 | 0.333 | "Mark Carney" |
| Intel's chief executive is Lip-Bu Tan | 0/3 | 67/67 | 0 | "Lipman" |
| Germany's chancellor is Friedrich Merz | 0/3 | 68/68 | 0 | "Friedrich Mittermeier" |

The label moves the first token of the answer and nothing after it. The two zeros were scored by me only.

## 2. Reasoning effort in the published data (plan 1A.2)

| Sol setting | Rollouts | Wrote inside at some point | Final patch has internal tensors | Facts: wrote inside |
|---|---|---|---|---|
| low | 30 | 1 | 1 | 0 of 3 |
| medium | 30 | 4 | 3 | 0 of 3 |
| high | 30 | 8 | 6 | 0 of 3 |

## 3. Locking the token table (plan 1A.3)

`gpt-6.1-sol`, `factual_update`, seeds 2, 0 and 3, 4,096-token responses. "Scored" is a reward above 0; every
scoring locked episode passes 2 of 3 targets (reward 0.667) and keeps every control.

| Table | Setting | Ended with a final reply | Scored | Where the scoring patches are | Output tokens per episode |
|---|---|---|---|---|---|
| locked | low | 6 | 5 | MLP units in layers 30 (three), 31, 24 | 1,463 to 2,986 |
| locked | high | 6 | 5 | MLP units in layer 33 (four), 24 | 9,227 to 17,234 |
| open | high | 3 | 2, both 0.333 | token-table rows | 10,823 to 12,716 |

- No locked episode attempted a write to the table. The lock reaches the editor only through the `writable` flag
  in the manifest.
- The two locked episodes that scored 0 wrote one unit in layer 33 (low, seed 0) and three in layer 23 (high,
  seed 3).
- Each patch is one to three units written as a whole (input rows and output column), except one high episode
  with eleven units and one that also changes a normalisation element.
- Four of the five scoring patches at high are in the last layer, which the plan marks as possibly a substitute
  for the table on the output side. At low the scoring patches are in layers 24 to 31.
- The third open-table episode at high (seed 2) wrote only MLP units, in layers 20 and 33, and scored 0.

By the readings fixed in advance, both settings fall under "scored at least once": this editor can make a scoring
edit of a fact without the token table, and at high the weaker form of the statement applies to four of five.

## 4. Fable at low effort (plan 1A.4)

`claude-fable-5-2`, effort low, 32,000-token responses, token table open.

| Seed | How it ended | Output tokens (share of the high episode) | Final patch | Reward | Cost, USD |
|---|---|---|---|---|---|
| 2 | final reply | 47,191 (40%) | layer 29, one live unit; no table rows | 0.333 | 5.34 |
| 0 | tool-call limit, no final reply | 114,806 (75%) | units in layers 25, 27, 29 with zero output columns; 24 normalisation elements; 3 table elements | 0 | 9.79 |
| 3 | stopped by the provider's usage limit after 5 turns | not read | none | none | 2.37 |
| 3, rerun | final reply | 58,964 (36%) | layer 30, two units; 40 table rows zero-filled as scratch | 0.667 | 4.85 |

For the rerun of seed 3 the parts were graded alone: internal part 0.667, token-table part 0. Seed 2's patch is
internal only.

Read by the rules fixed in advance: the two episodes that end with a final reply pass the check that output fell
below half, and in both the internal part alone scores. This model takes and completes the internal route at the
lower setting too. Its output is still some 15 to 40 times Sol's at low. What fell is output, not tool use: 86 and 113 tool calls at
low against 78 and 99 at high. Seed 0 is not read.

## 5. The two kinds of edit on the same probes (plan part two)

**Matching.** Each candidate was graded on the benchmark's items one prompt at a time.

| Seed | Our Sol patch | Published Sol low / medium / high | Fable full / internal only | Token-table patch used |
|---|---|---|---|---|
| 2 | 0.333 | 0.667 / 0.667 / 0.667 | 0.667 / 0.667 | published low |
| 0 | 0.667 | 0.667 / 0.667 / 0.667 | 0.667 / 0.667 | ours |
| 3 | 0.333 | 0.299 / 0.667 / 0.667 | 0.667 / 0.667 | published medium |

**Probes.** 84 per fact, 12 per category, none dropped. Cells: probes meeting the expectation for base /
token-table patch / neuron patch, then in brackets how many favour the token-table patch and how many the neuron
patch.

| Category (expectation) | Seed 2 | Seed 0 | Seed 3 |
|---|---|---|---|
| Paraphrase (contains the new surname) | 0/10/11 [1, 2] | 2/12/12 [0, 0] | 0/7/7 [4, 4] |
| Implied (stated string) | 3/4/5 [0, 1] | 0/0/0 [0, 0] | 3/3/4 [0, 1] |
| Other date (same as base) | 12/8/6 [2, 0] | 12/1/5 [0, 4] | 12/11/7 [4, 0] |
| Neighbours (same as base) | 12/4/7 [0, 3] | 12/10/11 [0, 1] | 12/12/11 [1, 0] |
| Other uses of the names (same as base) | 12/2/8 [0, 6] | 12/6/9 [1, 4] | 12/10/8 [3, 1] |
| Free text (contains the new surname) | 2/11/6 [5, 0] | 2/7/7 [2, 2] | 0/2/3 [1, 2] |

By the rule fixed in advance (in every seed at least two probes favour the same patch and none the other), **no
category shows a consistent difference.** Where one seed leans one way another leans the other: other dates
favour the token-table patch in seeds 2 and 3 and the neuron patch in seed 0; other uses of the names favour the
neuron patch in seeds 2 and 0 and the token-table patch in seed 3.

What both kinds share:

- Both change answers for dates before the reference date, where the base model's answer was right. Reading the
  short answers: Carney is named prime minister for a date in 2024 by both patches; Lip-Bu Tan is named Intel's
  chief executive for a date in 2022 by both, and by the token-table patch also for dates in 2010 and 2020; Merz
  is named chancellor for dates in 2023 and 2024 by the neuron patch and not by the token-table patch. Questions
  about dates after the reference date mostly get a long hedged reply from every model, and their counts in the
  table are dominated by wording drift.
- The counts in the "implied" row are not reliable: the expectation is a substring, and "No" or "Yes" also occur
  inside longer replies. Read directly, the outputs say this. Asked whether the new person holds the office on
  the reference date (13 yes/no questions over the three facts), every model, edited or not, opens with "No" or
  "False"; asked whether the old holder does (6 questions), every one opens with "Yes". Asked for the surname
  only, no patch gives it. Asked for the first name only, the neuron patch gives it in both facts where it was
  asked ("Mark", "Friedrich") and the token-table patch gives "Mark Carney" and "Olaf". So the edit changes what
  the model says when asked who holds the office, in either form, and does not change what it affirms.
- Questions with no date (reported without an expectation, as planned): where the base model gives a bare name,
  both patches replace it with the new name in seeds 2 and 0; in seed 3 the neuron patch does and the token-table
  patch does not. Where the base model replies with a hedged sentence, the patches mostly leave it.
- Fable's full patch and its internal part give identical outputs on every probe and every benchmark item, so
  the scratch rows it left behind change nothing measurable here.

About the "same as base" columns: of 80 misses, about 60 to 70% are a different answer and about 30% are wording
drift late in a long reply, by the recomputing agent's rule and reading of a sample. The counts overstate real
changes by roughly that much. The surname "Tan" also matches inside other words; with whole-word matching seed 0
becomes 0/11/11 for paraphrase and 0/6/4 [2, 0] for free text, and the decision does not change.

## 6. What this does and does not answer

The question was why one editor wrote token-table rows and the other MLP neurons.

- It is not that Sol cannot. Locked out of the table it writes scoring neurons, with a few thousand output tokens.
- It is not Sol's reasoning setting, within what was run: more reasoning brings more internal attempts in other
  families, none on facts with the table open, and one unprompted internal attempt in our own three episodes,
  which scored 0.
- It is not that Fable needs its large output to go inside. At 36 to 40% of it, it still does.
- So on these facts the difference is in which route each editor takes when both are open. Why each takes its
  route is not shown. The API does not return either model's reasoning.
- The two routes give edits that the benchmark and 216 scored probes do not consistently tell apart.

Not shown: anything beyond one family, three facts and two editors; what Sol would do with larger responses; what
Fable would do with the table locked or at Sol's output volume.

## 7. Deviations from the plan

- **Driver failures and reruns.** Three Sol episodes ended on a truncated HTTP response before any patch was
  submitted (`lock-high-s3-r1`, `lock-high-s3-r2`, `open-high-s0`). The driver now resends such a request. Three
  replacement episodes were run under new labels. The plan had no rule for this; the failed records are kept.
- **Provider limit.** `fable-low-32k-s3` was stopped by the account's usage limit and rerun as
  `fable-low-32k-s3-r2` after the limit was raised. Seed 0, which ended by the benchmark's own tool-call limit,
  was not rerun.
- **Budget.** The cap was raised from 60 to 85 US dollars during the stage, by the author, to allow the two
  further Fable episodes. Spent: 65.73, of which 7.91 on the 18 Sol episodes and 22.35 on the four Fable ones.
- **Order of the Fable episodes.** The plan made seeds 0 and 3 conditional on the ledger; with the raised cap
  both were run.
- **Margins.** Taken for the paraphrase category only, at the first answer position. They are not used in any
  statement: replies that open with "As of" make the first-position margin uninformative.
- **The non-blind probe group** of plan 2.2 (surname only, deputy prime minister, party leader) was not run as a
  separate group. The blind writers produced surname-only probes on their own, reported above.
- **Second review of version 3's last section** was done on 1A.4 only after the Sol run had started.

## 8. Limits

- Two to six episodes per cell. These are observations.
- Our open-table Sol episodes at high scored 0, 0.333 and 0.333, against 0.667 three times in the authors' data.
  The cause is not known (a different API surface, a different day, or chance), and it means our driver is not
  shown to reproduce the authors' runs at that setting.
- Locking the table follows a condition in Hu et al. (arXiv 2611.09382). What is added is the same benchmark,
  editor and facts as the open-table data.
- The probe categories were chosen by someone who had read the editors' own reports; the probes were written
  blind.
- The reading of section 6 on "which route each takes" rests on Sol never trying the internal route on facts
  with the table open in 14 of 15 completed episodes (6 of ours and the authors' 9), and Fable taking it in all 5
  that ended with a final reply.
