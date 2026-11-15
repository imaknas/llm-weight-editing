# Draft of a GitHub issue for HCLTech-robotics/iwibench

*To be posted by Knas Xiao (imaknas) after review. Suggested title:*

**Observations from re-grading the published rollouts: no score traces to an internal-layer edit; notes on `factual_update` and on running off-CUDA**

Hi folks,

I re-graded some of the published patches locally with your grading worker and scorer, and ran a few extra editor episodes against your `Service` class. This was only possible because you published the rollouts, patches and native traces. Four observations follow. Scripts, a fuller write-up and the limits are at https://github.com/imaknas/llm-weight-editing (`docs/article-draft-en.md`).

**1. In the 330 family-0.2.2 rollouts, no score is attributable to an internal-layer edit.**
From the service trajectories and patch metadata: 3,342 of 3,683 successful writes go to the tied embedding/output matrix; 28 final patches contain internal tensors, 24 of them score 0, and the 11 that contain only internal tensors all score 0. I split the four scored mixed patches by tensor and re-graded each part (base and full reproduce your published numbers exactly on Apple MPS):

| trace | full | embedding part only | internal part only |
|---|---|---|---|
| `e9640019…` factual_update / gpt-5.6-terra | 1/3, 0.3085 | same; 74/74 outputs identical | 0; 74/74 identical to base |
| `36209ad6…` selective_factual_withholding / gpt-6.1-sol high | 3/3, 1.0 | same; 76/76 identical | 0 |
| `266ed91f…` selective_factual_withholding / gpt-6.1-sol medium | 3/3, 1.0 | same; 76/76 identical | 0 |
| `7784325c…` resistance_to_false_suggestions / gpt-6.1-sol medium | 47/64, 0.4333 | same score | 0 |

The paper notes that Astra's writes are all in the embedding matrix. The same holds for every scored rollout, including the four whose patches contain internal tensors.

**2. An internal-layer edit can score under your prompt, tools and service budgets.**
Driving Claude Fable 5.2 through the API with only your prompt and tool definitions (byte-identical prompt; checked against your native traces), three `factual_update` episodes (seeds 2, 0, 3) each ended with MLP-neuron edits in layers 27–31 scoring 0.667, where the internal part alone reproduces the score. Two caveats. First, I set its response cap to 32,000 tokens instead of your 4,096, because thinking cannot be disabled on that model and counts toward the cap; 9 to 11 responses per episode exceed 4,096, so this is not a leaderboard-comparable run. Second, two of the three episodes were interrupted by my own cost cap and resumed. gpt-6.1-sol at your "low" setting, on the same driver, reproduced embedding-only edits.

**3. `factual_update`: the surname-only target, and what controls cover.**
None of the nine patches I graded locally passes `target:downstream`, and no published `factual_update` rollout exceeds 2/3 targets. If the missed target is the same one, the family's effective ceiling is 2/3 for every editor so far. Separately, the editors' own probes found side effects that the four nearby controls do not test (other officer roles of the same entity flipping to the new name).

**4. Two practical notes.**
- Off-CUDA: on Apple MPS with source commit `db9475f`, twelve seeds of `answer_length_control` (1, 2, 5–14) fail baseline calibration, including seeds 1, 2 and 5 that your calibration table lists as eligible. I did not establish the cause.
- Fragility: in my own run of gpt-6.1-sol at your "low" setting, one success on `factual_update` seed 3 leads its runner-up by one BF16 logit step; generating one prompt per batch instead of eight flips it ("Friedrich Merkel") and the reward goes from 0.667 to 0.333.

If I have misread your setup anywhere, please correct me. I directed this work and an AI coding agent did the implementation. I went through the evidence for each claim with the agent, and I am the one to ask about them.
