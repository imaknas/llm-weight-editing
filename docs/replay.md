# Rebuilding an episode's patch from its public record

2026-11-18. The weight patches the editors produced cannot be published, because they are derived from Gemma's
weights. The episode records under `runs/` can, and they contain no weight values. This note shows that the
records are enough to rebuild each patch, so that the scores in the reports can be checked without trusting the
authors.

## What was shown

Each of the 24 graded episodes that produced a patch was rebuilt from its public record alone, by re-sending the
recorded tool calls in order to a fresh benchmark session, with only handle ids remapped. In all 24:

- the prompt built for the new session has the same SHA-256 as the recorded one;
- every call succeeds or fails as it did in the record, and the revision and budget counters after each call match;
- the rebuilt patch has identical tensor contents: the same positions, and values equal bit for bit;
- the graded generations and the reward are the same.

The 24 are the 12 locked-table episodes and 6 open-table episodes of `gpt-6.1-sol` and the 6 graded episodes of
`claude-fable-5-2`, including those whose original run was interrupted and resumed. The per-episode results are in
`runs/replay/`.

An agent that had not taken part audited the claim. It parsed the original patch files with its own code,
recomputed the fingerprints for all 24, compared original and rebuilt tensors one by one, and checked that the
replay reads nothing but the public record. It found no way to break the claim and asked for the limits below to
be stated.

## Limits

- Tested once per episode, on the machine and software that produced the originals (Apple MPS, torch 2.15.0,
  safetensors 0.8.0). Most patches contain values computed from inspected activations, so bit-identity depends
  on the forward pass being deterministic. On other hardware or versions the values may differ in the last bits,
  and then the fingerprint will not match even if the reward does. That has not been tried.
- Identity is of tensor contents, not of the file. The patch file's own hash (`patch.sha256` in `meta.json`)
  differs between original and rebuilt files because the header is written in a different key order. Compare
  `patch.content_sha256`.
- Four of the 28 records have no patch and were not replayed: three ended on a network error and one on an
  account limit.
- You need the Gemma 3 4B checkpoint (revision in each prompt) and the benchmark's source to run this.

## How to run it

From the repository root, with the benchmark's source in `data/iwibench-src/src` and the checkpoint in
`data/hf-cache`:

```bash
python analysis/replay_episode.py runs/lock-low-s2-r1 --out data/replay
```

The last line printed says whether the prompt, the patch fingerprint, the reward and the per-group counts match
the record. The shortest episode takes a few minutes on an Apple M4 Pro.

## How it works

`analysis/replay_episode.py` creates a session for the record's family and seed (frozen tensors as recorded),
then sends each recorded tool call. Handles in a record belong to the session that produced it, so after each
call the new output is laid out the way it was shown to the editor, truncation included, and the handle ids in it
are matched in order with those in the recorded output. Later calls use the new ids. Outputs that the public
record omits because they held weight values are never referred to by a later call.

The fingerprint is the SHA-256 over each tensor's name, positions and values, with tensors sorted by name. It
reveals no weight values.
