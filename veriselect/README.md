# veriselect

**The method.** A formally-sound game-theory *process verifier* used as a test-time
*best-of-N selector* — not as a training reward. Draw N i.i.d. traces per prompt, score
each with the verifier, return the argmax. Training-free; ground-truth-free in its
gold-free variant.

The pivot is **selection vs. shaping** of one identical signal: gradient-ascending a
policy against the verifier (shaping) lets it manufacture verifier-satisfying-but-wrong
traces and *hurts*; ranking N samples from a natural policy (selection) can't be gamed
and *helps*.

## Files

| file | what |
|---|---|
| `gt_prm_verifier.py` | the sound process verifier (GameSolve). Extracts formally-verifiable claims (`DOMINANCE` / `EQUILIBRIUM` / `BR_ACTION` / `EXP_PAYOFF`) from a free-form trace and checks each against the payoff matrices + solver ground truth. `verify_trace(response, game) -> {process_score, gold_free_score, accuracy, coverage, ...}`. Deps: `re`, `numpy`. |
| `gold_free_verifier.py` | the gold-free variant of the above: `verify_trace_gold_free(response, game)`. Scores claims without consulting the answer key. |
| `gtbench_gold_free_verifier.py` | the GTBench counterpart: `verify_trace_gtbench(response, observation)`. For interactive games, where the reference is the observation rather than a payoff matrix. |
| `veriselect.py` | the selector: `score_traces(traces, game, signal)` and `select(traces, game, rng, signal)` = argmax verifier score, random tie-break. `__main__` runs a Prisoner's-Dilemma self-check. |

`game` is a dict with `payoff_matrix_row`, `payoff_matrix_col`, `row_labels`,
`col_labels`, `task`, `ground_truth` — exactly the fields carried by each
`bench/gamesolve/*.jsonl` sample.

## Signals

- `process_score = 0.5·accuracy + 0.5·coverage` — primary selector signal; can't be
  gamed by one correct claim or a flood of wrong ones.
- `gold_free_score = correct_w / (correct_w+incorrect_w+1)` — uses **no answer key**;
  for the Nash task every claim is matrix-checkable, so this is deployment-realistic.

## Verify

Run from the repository root:

```bash
python3 veriselect/veriselect.py   # -> "veriselect selfcheck OK"
```

## Not in this migration (eval integration, done separately)

- Plugging VeriSelect into the shared harness as a `baselines/base.py` agent: the
  runner (which holds the game) computes a per-candidate verifier score, then VeriSelect
  selects argmax over it — structurally identical to `oracle`, just fed the verifier
  score instead of `exact_match`. Needs a `verifier_score` field on `Candidate`.
  (`baselines/` is not part of this repository.)
- The N-trace generation + bootstrap-curve analysis harness (BETA `generate.py` /
  `analyze.py`): the eval side, not the method engine.
