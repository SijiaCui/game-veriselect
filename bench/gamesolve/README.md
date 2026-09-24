# GameSolve-Hard — redesigned benchmark generators

A harder, more diverse redesign of the two GameSolve tasks
(`best_response`, `nash_equilibrium`). Same 2-player normal-form setting and the
**same JSONL schema** as the original `bench/gamesolve_bench.jsonl` (drop-in for the
generation / verifier / reward pipeline), but with a controllable difficulty ladder
and a redesigned, scale-invariant reward.

## Why

The original bench was too easy — especially `best_response`:
- BR reduced to `argmax(R @ σ)` against a *fixed* opponent; ~half the games were 2×2.
- the opponent mixed strategy was peaked, so a trivial *"best-respond to the single
  most-likely column"* heuristic solved ~81% of BR games.
- Nash was dominated by pure-only games; genuinely-hard **mixed-NE** games were only ~13%.
- the outcome reward used `1 − error/10` tolerances, so naming the right BR action and
  writing zeros for every expected value scored ≈0.90.

## Files

| file | role |
|---|---|
| `solvers.py` | exact oracles: pure/mixed NE (nashpy, **degenerate games rejected**), best response, EU gap, strict dominance, IESDS, the two BR "shortcut" heuristics. |
| `games.py` | knob-controlled game constructors (reject sampling to hit structural constraints). |
| `descriptions.py` | NL renderings. ID styles `abstract/story/compact`; OOD styles `math/json/markdown/enumerated`. Abstract, **prior-free** action labels (disjoint across players). |
| `cot.py` | solver-grounded reference chain-of-thought (gold traces) ending in the canonical ANSWER block. |
| `reward.py` | **redesigned** outcome reward + strict `exact_match` (replaces `../gamesolve_reward.py`). |
| `generate.py` | tiered generation plan + driver → `gamesolve_hard.jsonl`, `gamesolve_hard_ood.jsonl` (+ `*_stats.json`). |
| `difficulty.py` | offline (no-GPU) difficulty audit + gold-trace self-consistency check. |

## Difficulty knobs

**best_response**
- `opp_alpha` — Dirichlet concentration of the opponent strategy. small→peaked (a
  shortcut works), large→~uniform (the full expectation is required).
- `gap_max` / `gap_min` — cap/floor the best-vs-2nd expected-payoff gap (small→near-tie,
  exact arithmetic required).
- `adversarial` — require **both** trivial heuristics (mode-column, global-max row) to
  *disagree* with the true best response → the shortcuts become traps.
- `force_ties` — make the best response a set (non-unique), via a σ-orthogonal
  perturbation that produces an *exact* tie (off by default; not used by any tier).

**nash_equilibrium**
- `require_mixed` — no pure NE, so a mixed NE must be solved for.
- `require_multi_pure` — ≥2 pure NE (coordination; must enumerate all).
- `distract_rows` / `distract_cols` — append **strictly-dominated distractor** strategies.
  They are single-step strictly dominated by every real strategy, so they are never played
  in any NE (the true equilibria are unchanged, just embedded with zeros) but force
  iterated elimination first.

shared: `(m, n)` size, `integer` vs 2-decimal payoffs, payoff range `[low, high]`,
`game_type ∈ {general, zero_sum, symmetric}` (Nash exercises all three; BR uses
general / zero-sum).

OOD samples additionally carry `metadata.ood_category ∈ {large_wide, asymmetric,
distractor_heavy}` for per-category analysis.

## Tiers

`easy → medium → hard → expert`. `easy` roughly matches the old difficulty (kept for
comparability); `hard`/`expert` turn on flat opponents / small gaps / adversarial traps
(BR) and mixed-required + distractors (Nash). See `BR_TIERS` / `NASH_TIERS` in
`generate.py`.

## Reproduce

```bash
python generate.py --out_dir . --scale 1.0 --seed 42   # ~1720 ID + ~220 OOD
python generate.py --out_dir . --scale 0.1             # quick smoke run
python difficulty.py gamesolve_hard.jsonl --gold       # audit + gold self-check
```

## Measured difficulty (scale 1.0, seed 42 → 1860 ID + 220 OOD)

best_response — fraction a trivial heuristic already gets right:

| tier | #act | eu_gap | opp_entropy | mode-col hit | global-max hit |
|---|--:|--:|--:|--:|--:|
| easy | 2.0 | 3.77 | 0.57 | 0.95 | 0.80 |
| medium | 3.0 | 1.78 | 0.78 | 0.76 | 0.54 |
| hard | 4.3 | 0.41 | 0.94 | **0.00** | **0.00** |
| expert | 4.3 | 0.36 | 0.96 | **0.00** | **0.00** |

nash_equilibrium — fraction requiring a mixed NE (no pure NE):

| tier | mixed-required | has-mixed | #dominated | IESDS steps |
|---|--:|--:|--:|--:|
| easy | 0.10 | 0.20 | 0.89 | 1.4 |
| medium | 0.14 | 0.50 | 1.10 | 1.6 |
| hard | 1.00 | 1.00 | 0.70 | 0.9 |
| expert | 1.00 | 1.00 | 2.51 | 3.6 |

Gold reference traces score **reward 1.0000 / exact-match 1.0000** under `reward.py`
(schema ↔ parser ↔ reward are self-consistent).

### Empirical difficulty (6 Qwen3, n=8, exact_match)
**best_response** is reachable **non-thinking** with a clean size trend and best-of-N headroom at
every tier (32B: BR-easy 1.00 → hard 0.63 → expert 0.28 pass@1; BR-hard oracle@8 0.93).

**nash_equilibrium is reasoning-gated.** Non-thinking, NE-hard/expert score 0 pass@1 *and* 0
oracle@8 (single-pass reasoning is insufficient — not a metric or decoding-budget artifact; gold
scores 1.0). With Qwen3 **thinking enabled** (needs a large budget, `max_tokens≈24k` — at 8k the
traces truncate before emitting an answer) the tiers un-floor cleanly and offer strong selection
headroom:

| Qwen3-32B (thinking) | pass@1 exact | oracle@8 exact | pass@1 full |
|---|--:|--:|--:|
| NE-medium | 0.10 | 0.33 | 0.10 |
| NE-hard | 0.17 | 0.46 | 0.17 |
| NE-expert | 0.30 | 0.75 | 0.25 |

`exact_full ≤ exact` throughout (models recover the mixed *support* more often than the exact
probability vector — the residual gap is exact-probability arithmetic), which is exactly why the
structural `exact_match` is the primary metric.

**Protocol implication:** evaluate NE with thinking on (BR is fine either way). Selection headroom
lives in BR (all tiers, non-thinking) and NE-medium/hard/expert (thinking); OOD via BR-hard.

## ⚠️ Caveats for the pipeline

1. **Reward is re-baselined.** `reward.py` is *not* comparable to the old
   `gamesolve_reward.py`: BR numeric tolerances are now scale-invariant
   (`|v−g| ≤ 0.02·|g| + 0.01`) and the BR action term dominates. For NE the mixed axis is
   split into **support** (which actions are randomised over — structural, LLM-reachable)
   and **vec** (the exact probabilities — hard arithmetic, size-adaptive tolerance).
   `exact_match` (primary) keys off pure NE + class + mixed *support*; `exact_match_full`
   additionally requires the exact probability vectors. This is because strict full-vector
   matching floored every model at 0 on hard/expert NE (even oracle@8 = 0); support-level
   scoring restores discrimination. Every reported number must be recomputed; the old
   canonical anchors (VeriSelect@16 ID 0.741, …) do not transfer.

2. **The process verifier needs a mixed-NE checker before the hard Nash tier is usable
   as a VeriSelect selection signal.** `veriselect/gt_prm_verifier.py` extracts only
   *pure*-NE pairs and dominance claims — it has no mixed-NE verification. On the new
   hard/expert Nash tiers (100% mixed-required) a correct gold trace therefore scores
   `process_score ≈ 0`. BR is unaffected (gold BR → `process_score = 1.0`). Extending the
   verifier to check a claimed `(σ_row, σ_col)` via the indifference / best-response
   conditions against the payoff matrix (still gold-free, still formally checkable) is a
   prerequisite for using the hard Nash tier in the selection experiments.

3. **Rare NE-set incompleteness on large / near-degenerate games (inherited).** Ground
   truth comes from nashpy support-enumeration, which can under-report equilibria on
   degenerate games. Degenerate games are rejected during sampling, but ~0.5% of the
   largest (padded distractor / asymmetric) Nash games may still store a *valid but
   possibly incomplete* equilibrium set. Every stored equilibrium is verified correct
   (sound); the residual risk is a missed alternative equilibrium on the "find ALL NE"
   task. This limitation is inherited from the original solvers. A slow cross-check
   (vertex/Lemke-Howson enumeration) could be added as an offline audit if needed.
