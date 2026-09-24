# `eval/gtbench` — GTBench evaluation harness

Everything for running **GTBench** (7 sequential interactive games from
[jinhaoduan/GTBench](https://github.com/jinhaoduan/GTBench), the `gamingbench` module) against
our local Qwen models via vLLM, and for the **veriselect best-of-N** experiment (gold-free
process-verifier test-time selection). Every candidate plays vs GTBench's built-in
`random_agent`; the score is a win-rate `(win + 0.5·draw)/completed`.

The 7 games: `tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction`.

> **Which files make the paper numbers?** Jump to [What produces the paper data](#what-produces-the-paper-data).

---

## 0. One-time setup

GTBench is vendored as plain source under `bench/GTBench`, patched to talk to local vLLM. Run
**from anywhere** (`eval/gtbench/setup.sh` resolves its own paths), which:
1. installs GTBench's real deps (`open_spiel==1.4`, `ml_collections`, `jsonlines`,
   `gymnasium`, `python-box`, `retrying`, … — **not** its pinned `openai`/`langchain`),
2. overwrites `bench/GTBench/gamingbench/chat/chat.py` with our [`gtbench_patch/chat.py`](gtbench_patch/chat.py)
   (modern OpenAI client → local vLLM endpoint),
3. copies [`gtbench_model_configs/*.yaml`](gtbench_model_configs) into `bench/GTBench/gamingbench/configs/model_configs/`.

Steps 2–3 are the part that matters on a fresh checkout; the script also verifies the patched
backend imports. Re-run it after pulling a new GTBench.

```bash
bash eval/gtbench/setup.sh          # idempotent
```

**Model weights.** Every runner here needs `MODEL_ROOT` pointing at the directory holding the
model directories (e.g. `Qwen2.5-7B-Instruct`):

```bash
export MODEL_ROOT=/path/to/models
```

**Model routing.** Model path convention `local/<served>:think|:nothink` routes through the
`LOCAL_LLM_ENDPOINTS` JSON env var (served-name → `http://127.0.0.1:<port>/v1`) and toggles Qwen3
thinking. In `:think` mode everything up to `</think>` is stripped from the returned move but kept
in `raw_reasoning`. Every runner here sets `LOCAL_LLM_ENDPOINTS` itself and unsets `*_proxy` for
localhost — no manual export needed.

---

## 1. Two experiment tracks

| track | question | policy | selection | runners | results dir |
|---|---|---|---|---|---|
| **A. win-rate scaling** | how good is each model at *playing* the games? | Qwen3 (think/nothink) or Qwen2.5 | none (`prompt_agent`, n=1) | `run_gtbench.sh`, `run_gtbench_qwen25.sh` | `results/gtbench/` |
| **B. veriselect best-of-N** | does the gold-free process verifier pick better moves than random / majority / a judge? | Qwen2.5-7B (+ scaling sweep) CoT, n=8 | single / random@8 / veriselect@8 | `run_veriselect_*.sh`, `run_gtbench_scaling_q25.sh` | `results/gtbench_veriselect/` |

Track B is the paper's contribution; track A is the base-capability context.

---

## 2. File-by-file

### Verifier-selection experiment (track B) — the paper

| file | role |
|---|---|
| **`analyze_veriselect_gtbench.py`** | **The core offline analyzer.** For each move-state with N candidate (reasoning, move) pairs, scores every reasoning with the gold-free verifier and compares **selection rules on move QUALITY** (an expectimax/EV-vs-random *evaluation* oracle, separate from the gold-free verifier): `pass@1` / `maj@8` (self-consistency vote) / `vsel@8` (verifier argmax) / `judge@8` (LLM-judge argmax, if scored) / `oracle@8` (ceiling). Also reports verifier coverage + per-claim-type correct/total. **No GPU** — iterate the verifier here on frozen data. `--selfcheck` runs the majority tie-logic self-test. |
| **`judge_reward_gtbench.py`** | **LLM-as-judge baseline** (holistic, gold-free). A judge LLM (we use Qwen2.5-72B) grades each whole (reasoning, move) trace `SCORE: 0–10`, seeing only the per-move prompt the player saw. Writes `judge_rewards` **in place** into the JSONL (additive, atomic, resumable). `analyze_veriselect_gtbench.py` then reads it as the `judge@8` column. `--selfcheck` = no-GPU parser test. |
| `run_veriselect_datagen_q25_7b.sh` | **Generates the offline n=8 data.** Qwen2.5-7B CoT, `num_generations=N` (default 8), saves all N candidates per move into `results/gtbench_veriselect/q25_7b-cot-n8/`. Prompt template frozen so the verifier iterates offline. |
| `run_veriselect_online100.sh` | **The headline online table** (100 matches/game). 3 arms (single / random@N / veriselect@N) in parallel, one 7B server per GPU, gated on all 4 GPUs free. → `results/gtbench_veriselect/online100/` (set `RESDIR=online100_fixed` for the post-verifier-fix run). |
| `run_veriselect_online_q25_7b.sh` | Same 3 arms, smaller/exploratory (20 matches, one shared server). → `.../online/`. |
| `run_veriselect_refine.sh` | Targeted re-validation of a verifier change: re-runs **only nim + kuhn** (the games whose scoring changed) fresh, all 3 arms. → `.../online_refine/`. |
| `rerun_veriselect_arm.sh` | Fast iteration: re-runs **only the veriselect@N arm** (single/random are verifier-independent and reused), then re-summarizes. → `.../online/`. |
| `run_gtbench_scaling_q25.sh` | **veriselect scaling sweep** across all 7 Qwen2.5 sizes (0.5B–72B), serial; each model served once (TP=4, TP=2 for 0.5B), 3 arms concurrent. → `.../scaling_q25/`. |
| `summarize_online.py` | Win-rate table `(win+0.5·draw)/normal` per game×arm for any online results dir. |
| `summarize_scaling_q25.py` | Per-model mean win-rate for the scaling sweep + veriselect deltas + z. |

### Win-rate scaling (track A) — base capability

| file | role |
|---|---|
| `run_gtbench.sh` | One Qwen3 model+mode vs random_agent. `bash run_gtbench.sh <served> <think\|nothink> [matches] [games…]`. Needs servers up (`eval/serve.sh`). → `results/gtbench/<served>-<mode>/`. |
| `run_gtbench_qwen25.sh` | The 7 Qwen2.5 sizes, nothink only (Qwen2.5 has no think mode); handles its own serving incl. 72B on TP=2. |
| `analyze_gtbench_qwen25.py` | Aggregates the Qwen2.5 win-rate runs → `results/gtbench/REPORT_qwen25.md` + `summary_qwen25.json`. |

### Shared config / patch

| path | role |
|---|---|
| `gtbench_patch/chat.py` | The **only** file we override in the vendored GTBench tree: routes its LLM calls to local vLLM via the modern OpenAI client. Applied by `eval/gtbench/setup.sh`. |
| `gtbench_model_configs/*.yaml` | Qwen3 served-name → `local/<name>:think\|:nothink` model configs (+ `dummy-random.yaml` opponent). Copied into the vendored tree by `eval/gtbench/setup.sh`. |

---

## 3. Reproducing the paper's veriselect result (end to end)

```bash
# 0. one-time
bash eval/gtbench/setup.sh

# 1. generate the frozen offline n=8 data (Qwen2.5-7B CoT, all 7 games)   [~1 GPU, minutes]
bash eval/gtbench/run_veriselect_datagen_q25_7b.sh          # -> results/gtbench_veriselect/q25_7b-cot-n8/

# 2. OFFLINE selector comparison — pass@1 / majority@8 / veriselect@8 / oracle@8   [NO GPU]
python3 eval/gtbench/analyze_veriselect_gtbench.py          # prints the per-game table

# 3. (optional) add the LLM-judge@8 baseline column          [1 GPU pass over saved traces]
python3 eval/gtbench/judge_reward_gtbench.py \
    --root eval/results/gtbench_veriselect/q25_7b-cot-n8 \
    --judge_model_path <MODELS>/Qwen2.5-72B-Instruct --tp 4 --gpu 0,1,2,3
python3 eval/gtbench/analyze_veriselect_gtbench.py          # judge@8 column now populated

# 4. the ONLINE 100-match win-rate table (the headline)     [all 4 GPUs, gated]
RESDIR=online100_fixed bash eval/gtbench/run_veriselect_online100.sh

# 5. figures + tidy data   [not in this repo — see "Downstream analysis" below]
python3 analysis/gtbench_veriselect/build.py                # -> data_offline.csv / data.json / …
python3 analysis/gtbench_veriselect/plot_paper.py           # -> fig_*.{pdf,png}
```

Common env overrides on the runners: `N` (candidates/generations, default 8),
`NUM_MATCHES`, `WORKERS`, `GAMES`, `GPU`/`PORT`, `MAXTOK`.

---

## 4. What produces the paper data

- **Offline per-move selection table + baselines** (pass@1 / majority@8 / LLM-judge@8 /
  veriselect@8 / oracle@8): `analyze_veriselect_gtbench.py` + `judge_reward_gtbench.py`, over
  `results/gtbench_veriselect/q25_7b-cot-n8/` (from `run_veriselect_datagen_q25_7b.sh`).
- **Online 100-match win-rate table** (single / random@8 / veriselect@8, 7 games):
  `run_veriselect_online100.sh` (`RESDIR=online100_fixed` = final) + `summarize_online.py`.
- **veriselect scaling across Qwen2.5 sizes**: `run_gtbench_scaling_q25.sh` + `summarize_scaling_q25.py`.
- **Base-capability win-rate scaling**: `run_gtbench*.sh` + `analyze_gtbench_qwen25.py`.

**Downstream analysis / figures are NOT in this repository.** They live in the sibling
`analysis/gtbench_veriselect/` (headline 7B experiment) and `analysis/gtbench_veriselect_models/`
(cross-size scaling) trees of the private research checkout — their `build.py` reads the
`results/gtbench_veriselect/` dirs produced by the runners above, and `plot_paper.py` renders the
figures. The gold-free verifier itself **is** here: `veriselect/gtbench_gold_free_verifier.py`.

---

## 5. Notes / gotchas

- **Raw results are gitignored** (`eval/results/` isn't committed). Regenerate from these runners;
  the frozen n=8 data + backups live only on disk.
- **Kill safety:** runners kill **only the vLLM PIDs they started** (tracked per-script). Never
  `pkill vllm` on this shared cluster.
- **TP must divide num_attention_heads:** TP=4 everywhere except Qwen2.5-0.5B (14 heads → TP=2).
- **`judge_reward_gtbench.py` edits data in place** — it only *adds* a `judge_rewards` field
  (atomic writes, resumable). Back up `q25_7b-cot-n8/` first if paranoid (a `.pre_judge_bak_*` exists).
- **online vs offline:** per-move selection changes the game trajectory, so the win-rate table
  **must** be run online; the offline analyzer scores a fixed candidate pool and is the low-noise
  iteration signal for perfect-info games (validate imperfect-info games — kuhn/nim/liars — online).
- **`random@8` sits ~systematically below `pass@1`** (different candidate-parsing path); the
  verifier-isolating metric is **Δ vs random@8** (same n=8 pool, only the selection rule differs).
