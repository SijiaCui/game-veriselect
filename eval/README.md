# eval — evaluation harness

Two evaluation tracks, both scoring a candidate model against a reference:

- **GameSolve-Hard** (static, in-house) — `gamesolve/`. Per-prompt pass@1 / majority@N /
  oracle@N over matrix games, plus LLM-judge baselines.
- **GTBench** (interactive, external) — `gtbench/`. Candidate plays GTBench's built-in
  `random_agent`; this is where the VeriSelect selection experiment lives.

Models are served locally by vLLM (OpenAI-compatible). Thinking on/off is a per-request
parameter, so one server per model covers both decode modes.

## Before running anything

Model weights live **outside** this repo. Every runner requires `MODEL_ROOT`:

```bash
export MODEL_ROOT=/path/to/models      # holds Qwen3-8B/, Qwen2.5-7B-Instruct/, ...
```

Scripts derive their own paths from their location, so they can be run from anywhere.
Interpreter defaults to `python3`; override with `PY=/path/to/python` if needed.

## Reproduce — GTBench (Qwen3 size scaling)

```bash
bash eval/setup.sh                 # GTBench deps + apply patch/configs
bash eval/serve.sh                 # 6 vLLM servers, ports 8001-8006
bash eval/run_all.sh --smoke       # quick end-to-end sanity (2 models, 1 game, 2 eps)
bash eval/run_all.sh               # full matrix: 6 models x {think,nothink} on GTBench
python3 eval/analyze_eval.py       # -> eval/results/{REPORT.md, summary.json}
bash eval/stop.sh                  # release GPUs (kills ONLY this session's server PIDs)
```

> Teardown is by PID (`eval/logs/server_pids.txt`, written by `serve.sh`). Do **not**
> `pkill vllm` — other tasks may run their own vLLM servers on other GPUs.

> GPU assignment is a convention in the scripts (GTBench serving on 2 cards, GameSolve on
> 0-3), not a hard requirement — adjust the `CUDA_VISIBLE_DEVICES` / `--gpu` values in the
> runners to match your box.

## Reproduce — GameSolve-Hard

```bash
export MODEL_ROOT=/path/to/models
bash eval/gamesolve/run_gamesolve.sh          # n=8, all series, one model at a time
python3 eval/analyze_eval.py                  # aggregate
```

## Layout

```
eval/
├── setup.sh                 GTBench deps + patch/config install
├── serve.sh                 co-located vLLM servers; records PIDs to logs/server_pids.txt
├── serve_one.sh             one model, tensor-parallel across both cards (per-model pattern)
├── stop.sh                  precise teardown of this session's servers (by PID, no broad pkill)
├── run_all.sh               orchestrate models x {think,nothink} on GTBench
├── analyze_eval.py          aggregate GTBench -> results/REPORT.md, summary.json
├── README.md
├── gtbench/                 GTBench track: runners, analyzers, patch + model configs
├── gamesolve/               in-house GameSolve-Hard eval: eval_core.py (shared engine),
│                            eval_qwen25.py / eval_qwen3.py (per-series entries),
│                            eval_think_diag.py, judge_reward.py, run_*.sh
├── logs/                    all *.log, server_pids.txt          (created at run time)
└── results/                 artifacts                                   (git-ignored)
```

Served-name → port: `q32b:8001 q14b:8002 q8b:8003 q4b:8004 q1_7b:8005 q0_6b:8006`.
