# eval — external transfer benchmarks

Size-scaling evaluation of the 6 Qwen3 chat models (`Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B`)
on the GTBench interactive-game benchmark, in two decode modes (thinking **off** and
**on**).

- **GTBench** ([jinhaoduan/GTBench](https://github.com/jinhaoduan/GTBench)) — git submodule at
  `bench/GTBench`. Candidate plays the built-in `random_agent`.

Models are served locally by vLLM (OpenAI-compatible). All 6 run co-located on **GPU2 + GPU3**
(3 per card). Thinking on/off is a per-request parameter, so one server per model covers both
modes.

## Reproduce

```bash
bash eval/setup.sh                 # clone GTBench submodule + deps + apply patch/configs
bash eval/serve.sh                 # 6 vLLM servers on GPU2/GPU3, ports 8001-8006
bash eval/run_all.sh --smoke       # quick end-to-end sanity (2 models, 1 game, 2 eps)
bash eval/run_all.sh               # full matrix: 6 models x {think,nothink} on GTBench
python3 eval/analyze_eval.py       # -> eval/results/{REPORT.md, summary.json}
bash eval/stop.sh                  # release GPUs (kills ONLY this session's server PIDs)
```

> Teardown is by PID (`eval/logs/server_pids.txt`, written by `serve.sh`). Do **not**
> `pkill vllm` — other tasks may run their own vLLM servers on other GPUs.

## Layout

Organized into 4 subfolders (benchmark code + logs + results); cross-cutting
serving/orchestration scripts and docs stay at `eval/` root.

```
eval/
├── setup.sh                 GTBench submodule + deps + patch install
├── serve.sh                 6 co-located vLLM servers (GPU2/GPU3); records PIDs to logs/server_pids.txt
├── serve_one.sh             one model, tensor-parallel across GPU2+GPU3 (per-model pattern)
├── stop.sh                  precise teardown of this session's servers (by PID, no broad pkill)
├── run_all.sh               orchestrate 6 models x 2 modes on GTBench
├── analyze_eval.py          aggregate GTBench -> results/REPORT.md, summary.json
├── README.md  RESULTS.md
├── gtbench/                 run_gtbench.sh, gtbench_patch/chat.py, gtbench_model_configs/
├── gamesolve/               in-house GameSolve-Hard eval: eval_core.py (shared engine),
│                            eval_qwen25.py / eval_qwen3.py (per-series entries),
│                            eval_think_diag.py, run_*.sh  (runs on GPU0/GPU1)
├── logs/                    all *.log, server_pids.txt
└── results/                 artifacts (git-ignored): REPORT.md, summary.json,
                             gtbench/, gamesolve/
```

Served-name → port: `q32b:8001 q14b:8002 q8b:8003 q4b:8004 q1_7b:8005 q0_6b:8006`.

