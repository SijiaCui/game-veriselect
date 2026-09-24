# bench

Benchmark data and generators, plus external benchmarks used for evaluation.

- `gamesolve/` — in-house GameSolve-Bench matrix-game data/generators.
- `GTBench/` — [jinhaoduan/GTBench](https://github.com/jinhaoduan/GTBench) (interactive
  game-theory benchmark), **vendored as plain source** rather than a git submodule, so a
  checkout needs no extra fetch step.

GTBench's LLM backend is overridden for local vLLM by `eval/gtbench/gtbench_patch/chat.py`,
and the Qwen model configs are added by `eval/gtbench/gtbench_model_configs/` — `eval/setup.sh`
copies both into the tree. See `eval/README.md` for how these are used in the evaluation.
