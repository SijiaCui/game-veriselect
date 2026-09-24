# bench

Benchmark data and generators, plus external benchmarks used for evaluation.

- `gamesolve/` — in-house GameSolve-Bench matrix-game data/generators.
- `GTBench/` — [jinhaoduan/GTBench](https://github.com/jinhaoduan/GTBench) (interactive
  game-theory benchmark), **vendored as plain source** rather than a git submodule, so a
  checkout needs no extra fetch step.

GTBench's LLM backend is overridden for local vLLM by `eval/gtbench/gtbench_patch/chat.py`,
and the Qwen model configs live in `eval/gtbench/gtbench_model_configs/`. Both are already
applied in the vendored tree above; see `eval/README.md` for how to re-apply them after
pulling a new GTBench, and for how they are used in the evaluation.
