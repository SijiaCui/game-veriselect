# bench

Benchmark data and generators, plus external benchmarks used for evaluation.

- `gamesolve/` — in-house GameSolve-Bench matrix-game data/generators.
- `GTBench/` — [jinhaoduan/GTBench](https://github.com/jinhaoduan/GTBench) (interactive
  game-theory benchmark), **vendored as plain source** rather than a git submodule, so a
  checkout needs no extra fetch step.

GTBench's LLM backend is overridden for local vLLM by `eval/gtbench/gtbench_patch/chat.py`,
and the Qwen model configs live in `eval/gtbench/gtbench_model_configs/`. Both live outside
this tree and are **not** applied to the vendored source on checkout — run
`eval/gtbench/setup.sh` to copy them in (required before the GTBench runners will work).
See `eval/README.md` for how they are used in the evaluation.
