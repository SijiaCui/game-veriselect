"""Qwen3 GameSolve-Hard eval: 16384/18432 token budget; --enable_thinking toggles thinking.

Thin entry over the shared engine (eval_core.py) — only the Qwen3 series
defaults live here; sampling params (temp 0.6, top_p 0.95, no top_k, gpu_mem
0.90) come from the engine and are identical to the Qwen2.5 entry. Thinking is
off by default (matches run_gamesolve.sh); pass --enable_thinking for a full
thinking eval. The NE-tier thinking diagnostic stays in eval_think_diag.py.
"""
from eval_core import build_parser, run

if __name__ == "__main__":
    p = build_parser()
    p.set_defaults(enable_thinking=False, max_new_tokens=16384, max_model_len=18432)
    run(p.parse_args())
