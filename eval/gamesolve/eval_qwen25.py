"""Qwen2.5 GameSolve-Hard eval: non-thinking, 8192/10240 token budget.

Thin entry over the shared engine (eval_core.py) — only the Qwen2.5 series
defaults live here; sampling params (temp 0.6, top_p 0.95, no top_k, gpu_mem
0.90) come from the engine and are identical to the Qwen3 entry.
"""
from eval_core import build_parser, run

if __name__ == "__main__":
    p = build_parser()
    p.set_defaults(enable_thinking=False, max_new_tokens=8192, max_model_len=10240)
    run(p.parse_args())
