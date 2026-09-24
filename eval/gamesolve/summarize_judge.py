"""
judge@8 summary for the Qwen2.5 GameSolve series. Computes best-of-8 HOLISTIC-judge selection
accuracy from the *_rewards fields written by judge_reward.py:
  jL72b = llm-judge  (Qwen2.5-72B judges everyone)
  jSelf = self-judge (each model judges its own traces; 72B-self aliased to jL72b)
and prints them beside pass@1 / gold-free vsel@8 / oracle@8 (vsel_gf + oracle read from
veriselect_at8.json when present). Same select() + per-prompt tiebreak as judge_reward.
Usage: python3 eval/gamesolve/summarize_judge.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_reward import select, _pp_rng

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gamesolve")
MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
          "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct",
          "Qwen2.5-72B-Instruct"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sel_acc(pp, rw_key, seed):
    """best-of-N correctness selecting by the rw_key rewards; None if that judge didn't score it."""
    rw = pp.get(rw_key)
    if rw_key == "judge_self_rewards" and not rw:
        rw = pp.get("judge_llm72b_rewards")          # 72B-self == the llm-judge on the 72B file
    if not rw:
        return None
    return float(pp["exacts"][select(rw, _pp_rng(seed, pp["id"]))])


def main():
    at8 = {}
    p8 = os.path.join(RES, "veriselect_at8.json")
    if os.path.exists(p8):
        at8 = json.load(open(p8))

    hdr = (f"{'model':<22}{'split':<5}{'pass@1':>8}{'vsel_gf':>8}{'jL72b':>8}{'jSelf':>8}"
           f"{'oracle':>8}{'dJL':>8}{'dJS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in MODELS:
        fp = os.path.join(RES, m + ".json")
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp))
        seed = d.get("config", {}).get("seed", 42)
        for split in ("ID", "OOD"):
            pps = [pp for pp in d["per_prompt"] if pp["split"] == split]
            if not pps:
                continue
            pass1 = mean([pp["pass1"] for pp in pps])
            oracle = mean([pp["oracle"] for pp in pps])
            jL = mean([sel_acc(pp, "judge_llm72b_rewards", seed) for pp in pps])
            jS = mean([sel_acc(pp, "judge_self_rewards", seed) for pp in pps])
            a = at8.get(m, {}).get(split) or {}
            vgf = a.get("veriselect_gf")

            def f(x):
                return f"{x:8.3f}" if x is not None else f"{'-':>8}"
            dJL = f"{jL - pass1:+8.3f}" if jL is not None else f"{'-':>8}"
            dJS = f"{jS - pass1:+8.3f}" if jS is not None else f"{'-':>8}"
            print(f"{m:<22}{split:<5}{pass1:8.3f}{f(vgf)}{f(jL)}{f(jS)}{oracle:8.3f}{dJL}{dJS}")
    print("\n(jL72b=Qwen2.5-72B holistic judge best-of-8; jSelf=self-judge; dJ*=judge@8 - pass@1. "
          "vsel_gf=gold-free CLAIM-level verifier best-of-8 from veriselect_at8.json.)")


if __name__ == "__main__":
    main()
