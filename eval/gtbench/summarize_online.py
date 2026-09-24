"""Summarize the online veriselect arms: win-rate = (win + 0.5*draw)/normal per game.
Usage: python3 summarize_online.py <online_results_dir> "<space-separated games>"
"""
import glob
import json
import os
import sys

ROOT = sys.argv[1]
GAMES = (sys.argv[2].split() if len(sys.argv) > 2 else
         sorted(d for d in os.listdir(ROOT)))
ARMS = [d for d in ("single", "random8", "veriselect8") if os.path.isdir(os.path.join(ROOT, d))] or \
       sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d != "logs")


def winrate(arm, game):
    """(score, normal) for the LLM agent (model nick 'q25_7b-cot' in winner) in this arm/game."""
    files = glob.glob(os.path.join(ROOT, arm, game, "*.jsonl"))
    win = draw = normal = 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            for m in json.loads(line).get("matches", []):
                if m.get("status") != "Normal":
                    continue
                normal += 1
                w = m.get("winner", "")
                if w == "":
                    draw += 1
                elif "q25_7b-cot" in w:      # the LLM agent's model nick
                    win += 1
    score = (win + 0.5 * draw) / normal if normal else float("nan")
    return score, normal


print(f"\n# veriselect ONLINE win-rate (Qwen2.5-7B CoT vs random)  ({ROOT})\n")
hdr = f"{'game':<22}" + "".join(f"{a:>13}" for a in ARMS)
print(hdr)
tot = {a: [] for a in ARMS}
for g in GAMES:
    row = f"{g:<22}"
    for a in ARMS:
        s, n = winrate(a, g)
        row += f"{s:>10.3f}({n:>2})" if n else f"{'-':>13}"
        if n:
            tot[a].append(s)
    print(row)
avg = f"{'MEAN':<22}" + "".join(
    f"{(sum(tot[a]) / len(tot[a])):>13.3f}" if tot[a] else f"{'-':>13}" for a in ARMS)
print("-" * len(hdr)); print(avg)
if "veriselect8" in ARMS and "single" in ARMS and tot["single"] and tot["veriselect8"]:
    d = sum(tot["veriselect8"]) / len(tot["veriselect8"]) - sum(tot["single"]) / len(tot["single"])
    print(f"\n  mean Δ(veriselect8 - single/pass@1): {d:+.3f}")
