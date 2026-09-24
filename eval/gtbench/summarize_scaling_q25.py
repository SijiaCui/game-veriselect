"""Summarize the Qwen2.5 GTBench scaling sweep: per model, mean win-rate for
pass@1 / random@8 / veriselect@8 over the 7 games, + the veriselect deltas.
LLM agent won iff 'cot' in winner (the CoT nick); '' = draw; else = random opponent won.
Usage: python3 summarize_scaling_q25.py [scaling_results_dir]
"""
import glob, json, math, os, sys
from statistics import mean

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(__file__), "..", "results", "gtbench_veriselect", "scaling_q25")
# size order (tag -> label)
ORDER = [("q25_0_5b", "0.5B"), ("q25_1_5b", "1.5B"), ("q25_3b", "3B"), ("q25_7b", "7B"),
         ("q25_14b", "14B"), ("q25_32b", "32B"), ("q25_72b", "72B")]
ARMS = [("single", "pass@1"), ("random8", "random@8"), ("veriselect8", "veriselect@8")]
GAMES = ["tictactoe", "connect4", "kuhn_poker", "nim",
         "prisoners_dilemma", "liars_dice", "first_sealed_auction"]


def winrate(tag, arm, game):
    win = draw = normal = 0
    for f in glob.glob(os.path.join(ROOT, tag, arm, game, "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            for m in json.loads(line).get("matches", []):
                if m.get("status") != "Normal":
                    continue
                normal += 1
                w = (m.get("winner") or "")
                if w == "":
                    draw += 1
                elif "cot" in w.lower():
                    win += 1
    if not normal:
        return None
    r = (win + 0.5 * draw) / normal
    return r, normal


def se(r, n):
    return math.sqrt(max(r * (1 - r), 1e-9) / n) if n else 0.0


print(f"\n# Qwen2.5 GTBench veriselect SCALING  ({ROOT})\n")
print(f"{'model':<7}{'pass@1':>9}{'rand@8':>9}{'vsel@8':>9}"
      f"{'Δvs-p@1':>10}{'Δvs-r@8':>10}   per-game veriselect@8 Δvs-pass@1")
for tag, label in ORDER:
    if not os.path.isdir(os.path.join(ROOT, tag)):
        continue
    arm_means = {}
    per_game_dvs = []
    for arm, _ in ARMS:
        rs = [winrate(tag, arm, g) for g in GAMES]
        vals = [r for r in rs if r]
        arm_means[arm] = mean(v[0] for v in vals) if vals else None
    # per-game veriselect Δ vs pass@1
    detail = []
    for g in GAMES:
        s = winrate(tag, "single", g); v = winrate(tag, "veriselect8", g)
        if s and v:
            per_game_dvs.append(v[0] - s[0])
            detail.append(f"{g[:4]}{v[0]-s[0]:+.2f}")
    p1 = arm_means["single"]; r8 = arm_means["random8"]; v8 = arm_means["veriselect8"]
    if p1 is None or v8 is None:
        print(f"{label:<7}  (incomplete)")
        continue
    dvp = v8 - p1
    dvr = (v8 - r8) if r8 is not None else float("nan")
    print(f"{label:<7}{p1:>9.3f}{r8 if r8 is not None else float('nan'):>9.3f}{v8:>9.3f}"
          f"{dvp:>+10.3f}{dvr:>+10.3f}   {' '.join(detail)}")
print("\n(win-rate = (win+0.5*draw)/normal vs random_agent, mean over 7 games. "
      "Δvs-p@1 = veriselect@8 - pass@1 ; Δvs-r@8 isolates the verifier from the more-samples effect.)")
