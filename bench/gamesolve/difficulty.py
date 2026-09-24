"""
GameSolve — offline difficulty audit (no GPU / no model).
=========================================================
Two jobs:

  1. audit()  — per (task, tier) difficulty profile straight from the ground truth
     and generation metadata: how often a trivial heuristic shortcuts the answer,
     expected-payoff separation, opponent flatness, mixed-NE requirement, distractor
     load. This is how we verify the redesign is *actually* harder before spending
     any GPU on generation.

  2. validate_gold() — run each sample's reference chain-of-thought through the
     redesigned reward. Gold traces must score ~1.0 / exact; anything less means the
     ANSWER schema and the parser have drifted apart.

Usage:
  python difficulty.py gamesolve_hard.jsonl
  python difficulty.py gamesolve_hard.jsonl --gold
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import reward as W


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def audit(path):
    samples = _load(path)
    groups = defaultdict(list)
    for s in samples:
        groups[(s["task"], s["metadata"]["difficulty"])].append(s)

    order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
    print(f"\n=== difficulty audit: {path}  ({len(samples)} samples) ===\n")

    # best_response
    br_keys = sorted([k for k in groups if k[0] == "best_response"], key=lambda k: order.get(k[1], 9))
    if br_keys:
        print("BEST_RESPONSE")
        print(f"  {'tier':7s} {'n':>4s} {'#act':>5s} {'gap':>6s} {'norm_gap':>8s} "
              f"{'opp_H':>6s} {'mode_hit':>8s} {'gmax_hit':>8s} {'uniq':>5s}")
        for k in br_keys:
            g = groups[k]
            n_act = np.mean([s["dimensions"][0] for s in g])
            gap = np.mean([s["metadata"]["eu_gap"] for s in g])
            eus = [np.array(s["ground_truth"]["expected_payoffs"]) for s in g]
            ngap = np.mean([s["metadata"]["eu_gap"] / (e.max() - e.min() + 1e-9) for s, e in zip(g, eus)])
            H = np.mean([s["metadata"]["opp_entropy"] for s in g])
            mode = np.mean([s["metadata"]["shortcut_mode_column_correct"] for s in g])
            gmax = np.mean([s["metadata"]["shortcut_global_max_correct"] for s in g])
            uniq = np.mean([s["ground_truth"]["is_unique"] for s in g])
            print(f"  {k[1]:7s} {len(g):4d} {n_act:5.1f} {gap:6.2f} {ngap:8.2f} "
                  f"{H:6.2f} {mode:8.2f} {gmax:8.2f} {uniq:5.2f}")
        print("  (mode_hit/gmax_hit = fraction a trivial heuristic already gets right; lower = harder)")

    # nash_equilibrium
    ne_keys = sorted([k for k in groups if k[0] == "nash_equilibrium"], key=lambda k: order.get(k[1], 9))
    if ne_keys:
        print("\nNASH_EQUILIBRIUM")
        print(f"  {'tier':7s} {'n':>4s} {'#eq':>5s} {'mixed_req':>9s} {'has_mixed':>9s} "
              f"{'dominated':>9s} {'iesds':>6s}")
        for k in ne_keys:
            g = groups[k]
            neq = np.mean([s["ground_truth"]["n_equilibria"] for s in g])
            mreq = np.mean([s["ground_truth"]["pure_ne_count"] == 0 for s in g])
            hmix = np.mean([s["ground_truth"]["mixed_ne_count"] > 0 for s in g])
            dom = np.mean([s["metadata"]["n_dominated_strategies"] for s in g])
            iesds = np.mean([s["metadata"].get("iesds_eliminated", 0) for s in g])
            print(f"  {k[1]:7s} {len(g):4d} {neq:5.2f} {mreq:9.2f} {hmix:9.2f} {dom:9.2f} {iesds:6.2f}")
        print("  (mixed_req = fraction with NO pure NE, so a mixed NE must be solved; higher = harder)")


def validate_gold(path):
    samples = _load(path)
    rewards, exacts = [], []
    fails = []
    for s in samples:
        b = W.reward_breakdown(s["chain_of_thought"], s["ground_truth"],
                               s["task"], s["row_labels"], s["col_labels"])
        rewards.append(b["reward"]); exacts.append(b["exact"])
        if not b["exact"]:
            fails.append((s["id"], round(b["reward"], 3), b["components"]))
    print(f"\n=== gold-trace validation: {path} ===")
    print(f"  mean reward = {np.mean(rewards):.4f}   exact-match rate = {np.mean(exacts):.4f}   (target ~1.0)")
    if fails:
        print(f"  {len(fails)} non-exact gold traces (schema/parser drift):")
        for fid, r, c in fails[:10]:
            print(f"    {fid}  reward={r}  {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gold", action="store_true", help="also validate gold CoT traces")
    args = ap.parse_args()
    audit(args.path)
    if args.gold:
        validate_gold(args.path)
