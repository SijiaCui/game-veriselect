"""
GameSolve-Hard — dataset generator (redesigned).
================================================
Two tasks, four difficulty tiers, knob-controlled.  Produces JSONL whose schema
is a superset of the original GameSolve-Bench (same fields consumed by the
generation / verifier / reward pipeline, plus a ``metadata`` block with the
difficulty knobs and per-sample audit fields).

Tiers (best_response):
  easy    2×2–2×3, peaked opponent, well-separated payoffs      (baseline, for comparability)
  medium  3×3–3×4, α=1 opponent
  hard    4×4–5×4, flat opponent, small gap, adversarial traps
  expert  5×5–6×6, ~uniform opponent, tiny gap, adversarial, 2-decimal payoffs, ties

Tiers (nash_equilibrium):
  easy    2×2 general
  medium  3×3 general / zero-sum
  hard    3×3–4×4, a mixed NE is required (no pure NE) or several pure NE to enumerate
  expert  4×4–5×5, mixed required + strictly-dominated distractors (IESDS first), 2-decimal

Usage:
  python generate.py --out_dir . --scale 1.0 --seed 42
  python generate.py --out_dir . --scale 0.1            # quick smoke run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import numpy as np

import games as G
import solvers as S
import descriptions as D
import cot as COT


# ═══════════════════════════════════════════════════════════════════
#  tier -> base knobs
# ═══════════════════════════════════════════════════════════════════
BR_TIERS = {
    "easy":   dict(opp_alpha=0.5, gap_min=1.0, adversarial=False, integer=True, low=-5, high=5),
    "medium": dict(opp_alpha=1.0, adversarial=False, integer=True, low=-5, high=5),
    "hard":   dict(opp_alpha=4.0, gap_max=1.0, adversarial=True, integer=True, low=-5, high=5),
    # expert eased so a strong model can occasionally solve it (oracle@N > 0): smaller
    # gap floor lifted (0.5->0.8) and opponent a touch less flat (8->6).
    "expert": dict(opp_alpha=6.0, gap_max=0.8, adversarial=True, force_ties=False,
                   integer=False, low=-5, high=5),
}
NASH_TIERS = {
    "easy":   dict(integer=True, low=-5, high=5),
    "medium": dict(min_equilibria=1, integer=True, low=-5, high=5),
    "hard":   dict(require_mixed=True, integer=True, low=-5, high=5),
    # expert eased: one distractor per side (was 2 rows / 1 col) so support stays findable.
    "expert": dict(require_mixed=True, distract_rows=1, distract_cols=1, integer=False, low=-5, high=5),
}

# (task, tier, m, n, game_type, count, overrides)
PLAN_ID = [
    # best_response
    ("br", "easy",   2, 2, "general", 120, None),
    ("br", "easy",   2, 3, "general",  80, None),
    ("br", "medium", 3, 3, "general", 150, None),
    ("br", "medium", 3, 4, "general",  80, None),
    ("br", "hard",   4, 4, "general", 150, None),
    ("br", "hard",   4, 4, "zero_sum", 60, None),
    ("br", "hard",   5, 4, "general",  80, None),
    ("br", "expert", 4, 4, "general", 120, None),      # was 5x5
    ("br", "expert", 5, 5, "general",  60, None),      # was 6x6
    # nash_equilibrium
    ("nash", "easy",   2, 2, "general",  150, None),
    ("nash", "medium", 3, 3, "general",  150, None),
    ("nash", "medium", 3, 3, "zero_sum",  60, None),
    ("nash", "medium", 3, 3, "symmetric", 80, None),
    ("nash", "medium", 3, 3, "general",   80, dict(require_multi_pure=True)),  # coordination
    ("nash", "hard",   3, 3, "general",  120, None),
    ("nash", "hard",   4, 4, "general",  100, None),
    ("nash", "hard",   4, 4, "symmetric", 60, None),
    ("nash", "expert", 4, 4, "general",  100, None),
    ("nash", "expert", 4, 4, "general",   60, dict(distract_rows=1, distract_cols=1)),  # was 5x5
]

# OOD: genuine difficulty shift (larger, wider-range, 2-decimal, held-out surface forms,
# asymmetric, distractors) — but eased from the extreme so a strong model can occasionally
# solve it (oracle@N > 0). 8th element = ood_category (for per-category analysis).
PLAN_OOD = [
    ("br", "hard",   5, 5, "general",  40, dict(low=-15, high=15, gap_max=2.0), "large_wide"),
    ("br", "expert", 5, 6, "general",  40, dict(low=-15, high=15, gap_max=2.0), "large_wide"),
    ("br", "hard",   4, 6, "general",  30, None, "asymmetric"),
    ("nash", "hard",   4, 4, "general", 40, dict(low=-15, high=15), "large_wide"),
    ("nash", "expert", 4, 4, "general", 40, dict(distract_rows=2, distract_cols=1), "distractor_heavy"),
    ("nash", "hard",   3, 5, "general", 30, None, "asymmetric"),
]


# ═══════════════════════════════════════════════════════════════════
#  sample builders (schema = original superset)
# ═══════════════════════════════════════════════════════════════════
def _labels_and_context(m, n, rng):
    row_labels, col_labels = D.make_labels(m, n)
    ctx = D.CONTEXTS[int(rng.integers(len(D.CONTEXTS)))]
    return row_labels, col_labels, ctx


def build_br_sample(m, n, game_type, tier, overrides, styles, rng, ood_category=None):
    knobs = dict(BR_TIERS[tier]); knobs.update(overrides or {})
    R, C, sigma, br, meta = G.build_br_game(m, n, game_type, knobs, rng)
    row_labels, col_labels, ctx = _labels_and_context(m, n, rng)
    _, role_r, role_c = ctx

    descs = {st: D.br_description(R, C, row_labels, col_labels, ctx, sigma, st) for st in styles}
    cot = COT.br_cot(R, sigma, row_labels, col_labels, br, role_c)
    sid = hashlib.md5(json.dumps({"R": R, "s": sigma, "t": tier}).encode()).hexdigest()[:10]
    md = {"difficulty": tier, "knobs": knobs, **meta}
    if ood_category:
        md["ood_category"] = ood_category
    return {
        "id": f"br_{tier}_{m}x{n}_{game_type}_{sid}",
        "task": "best_response", "game_type": game_type, "dimensions": [m, n],
        "row_labels": row_labels, "col_labels": col_labels,
        "payoff_matrix_row": R, "payoff_matrix_col": C, "opponent_sigma": sigma,
        "context": ctx[0], "role_row": role_r, "role_col": role_c,
        "descriptions": descs, "ground_truth": br, "chain_of_thought": cot,
        "metadata": md,
    }


def build_nash_sample(m, n, game_type, tier, overrides, styles, rng, ood_category=None):
    knobs = dict(NASH_TIERS[tier]); knobs.update(overrides or {})
    R, C, pure, eqs, meta = G.build_nash_game(m, n, game_type, knobs, rng)
    if not eqs:
        return None
    m2, n2 = len(R), len(R[0])                       # dims may grow with distractors
    row_labels, col_labels, ctx = _labels_and_context(m2, n2, rng)
    _, role_r, role_c = ctx

    gt_eqs = []
    for s1, s2 in eqs:
        eu_r, eu_c = S.compute_nash_payoffs(R, C, s1, s2)
        gt_eqs.append({"sigma_row": s1, "sigma_col": s2,
                       "is_pure": S.is_pure_eq((s1, s2)), "payoffs": [eu_r, eu_c]})
    ground_truth = {
        "equilibria": gt_eqs, "n_equilibria": len(eqs),
        "equilibrium_class": S.classify_equilibria(eqs),
        "pure_ne_count": len(pure), "mixed_ne_count": len(eqs) - len(pure),
    }
    descs = {st: D.nash_description(R, C, row_labels, col_labels, ctx, st) for st in styles}
    cot = COT.nash_cot(R, C, row_labels, col_labels, pure, eqs, role_r, role_c)
    sid = hashlib.md5(json.dumps({"R": R, "C": C, "t": tier}).encode()).hexdigest()[:10]
    md = {"difficulty": tier, "knobs": knobs, **meta}
    if ood_category:
        md["ood_category"] = ood_category
    return {
        "id": f"nash_{tier}_{m2}x{n2}_{game_type}_{sid}",
        "task": "nash_equilibrium", "game_type": game_type, "dimensions": [m2, n2],
        "row_labels": row_labels, "col_labels": col_labels,
        "payoff_matrix_row": R, "payoff_matrix_col": C,
        "context": ctx[0], "role_row": role_r, "role_col": role_c,
        "descriptions": descs, "ground_truth": ground_truth, "chain_of_thought": cot,
        "metadata": md,
    }


# ═══════════════════════════════════════════════════════════════════
#  driver
# ═══════════════════════════════════════════════════════════════════
def generate(plan, styles_id, styles_ood, scale, rng):
    samples = []
    for entry in plan:
        task, tier, m, n, gtype, count, ovr = entry[:7]
        ood_category = entry[7] if len(entry) > 7 else None
        target = max(1, int(round(count * scale)))
        styles = styles_ood if styles_ood is not None else styles_id
        got, attempts, max_attempts = 0, 0, target * 8
        while got < target and attempts < max_attempts:
            attempts += 1
            try:
                if task == "br":
                    s = build_br_sample(m, n, gtype, tier, ovr, styles, rng, ood_category)
                else:
                    s = build_nash_sample(m, n, gtype, tier, ovr, styles, rng, ood_category)
                    if s is None:
                        continue
            except Exception:
                continue
            samples.append(s); got += 1
        print(f"  {task:4s} {tier:6s} {m}x{n} {gtype:8s}: {got}/{target}", flush=True)
    return samples


def _stats(samples):
    st = {"total": len(samples), "by_task": dict(Counter(s["task"] for s in samples)),
          "by_tier": dict(Counter(s["metadata"]["difficulty"] for s in samples))}
    br = [s for s in samples if s["task"] == "best_response"]
    if br:
        adv = [s for s in br if s["metadata"]["knobs"].get("adversarial")]
        st["br"] = {
            "n": len(br),
            "unique_frac": round(np.mean([s["ground_truth"]["is_unique"] for s in br]), 3),
            "mean_eu_gap": round(np.mean([s["metadata"]["eu_gap"] for s in br]), 3),
            "mode_shortcut_hit": round(np.mean([s["metadata"]["shortcut_mode_column_correct"] for s in br]), 3),
            "gmax_shortcut_hit": round(np.mean([s["metadata"]["shortcut_global_max_correct"] for s in br]), 3),
            "adversarial_met_frac": (round(np.mean([s["metadata"]["constraints_met"]["adversarial"]
                                                    for s in adv]), 3) if adv else None),
            "mean_opp_entropy": round(np.mean([s["metadata"]["opp_entropy"] for s in br]), 3),
        }
    ne = [s for s in samples if s["task"] == "nash_equilibrium"]
    if ne:
        st["nash"] = {
            "n": len(ne),
            "class_dist": dict(Counter(s["ground_truth"]["equilibrium_class"] for s in ne)),
            "mixed_required_frac": round(np.mean([s["ground_truth"]["pure_ne_count"] == 0 for s in ne]), 3),
            "mean_n_equilibria": round(np.mean([s["ground_truth"]["n_equilibria"] for s in ne]), 3),
            "mean_dominated": round(np.mean([s["metadata"]["n_dominated_strategies"] for s in ne]), 3),
        }
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every plan count")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[ID]")
    id_samples = generate(PLAN_ID, D.ID_STYLES, None, args.scale, rng)
    print("[OOD]")
    ood_samples = generate(PLAN_OOD, D.ID_STYLES, D.OOD_STYLES, args.scale, rng)

    for name, samples in [("gamesolve_hard", id_samples), ("gamesolve_hard_ood", ood_samples)]:
        with open(f"{args.out_dir}/{name}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        stats = _stats(samples)
        with open(f"{args.out_dir}/{name}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n{name}: {stats['total']} samples -> {name}.jsonl")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
