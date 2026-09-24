"""
GameSolve — solver-grounded reference chain-of-thought (gold traces).
=====================================================================
Every trace here is generated directly from the exact solver output, so it is
correct by construction. Each ends with the canonical ANSWER block, so a gold
trace is a valid, high-scoring response under both the process verifier
(EU / BR_ACTION / EQUILIBRIUM / DOMINANCE claim extraction) and the outcome reward.
"""
from __future__ import annotations

import json

import numpy as np

import solvers as S
from descriptions import _fmt, _sigma_text


def br_cot(R, sigma, row_labels, col_labels, br, role_c):
    n = len(sigma)
    lines = ["## Step 1: Opponent strategy",
             f"{role_c} plays {_sigma_text(sigma, col_labels)}.",
             "", "## Step 2: Expected payoff of each row action"]
    for i, a in enumerate(row_labels):
        terms = " + ".join(f"{_fmt(R[i][j])}×{sigma[j]:.4f}" for j in range(n))
        lines.append(f"EU({a}) = {terms} = {br['expected_payoffs'][i]:.4f}")
    br_names = [row_labels[i] for i in br["best_response_actions"]]
    lines += ["", "## Step 3: Best response",
              f"Maximum expected payoff = {br['best_response_value']:.4f}, attained by "
              f"{{{', '.join(br_names)}}}."]
    if not br["is_unique"]:
        lines.append("Several actions tie, so any mixture over them is also a best response.")
    answer = {"best_response_actions": br_names,
              "expected_payoffs": [round(float(v), 4) for v in br["expected_payoffs"]],
              "best_response_value": round(float(br["best_response_value"]), 4)}
    lines += ["", "ANSWER:", "```json", json.dumps(answer), "```"]
    return "\n".join(lines)


def nash_cot(R, C, row_labels, col_labels, pure, eqs, role_r, role_c):
    R, C = np.asarray(R, float), np.asarray(C, float)
    m, n = R.shape
    lines = ["## Step 1: Game structure",
             f"A {m}×{n} two-player normal-form game."]

    dom = S.iesds(R, C)
    if dom["steps"]:
        lines += ["", "## Step 2: Eliminate strictly dominated strategies"]
        for who, idx, by in dom["steps"]:
            labs = row_labels if who == "row" else col_labels
            lines.append(f"{'Row' if who=='row' else 'Column'} {labs[idx]} is strictly "
                         f"dominated by {labs[by]}; remove it.")

    lines += ["", "## Step 3: Pure-strategy Nash equilibria (best-response check)"]
    if pure:
        for i, j in pure:
            lines.append(f"({row_labels[i]}, {col_labels[j]}) is a Nash equilibrium: "
                         f"neither player can profitably deviate.")
    else:
        lines.append("No pure-strategy Nash equilibrium exists.")

    mixed = [e for e in eqs if not S.is_pure_eq(e)]
    lines += ["", "## Step 4: Mixed-strategy Nash equilibria"]
    if mixed:
        for s1, s2 in mixed:
            lines.append(f"{role_r} mixes {[round(x,4) for x in s1]}, "
                         f"{role_c} mixes {[round(x,4) for x in s2]} "
                         f"(each player indifferent across their support).")
    else:
        lines.append("No additional mixed-strategy equilibrium.")

    answer = {"pure_ne": [[row_labels[i], col_labels[j]] for i, j in pure],
              "mixed_ne": [[[round(float(x), 4) for x in s1], [round(float(x), 4) for x in s2]]
                           for s1, s2 in mixed]}
    lines += ["", "ANSWER:", "```json", json.dumps(answer), "```"]
    return "\n".join(lines)
