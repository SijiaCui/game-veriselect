"""
GameSolve — natural-language game descriptions.
===============================================
Surface renderings of a game. Two roles:

  * ID styles (``abstract`` / ``story`` / ``compact``) — the training-time formats.
  * OOD styles (``math`` / ``json`` / ``markdown`` / ``enumerated``) — held-out
    surface forms used only in the OOD split.

Action labels are deliberately *abstract and disjoint* across the two players
(rows A,B,C… ; columns …X,Y,Z). Semantic labels such as "Cooperate"/"Defect"
leak the answer through folk priors (everyone knows Defect is dominant in a PD),
so the harder benchmark strips them.
"""
from __future__ import annotations

import json
import string

_ALPHA = string.ascii_uppercase

CONTEXTS = [
    ("Market competition", "Firm 1", "Firm 2"),
    ("Negotiation", "Buyer", "Seller"),
    ("Network routing", "Router 1", "Router 2"),
    ("Resource allocation", "Agent 1", "Agent 2"),
    ("Abstract game", "Row player", "Column player"),
    ("Arms control", "Country A", "Country B"),
]


def make_labels(m, n):
    """Disjoint, prior-free labels: rows A.. ; cols ..Z (fallback R#/C#)."""
    rows = list(_ALPHA[:m])
    cols = list(_ALPHA[26 - n:26])
    if set(rows) & set(cols) or m > 26 or n > 26:
        rows = [f"R{i + 1}" for i in range(m)]
        cols = [f"C{j + 1}" for j in range(n)]
    return rows, cols


def _fmt(x):
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"


def _matrix_text(R, C, row_labels, col_labels):
    m, n = len(R), len(R[0])
    cells = [[f"({_fmt(R[i][j])}, {_fmt(C[i][j])})" for j in range(n)] for i in range(m)]
    w = max(len(str(l)) for l in row_labels)
    cw = max(max(len(c) for row in cells for c in row), max(len(str(l)) for l in col_labels))
    header = " " * (w + 2) + "  ".join(f"{c:>{cw}}" for c in col_labels)
    lines = [header]
    for i in range(m):
        lines.append(f"{row_labels[i]:>{w}}  " + "  ".join(f"{cells[i][j]:>{cw}}" for j in range(n)))
    return "\n".join(lines)


def _sigma_text(sigma, col_labels):
    return ", ".join(f"P({col_labels[j]})={sigma[j]:.4f}" for j in range(len(sigma)))


# ═══════════════════════════════════════════════════════════════════
#  base game body per style
# ═══════════════════════════════════════════════════════════════════
def game_body(R, C, row_labels, col_labels, context, style):
    _, role_r, role_c = context
    m, n = len(R), len(R[0])
    mat = _matrix_text(R, C, row_labels, col_labels)

    if style == "abstract":
        return (f"Consider a two-player strategic-form game.\n"
                f"{role_r} (row) has {m} actions: {{{', '.join(row_labels)}}}.\n"
                f"{role_c} (col) has {n} actions: {{{', '.join(col_labels)}}}.\n"
                f"Payoff matrix (row player, column player):\n\n{mat}\n\n"
                f"Each cell shows (payoff to {role_r}, payoff to {role_c}).")
    if style == "story":
        return (f"Scenario: {context[0]}. {role_r} and {role_c} act simultaneously.\n"
                f"{role_r} chooses from {{{', '.join(row_labels)}}}; "
                f"{role_c} chooses from {{{', '.join(col_labels)}}}.\n"
                f"Payoffs (first = {role_r}, second = {role_c}):\n\n{mat}")
    if style == "compact":
        rows = [f"  ({row_labels[i]},{col_labels[j]})=({_fmt(R[i][j])},{_fmt(C[i][j])})"
                for i in range(m) for j in range(n)]
        return f"Two-player game ({context[0]}). Outcomes:\n" + "\n".join(rows)
    if style == "math":
        lines = [f"Let Γ = (N, S, u) be a finite two-player game.",
                 f"  N = {{{role_r}, {role_c}}}",
                 f"  S_1 = {{{', '.join(row_labels)}}},  S_2 = {{{', '.join(col_labels)}}}",
                 f"  u_1 (row payoffs):"]
        lines += [f"    [{', '.join(_fmt(R[i][j]) for j in range(n))}]" for i in range(m)]
        lines += [f"  u_2 (column payoffs):"]
        lines += [f"    [{', '.join(_fmt(C[i][j]) for j in range(n))}]" for i in range(m)]
        return "\n".join(lines)
    if style == "json":
        data = {"players": [role_r, role_c],
                "strategies": {role_r: row_labels, role_c: col_labels},
                "payoffs": {f"({row_labels[i]},{col_labels[j]})":
                            [R[i][j], C[i][j]] for i in range(m) for j in range(n)}}
        return "Game specification (JSON):\n" + json.dumps(data, indent=2)
    if style == "markdown":
        head = "| | " + " | ".join(col_labels) + " |"
        sep = "|---|" + "|".join(["---"] * n) + "|"
        body = [f"| {row_labels[i]} | " +
                " | ".join(f"({_fmt(R[i][j])}, {_fmt(C[i][j])})" for j in range(n)) + " |"
                for i in range(m)]
        return (f"# {m}x{n} Strategic Game\n**{role_r}** (rows) vs **{role_c}** (columns)\n\n"
                + "\n".join([head, sep] + body))
    if style == "enumerated":
        lines = [f"Strategic interaction between {role_r} and {role_c}.",
                 f"{role_r} options: {', '.join(row_labels)}",
                 f"{role_c} options: {', '.join(col_labels)}", "Outcomes:"]
        lines += [f"  - {row_labels[i]} vs {col_labels[j]}: {role_r} gets {_fmt(R[i][j])}, "
                  f"{role_c} gets {_fmt(C[i][j])}" for i in range(m) for j in range(n)]
        return "\n".join(lines)
    raise ValueError(f"unknown style: {style}")


def nash_description(R, C, row_labels, col_labels, context, style):
    return game_body(R, C, row_labels, col_labels, context, style)


def br_description(R, C, row_labels, col_labels, context, sigma, style):
    _, role_r, role_c = context
    body = game_body(R, C, row_labels, col_labels, context, style)
    return (f"{body}\n\n{role_c} plays the mixed strategy: {_sigma_text(sigma, col_labels)}.\n"
            f"What is {role_r}'s best response?")


ID_STYLES = ["abstract", "story", "compact"]
OOD_STYLES = ["math", "json", "markdown", "enumerated"]
