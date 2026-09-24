"""
GameSolve — exact, verifiable solvers (2-player normal-form).
=============================================================
These are the ground-truth oracles for both benchmark tasks. Everything here is
deterministic and closed-form (support enumeration via nashpy; pure best response
via a matrix-vector product; strict dominance / IESDS by direct comparison), so
the generated ground truth is exact and independently checkable.

Nothing here samples games — see ``games.py`` for the (knob-controlled) samplers.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np
import nashpy as nash

TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
#  Nash equilibria
# ═══════════════════════════════════════════════════════════════════
def find_pure_nash(R, C) -> list:
    """All pure-strategy Nash equilibria as (row_idx, col_idx) pairs."""
    R, C = np.asarray(R, float), np.asarray(C, float)
    m, n = R.shape
    eqs = []
    for i, j in itertools.product(range(m), range(n)):
        if R[i, j] >= R[:, j].max() - TOL and C[i, j] >= C[i, :].max() - TOL:
            eqs.append((i, j))
    return eqs


def find_all_nash_checked(R, C, tol: float = 1e-6):
    """All Nash equilibria (pure + mixed) via nashpy support enumeration,
    de-duplicated as ``(sigma_row, sigma_col)`` python lists, plus a degeneracy flag.

    nashpy warns (and may return an even / incomplete set of equilibria) on
    degenerate games — those have equilibrium continua and are unsafe as exact
    ground truth, so the generator rejects them. Returns ``(eqs, degenerate)``.
    """
    R, C = np.asarray(R, float), np.asarray(C, float)
    game = nash.Game(R, C)
    eqs = []
    degenerate = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            for s1, s2 in game.support_enumeration():
                if (s1 >= -tol).all() and (s2 >= -tol).all():
                    s1 = np.clip(s1, 0, None); s1 = s1 / s1.sum()
                    s2 = np.clip(s2, 0, None); s2 = s2 / s2.sum()
                    eqs.append((np.round(s1, 6).tolist(), np.round(s2, 6).tolist()))
        except Exception:
            degenerate = True
        if any("degenerate" in str(w.message).lower() for w in caught):
            degenerate = True
    unique = []
    for eq in eqs:
        if not any(_eq_close(eq, u) for u in unique):
            unique.append(eq)
    return unique, degenerate


def _eq_close(a, b, tol: float = 1e-4) -> bool:
    return (len(a[0]) == len(b[0]) and len(a[1]) == len(b[1])
            and np.allclose(a[0], b[0], atol=tol)
            and np.allclose(a[1], b[1], atol=tol))


def _is_pure(sigma, tol: float = 1e-4) -> bool:
    return any(abs(p - 1.0) < tol for p in sigma)


def is_pure_eq(eq, tol: float = 1e-4) -> bool:
    """True iff BOTH players play a pure strategy in ``eq = (sigma_row, sigma_col)``."""
    return _is_pure(eq[0], tol) and _is_pure(eq[1], tol)


def classify_equilibria(eqs: list) -> str:
    """pure / mixed / both / none."""
    pure  = any(is_pure_eq(e) for e in eqs)
    mixed = any(not is_pure_eq(e) for e in eqs)
    if pure and mixed: return "both"
    if pure:           return "pure"
    if mixed:          return "mixed"
    return "none"


def compute_nash_payoffs(R, C, sigma_r, sigma_c) -> tuple:
    R, C = np.asarray(R, float), np.asarray(C, float)
    s1, s2 = np.asarray(sigma_r, float), np.asarray(sigma_c, float)
    return round(float(s1 @ R @ s2), 6), round(float(s1 @ C @ s2), 6)


# ═══════════════════════════════════════════════════════════════════
#  Best response (row player vs a FIXED column-player mixed strategy)
# ═══════════════════════════════════════════════════════════════════
def compute_best_response(R, opponent_sigma) -> dict:
    R = np.asarray(R, float)
    sigma = np.asarray(opponent_sigma, float)
    eu = R @ sigma
    max_val = eu.max()
    br_actions = np.where(np.abs(eu - max_val) < TOL)[0].tolist()
    br_mixed = np.zeros(len(eu))
    for a in br_actions:
        br_mixed[a] = 1.0 / len(br_actions)
    return {
        "expected_payoffs": np.round(eu, 6).tolist(),
        "best_response_actions": br_actions,
        "best_response_value": round(float(max_val), 6),
        "is_unique": len(br_actions) == 1,
        "br_mixed_strategy": np.round(br_mixed, 6).tolist(),
    }


def eu_gap(eu) -> float:
    """Gap between the best expected payoff and the best *strictly-worse* one.

    Returns 0.0 only when *every* action ties the maximum; a partial tie such as
    ``[1.5, 1.5, 1.0]`` returns the gap to the next distinct value (0.5). Whether
    the best response is unique is reported separately by ``is_unique``.
    """
    eu = np.sort(np.asarray(eu, float))[::-1]
    top = eu[0]
    worse = eu[eu < top - TOL]
    return float(top - worse.max()) if worse.size else 0.0


def br_shortcut_actions(R, opponent_sigma) -> dict:
    """Actions returned by the two trivial BR heuristics that make the task easy:

      - ``mode_column`` : best-respond to the opponent's single most-likely column
                          (argmax_i R[i, argmax(sigma)]) — solves ~80% of the old bench.
      - ``global_max``  : the row containing R's global-max entry.

    Used both to *construct* adversarial games (where these disagree with the true
    best response) and to *audit* how many samples a heuristic can shortcut.
    """
    R = np.asarray(R, float)
    sigma = np.asarray(opponent_sigma, float)
    mode_col = int(np.argmax(sigma))
    return {
        "mode_column": int(np.argmax(R[:, mode_col])),
        "global_max": int(np.unravel_index(np.argmax(R), R.shape)[0]),
    }


# ═══════════════════════════════════════════════════════════════════
#  Strict dominance & iterated elimination (IESDS)
# ═══════════════════════════════════════════════════════════════════
def row_strictly_dominated(R, i, active_rows, active_cols) -> Optional[int]:
    """Return a row k that strictly dominates row i (over active cols), else None."""
    R = np.asarray(R, float)
    for k in active_rows:
        if k == i:
            continue
        if all(R[k, j] > R[i, j] + TOL for j in active_cols):
            return k
    return None


def col_strictly_dominated(C, j, active_rows, active_cols) -> Optional[int]:
    """Return a col l that strictly dominates col j (over active rows), else None."""
    C = np.asarray(C, float)
    for l in active_cols:
        if l == j:
            continue
        if all(C[i, l] > C[i, j] + TOL for i in active_rows):
            return l
    return None


def iesds(R, C) -> dict:
    """Iterated elimination of strictly dominated pure strategies.

    Returns surviving row/col indices and the ordered elimination trace
    (player, eliminated_index, dominating_index).
    """
    R, C = np.asarray(R, float), np.asarray(C, float)
    m, n = R.shape
    rows, cols = list(range(m)), list(range(n))
    steps = []
    changed = True
    while changed:
        changed = False
        for i in list(rows):
            k = row_strictly_dominated(R, i, rows, cols)
            if k is not None:
                rows.remove(i); steps.append(("row", i, k)); changed = True
        for j in list(cols):
            l = col_strictly_dominated(C, j, rows, cols)
            if l is not None:
                cols.remove(j); steps.append(("col", j, l)); changed = True
    return {"surviving_rows": rows, "surviving_cols": cols, "steps": steps,
            "n_eliminated": len(steps)}


def count_dominated(R, C) -> int:
    """Number of pure strategies strictly dominated in the *original* game."""
    m, n = np.asarray(R).shape
    allr, allc = list(range(m)), list(range(n))
    r = sum(1 for i in allr if row_strictly_dominated(R, i, allr, allc) is not None)
    c = sum(1 for j in allc if col_strictly_dominated(C, j, allr, allc) is not None)
    return r + c
