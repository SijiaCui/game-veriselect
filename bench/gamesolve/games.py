"""
GameSolve — knob-controlled game samplers / constructors.
=========================================================
Each constructor turns a set of *difficulty knobs* into a concrete game plus its
exact ground truth. Knobs are the levers identified in the benchmark audit:

  best_response
    opp_alpha   Dirichlet concentration for the opponent mixed strategy.
                small (0.3) -> peaked  (a "respond to the mode column" shortcut works)
                large (8.0) -> ~uniform (the full expectation is required)
    gap_max     cap the best-vs-2nd expected-payoff gap (small -> near-tie, exact
                arithmetic required to break it).
    adversarial require the two trivial heuristics (mode-column / global-max row)
                to BOTH disagree with the true best response -> shortcuts are traps.
    force_ties  make the best response a *set* (non-unique) -> harder to report.

  nash_equilibrium
    require_mixed     no pure NE exists, so a mixed NE must be solved for.
    require_multi_pure  >=2 pure NE (coordination -> must enumerate all).
    distract_rows/cols  append strictly-dominated distractor strategies. They are
                never played in any NE (so the true equilibria are unchanged, just
                embedded with zeros) but force iterated elimination first.

  shared: (m, n) size, integer vs 2-decimal payoffs, payoff range [low, high],
          game_type in {general, zero_sum, symmetric}.

Reject sampling is used to hit the structural constraints; every constructor
returns a ``meta`` dict recording which constraints were actually met (so an
unsatisfiable request degrades gracefully instead of hanging).
"""
from __future__ import annotations

import numpy as np

import solvers as S


# ═══════════════════════════════════════════════════════════════════
#  primitive payoff / strategy samplers
# ═══════════════════════════════════════════════════════════════════
def _draw(shape, low, high, integer, rng):
    """Sample a payoff block in ``[low, high]`` — integer, or rounded to 2 decimals."""
    if integer:
        return rng.integers(low, high + 1, size=shape).astype(float)
    return np.round(rng.uniform(low, high, size=shape), 2)


def sample_payoffs(m, n, game_type, low, high, integer, rng):
    def draw(shape):
        return _draw(shape, low, high, integer, rng)

    if game_type == "zero_sum":
        R = draw((m, n)); C = -R
    elif game_type == "symmetric":
        assert m == n, "symmetric games require m == n"
        R = draw((m, n)); C = R.T.copy()
    else:
        R, C = draw((m, n)), draw((m, n))
    return R, C


def sample_opponent_mixed(n, alpha, rng):
    """Dirichlet(alpha) mixed strategy. alpha small -> peaked, large -> ~uniform."""
    x = rng.dirichlet(np.full(n, float(alpha)))
    return np.round(x, 4)


def opponent_entropy(sigma) -> float:
    """Normalised Shannon entropy in [0, 1] (1 = uniform, 0 = a pure column)."""
    p = np.asarray(sigma, float)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(sigma)))


# ═══════════════════════════════════════════════════════════════════
#  strictly-dominated distractor strategies (do NOT change the NE set)
# ═══════════════════════════════════════════════════════════════════
def add_distractors(R, C, k_rows, k_cols, low, high, integer, rng):
    """Append ``k_rows`` strictly-dominated rows and ``k_cols`` dominated cols.

    Distractors are lowered relative to the *real* (core) strategies 0..m0-1 /
    0..n0-1 so that each appended row is single-step strictly dominated by **every**
    real row (across all columns, including distractor columns) and each appended
    column by every real column (across all rows). This preserves the NE set — a
    strictly-dominated strategy is never played in any equilibrium — while forcing
    iterated elimination first.
    """
    R, C = np.asarray(R, float), np.asarray(C, float)
    m0, n0 = R.shape                                          # real (core) dims

    def _delta():
        return int(rng.integers(1, 4)) if integer else float(np.round(rng.uniform(0.5, 3.0), 2))

    # distractor columns: strictly below every REAL column for the col player, per row
    for _ in range(k_cols):
        m, n = C.shape
        d = _delta()
        new_C = C[:, :n0].min(axis=1) - d
        new_R = _draw(m, low, high, integer, rng)
        C = np.hstack([C, new_C.reshape(-1, 1)])
        R = np.hstack([R, new_R.reshape(-1, 1)])

    # distractor rows: strictly below every REAL row for the row player, in every column
    dcols = list(range(n0, R.shape[1]))                       # distractor column indices
    for _ in range(k_rows):
        m, n = R.shape
        d = _delta()
        new_R = R[:m0, :].min(axis=0) - d                     # below real-row min, all columns
        new_C = _draw(n, low, high, integer, rng)
        if dcols:                                             # keep distractor cols dominated here too
            real_cols = [j for j in range(n) if j not in dcols]
            base = new_C[real_cols].min() if real_cols else 0.0
            for dc in dcols:
                new_C[dc] = base - d
        R = np.vstack([R, new_R]); C = np.vstack([C, new_C])
    return R, C


# ═══════════════════════════════════════════════════════════════════
#  BEST-RESPONSE game constructor
# ═══════════════════════════════════════════════════════════════════
def build_br_game(m, n, game_type, knobs, rng, max_attempts=6000):
    opp_alpha  = knobs.get("opp_alpha", 1.0)
    gap_max    = knobs.get("gap_max", None)
    gap_min    = knobs.get("gap_min", None)
    adversarial = knobs.get("adversarial", False)
    force_ties = knobs.get("force_ties", False)
    low, high  = knobs.get("low", -5), knobs.get("high", 5)
    integer    = knobs.get("integer", True)

    best = None
    for attempt in range(max_attempts):
        R, C = sample_payoffs(m, n, game_type, low, high, integer, rng)
        sigma = sample_opponent_mixed(n, opp_alpha, rng)

        if force_ties and not integer:
            R = _inject_tie(R, sigma, rng)

        br = S.compute_best_response(R, sigma)
        g = S.eu_gap(br["expected_payoffs"])
        sc = S.br_shortcut_actions(R, sigma)
        br_set = set(br["best_response_actions"])

        met_gap = ((gap_max is None or g <= gap_max + 1e-6) and
                   (gap_min is None or g >= gap_min - 1e-6))
        met_adv = (not adversarial or
                   (sc["mode_column"] not in br_set and sc["global_max"] not in br_set))
        met_tie = (not force_ties or not br["is_unique"])

        cand = (R, C, sigma, br, g, sc, {"gap": met_gap, "adversarial": met_adv, "ties": met_tie})
        # keep the first candidate as a fallback; prefer one satisfying more constraints
        if best is None or _n_met(cand[6]) > _n_met(best[6]):
            best = cand
        if met_gap and met_adv and met_tie:
            break

    R, C, sigma, br, g, sc, met = best
    meta = {
        "opp_alpha": opp_alpha,
        "opp_entropy": round(opponent_entropy(sigma), 4),
        "eu_gap": round(g, 6),
        "shortcut_mode_column_correct": sc["mode_column"] in set(br["best_response_actions"]),
        "shortcut_global_max_correct": sc["global_max"] in set(br["best_response_actions"]),
        "constraints_met": met,
        "attempts": attempt + 1,
    }
    return R.tolist(), C.tolist(), sigma.tolist(), br, meta


def _inject_tie(R, sigma, rng):
    """Make two rows share the top expected payoff, as an *exact* tie.

    Row ``b`` is set to (best row ``a``) + δ where δ is supported on two columns
    with δ·σ = 0, so EU(b) == EU(a) at machine precision (no rounding that would
    break the tie). The perturbed row is left at full precision on purpose.
    """
    R = np.asarray(R, float)
    m, n = R.shape
    if m < 2 or n < 2:
        return R
    sigma = np.asarray(sigma, float)
    eu = R @ sigma
    a = int(np.argmax(eu))
    b = int(rng.choice([i for i in range(m) if i != a]))
    j1, j2 = rng.choice(n, size=2, replace=False)
    scale = float(rng.uniform(0.5, 2.0))
    delta = np.zeros(n)
    delta[j1] = sigma[j2] * scale                            # δ·σ = σ[j1]σ[j2] − σ[j2]σ[j1] = 0
    delta[j2] = -sigma[j1] * scale
    R = R.copy()
    R[b] = R[a] + delta
    return R


def _n_met(met: dict) -> int:
    return sum(1 for v in met.values() if v)


# ═══════════════════════════════════════════════════════════════════
#  NASH game constructor
# ═══════════════════════════════════════════════════════════════════
def build_nash_game(m, n, game_type, knobs, rng, max_attempts=6000):
    require_mixed      = knobs.get("require_mixed", False)
    require_multi_pure = knobs.get("require_multi_pure", False)
    min_equilibria     = knobs.get("min_equilibria", 1)
    distract_rows      = knobs.get("distract_rows", 0)
    distract_cols      = knobs.get("distract_cols", 0)
    low, high          = knobs.get("low", -5), knobs.get("high", 5)
    integer            = knobs.get("integer", True)

    best = None
    for attempt in range(max_attempts):
        R, C = sample_payoffs(m, n, game_type, low, high, integer, rng)
        pure = S.find_pure_nash(R, C)
        eqs, degenerate = S.find_all_nash_checked(R, C)
        if not eqs or degenerate:                    # reject: unsafe as exact ground truth
            continue
        met_mixed = (not require_mixed or (len(pure) == 0 and len(eqs) > 0))
        met_multi = (not require_multi_pure or len(pure) >= 2)
        met_count = (len(eqs) >= min_equilibria)

        cand = (R, C, pure, eqs, {"mixed": met_mixed, "multi_pure": met_multi, "count": met_count})
        if best is None or _n_met(cand[4]) > _n_met(best[4]):
            best = cand
        if met_mixed and met_multi and met_count:
            break

    if best is None:
        return [], [], [], [], {"constraints_met": {}, "attempts": max_attempts}
    R, C, pure, eqs, met = best

    # append distractors AFTER solving the core game, then re-solve the padded game
    core_dims = [int(R.shape[0]), int(R.shape[1])]
    if distract_rows or distract_cols:
        R, C = add_distractors(R, C, distract_rows, distract_cols, low, high, integer, rng)
        pure = S.find_pure_nash(R, C)
        eqs, degenerate = S.find_all_nash_checked(R, C)
        if not eqs or degenerate:                    # distractors made it degenerate -> skip
            return [], [], [], [], {"constraints_met": met, "attempts": attempt + 1}

    dom = S.iesds(R, C)
    meta = {
        "core_dims": core_dims,
        "distract_rows": distract_rows,
        "distract_cols": distract_cols,
        "n_equilibria": len(eqs),
        "pure_ne_count": len(pure),
        "mixed_ne_count": len(eqs) - len(pure),
        "n_dominated_strategies": S.count_dominated(R, C),
        "iesds_eliminated": dom["n_eliminated"],
        "constraints_met": met,
        "attempts": attempt + 1,
    }
    return R.tolist(), C.tolist(), pure, eqs, meta
