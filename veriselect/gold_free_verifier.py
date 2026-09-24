"""
Gold-FREE game-theory process verifier.

Same claim extraction as `gt_prm_verifier`, but every claim is checked using ONLY the
problem inputs the model was given — the payoff matrices (and, for the BR task, the
opponent's mixed strategy) — never `ground_truth`. So the resulting score is
deployment-realistic: no answer key.

Where `gt_prm_verifier` support-matches mixed-NE claims against the *solved* equilibria
and checks BR against `ground_truth`, this module instead:
  - EQUILIBRIUM (pure) : mutual best response on A, B         (already answer-key-free)
  - EQUILIBRIUM (mixed): self-contained ε-Nash indifference on A, B  ← the only new check
  - BR_ACTION          : argmax of A @ opponent_sigma          (opponent_sigma is given)
  - EXP_PAYOFF         : (A @ opponent_sigma)[i]               (opponent_sigma is given)
  - DOMINANCE          : payoff-matrix comparison              (already answer-key-free)

Score: gold_free_score = distinct_correct_w / (distinct_correct_w + incorrect_w + 1).
Distinct correct facts (deduped by support/action) reward substance without letting a
trace inflate by restating one true claim; incorrect claims pull it down; +1 smoothing
keeps an empty trace at 0. No coverage/recall term — that needs the equilibrium *count*,
which is not derivable from the matrix without a full solver (mixed games), so we don't
fake one.
"""
import numpy as np

from gt_prm_verifier import (
    extract_dominance_claims, extract_ne_claims, extract_mixed_ne_claims,
    extract_br_action_claims, extract_exp_payoff_claims,
    _label_idx, _row_weakly_dominates, _col_weakly_dominates, _is_pure_ne, _support_of,
    TYPE_WEIGHTS, BINARY_WEIGHTS, _SUPP_THR,
)

# indifference slack as a fraction of each player's payoff spread. Rounded probabilities
# (a model answering to 1–2 d.p.) shift the indifference point by up to a few % of the
# spread, so a spread-relative tolerance accepts a correctly-rounded mixed NE.
_NE_TOL_FRAC = 0.06
_EU_TOL = 0.10   # relative tolerance for EXP_PAYOFF (matches gt_prm_verifier)


def _verify_ne_selfcontained(A, B, sr, sc, tol_frac=_NE_TOL_FRAC):
    """True iff profile (sr, sc) is an (approximate) Nash equilibrium of (A, B), checked
    purely from the matrices: every action a player randomises over must earn within a
    small slack of that player's best attainable expected payoff (ε-Nash indifference).

    ponytail: spread-relative slack `tol_frac` is a heuristic ceiling — a coarsely-rounded
    correct profile and a close-but-wrong one have overlapping indifference gaps, so this
    cannot separate them as cleanly as support-matching a solved equilibrium (what the
    gold `process_score` does). Fine for a *selection* signal; tighten if precision matters.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    sr = np.asarray(sr, float); sc = np.asarray(sc, float)
    m, n = A.shape
    if sr.shape != (m,) or sc.shape != (n,) or sr.sum() <= 0 or sc.sum() <= 0:
        return False
    sr = sr / sr.sum(); sc = sc / sc.sum()
    tol_r = tol_frac * (A.max() - A.min() + 1e-9)
    tol_c = tol_frac * (B.max() - B.min() + 1e-9)
    u = A @ sc          # row player's expected payoff per row action
    w = sr @ B          # col player's expected payoff per col action
    row_ok = all(u[i] >= u.max() - tol_r for i in np.where(sr > _SUPP_THR)[0])
    col_ok = all(w[j] >= w.max() - tol_c for j in np.where(sc > _SUPP_THR)[0])
    return bool(row_ok and col_ok)


def verify_trace_gold_free(response, game, weights=None):
    """Gold-free counterpart of `gt_prm_verifier.verify_trace`. Returns
    {gold_free_score, accuracy, correct_w, incorrect_w, n_correct, n_incorrect, by_type}."""
    weights = weights or BINARY_WEIGHTS
    A = game["payoff_matrix_row"]
    B = game["payoff_matrix_col"]
    row_labels = game["row_labels"]
    col_labels = game["col_labels"]
    task = game.get("task", "nash_equilibrium")

    by_type = {k: [0, 0] for k in TYPE_WEIGHTS}
    seen_ok, seen_bad = set(), set()   # distinct facts, so repetition can't inflate either side
    correct_w = incorrect_w = 0.0

    def record(t, ok, key):
        by_type[t][0 if ok else 1] += 1
        bucket = seen_ok if ok else seen_bad
        if (t, key) in bucket:
            return 0.0, 0.0            # already counted this fact
        bucket.add((t, key))
        return (weights[t], 0.0) if ok else (0.0, weights[t])

    # DOMINANCE — payoff-matrix comparison (no answer key)
    for dom, dnt in extract_dominance_claims(response, row_labels, col_labels):
        if dom in row_labels and dnt in row_labels:
            i, k = _label_idx(dom, row_labels), _label_idx(dnt, row_labels)
            ok = i is not None and k is not None and i != k and _row_weakly_dominates(A, i, k)
        elif dom in col_labels and dnt in col_labels:
            j, l = _label_idx(dom, col_labels), _label_idx(dnt, col_labels)
            ok = j is not None and l is not None and j != l and _col_weakly_dominates(B, j, l)
        else:
            continue
        c, w = record("DOMINANCE", ok, (dom, dnt))
        correct_w += c; incorrect_w += w

    # EQUILIBRIUM — pure: mutual BR on A,B ; mixed: self-contained ε-Nash on A,B
    if task == "nash_equilibrium":
        for (rlab, clab) in extract_ne_claims(response, row_labels, col_labels):
            i, j = _label_idx(rlab, row_labels), _label_idx(clab, col_labels)
            ok = i is not None and j is not None and _is_pure_ne(A, B, i, j)
            c, w = record("EQUILIBRIUM", ok, (frozenset([i]), frozenset([j])))
            correct_w += c; incorrect_w += w
        for (sr, sc) in extract_mixed_ne_claims(response, row_labels, col_labels):
            ok = _verify_ne_selfcontained(A, B, sr, sc)
            c, w = record("EQUILIBRIUM", ok, (_support_of(sr), _support_of(sc)))
            correct_w += c; incorrect_w += w

    # BEST RESPONSE — computed from the GIVEN opponent strategy
    if task == "best_response":
        sigma = game.get("opponent_sigma")
        if sigma is not None:
            eu = np.asarray(A, float) @ np.asarray(sigma, float)
            best = set(np.where(eu >= eu.max() - 1e-9)[0].tolist())
            for lab in extract_br_action_claims(response, row_labels):
                idx = _label_idx(lab, row_labels)
                ok = idx is not None and idx in best
                c, w = record("BR_ACTION", ok, idx)
                correct_w += c; incorrect_w += w
            for lab, v in extract_exp_payoff_claims(response, row_labels):
                idx = _label_idx(lab, row_labels)
                if idx is None or idx >= len(eu):
                    continue
                ok = abs(v - eu[idx]) < _EU_TOL * (abs(eu[idx]) + 1)
                c, w = record("EXP_PAYOFF", ok, idx)
                correct_w += c; incorrect_w += w

    n_correct = sum(v[0] for v in by_type.values())
    n_incorrect = sum(v[1] for v in by_type.values())
    accuracy = correct_w / (correct_w + incorrect_w) if (correct_w + incorrect_w) > 0 else 0.0
    gold_free_score = correct_w / (correct_w + incorrect_w + 1.0)

    return {
        "gold_free_score": float(gold_free_score),
        "accuracy": float(accuracy),
        "correct_w": float(correct_w), "incorrect_w": float(incorrect_w),
        "n_correct": n_correct, "n_incorrect": n_incorrect,
        "by_type": by_type,
    }


def _selfcheck():
    # Mixed-only game: true NE row [0.875,0.125], col [0.5,0.5] (no pure NE). Verified
    # WITHOUT ground_truth — the profile satisfies indifference on A, B alone.
    mix = {
        "payoff_matrix_row": [[5.0, 1.0], [2.0, 4.0]],
        "payoff_matrix_col": [[-2.0, -1.0], [4.0, -3.0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"], "task": "nash_equilibrium",
        "ground_truth": {},   # deliberately empty: gold-free must not touch it
    }
    gold  = "ANSWER:\nMixed NE: [([0.88,0.12],[0.50,0.50])]"      # right support, 2-dp
    dp1   = "ANSWER:\nMixed NE: [([0.9,0.1],[0.5,0.5])]"          # coarse 1-dp, still right
    wrong = "ANSWER:\nMixed NE: [([1,0],[0,1])]"                  # (A,Z): row would deviate
    js    = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.88,0.12],[0.5,0.5]]]}\n```'  # JSON, same claim
    assert verify_trace_gold_free(gold, mix)["gold_free_score"] > 0.4, verify_trace_gold_free(gold, mix)
    assert verify_trace_gold_free(dp1, mix)["accuracy"] == 1.0, verify_trace_gold_free(dp1, mix)
    assert verify_trace_gold_free(wrong, mix)["accuracy"] == 0.0, verify_trace_gold_free(wrong, mix)
    assert verify_trace_gold_free(js, mix)["accuracy"] == 1.0, verify_trace_gold_free(js, mix)

    # BR: checked against the GIVEN opponent_sigma, no ground_truth.
    br = {
        "payoff_matrix_row": [[-5.0, 3.0], [2.0, -1.0]], "payoff_matrix_col": [[0, 0], [0, 0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"], "task": "best_response",
        "opponent_sigma": [0.0116, 0.9884], "ground_truth": {},
    }
    good = "EU(A) = 2.9072\nEU(B) = -0.9652\nBest response action(s): [A]"
    bad  = "Best response action(s): [B]"
    js_br = ('ANSWER:\n```json\n{"best_response_actions": ["A"], '
             '"expected_payoffs": [2.91, -0.97], "best_response_value": 2.91}\n```')
    assert verify_trace_gold_free(good, br)["accuracy"] == 1.0, verify_trace_gold_free(good, br)
    assert verify_trace_gold_free(bad, br)["accuracy"] == 0.0, verify_trace_gold_free(bad, br)
    assert verify_trace_gold_free(js_br, br)["accuracy"] == 1.0, verify_trace_gold_free(js_br, br)

    # Repetition can't inflate: restating one true NE thrice == stating it once.
    once  = "Mixed NE: [([0.9,0.1],[0.5,0.5])]"
    three = "Mixed NE: [([0.9,0.1],[0.5,0.5])]\nMixed NE: [([0.9,0.1],[0.5,0.5])]\nMixed NE: [([0.9,0.1],[0.5,0.5])]"
    assert verify_trace_gold_free(once, mix)["gold_free_score"] == verify_trace_gold_free(three, mix)["gold_free_score"]
    print("gold_free_verifier selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
