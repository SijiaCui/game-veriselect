"""VeriSelect — the method: a sound game-theory *process verifier* used as a
test-time *best-of-N selector* (not as a training reward).

Draw N i.i.d. reasoning traces per prompt, score each with `gt_prm_verifier.verify_trace`
against the game's payoff matrices + solver ground truth, and return the argmax. The
verifier never sees the outcome / answer key in its `gold_free_score` variant, so this
is training-free and (for the Nash task) deployment-realistic.

Two selection methods, each its own verifier:
  - "process_score"   : gt_prm_verifier — 0.5*accuracy + 0.5*coverage, uses the solved
                        ground truth (equilibria set / BR key) for the recall term.
  - "gold_free_score" : gold_free_verifier — every claim checked from the problem inputs
                        alone (payoff matrices + given opponent_sigma), NO answer key.

This module is the engine only (scorer + argmax). Wiring it into the shared eval
harness as a `baselines/base.py` agent (populate a per-candidate verifier score, then
argmax like Oracle) is the eval-integration step, done separately.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_prm_verifier import verify_trace
from gold_free_verifier import verify_trace_gold_free

# a signal name -> the verifier that produces it (default: the gold ground-truth verifier)
_GOLD_FREE_SIGNALS = {"gold_free_score"}


def score_traces(traces, game, signal="process_score"):
    """Score each trace with the selected verifier. -> list[float] in [0,1].
    `signal="gold_free_score"` routes to the answer-key-free verifier; anything else
    (process_score, accuracy, coverage, ...) to the ground-truth verifier."""
    vt = verify_trace_gold_free if signal in _GOLD_FREE_SIGNALS else verify_trace
    return [vt(t, game)[signal] for t in traces]


def select(traces, game, rng, signal="process_score"):
    """Best-of-N: index of the trace with the max verifier score (random tie-break).
    Mirrors the project's `analyze.py` `veriselect` rule exactly."""
    sc = score_traces(traces, game, signal)
    mx = max(sc)
    return rng.choice([i for i, s in enumerate(sc) if s >= mx - 1e-12])


def _selfcheck():
    import random
    from gt_prm_verifier import verify_trace as vt

    # (1) MIXED-only nash game (the regime the new bench mostly emits). Correctness is
    #     SUPPORT-level (benchmark's primary `exact`): a claim is right iff it randomises
    #     over the same actions as a true equilibrium. Game: true NE row [0.875,0.125]
    #     col [0.5,0.5] (support {A,B}x{Y,Z}); no pure NE.
    mix = {
        "payoff_matrix_row": [[5.0, 1.0], [2.0, 4.0]],
        "payoff_matrix_col": [[-2.0, -1.0], [4.0, -3.0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"],
        "task": "nash_equilibrium",
        "ground_truth": {"pure_ne_count": 0, "n_equilibria": 1,
                         "equilibria": [{"sigma_row": [0.875, 0.125],
                                         "sigma_col": [0.5, 0.5], "is_pure": False}]},
    }
    gold_mix  = "No pure NE.\nANSWER:\nPure NE: none\nMixed NE: [([0.88,0.12],[0.50,0.50])]"  # 2-dp, right support
    gold_1dp  = "ANSWER:\nMixed NE: [([0.9,0.1],[0.5,0.5])]"   # coarse 1-dp rounding, still right support
    wrong_mix = "ANSWER:\nMixed NE: [([1,0],[0,1])]"           # WRONG support (claims pure (A,Z))
    assert vt(gold_mix, mix)["process_score"] > 0.8, vt(gold_mix, mix)
    assert vt(gold_1dp, mix)["process_score"] > 0.8, vt(gold_1dp, mix)     # rounding must not reject a correct answer
    assert vt(wrong_mix, mix)["process_score"] == 0.0, vt(wrong_mix, mix)  # nor credit a wrong-support one

    # (2) Single-char labels: a correct BUT verbose BR trace whose true answer is B must
    #     not be tanked by the English article 'a' being read as label 'A'.
    br = {
        "payoff_matrix_row": [[0.0, 0.0], [5.0, 5.0]],   # row B (idx 1) strictly best
        "payoff_matrix_col": [[0.0, 0.0], [0.0, 0.0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"],
        "task": "best_response",
        "ground_truth": {"best_response_actions": [1], "expected_payoffs": [0.0, 5.0]},
    }
    verbose = "A rational player maximizes payoff. The best response is a payoff-maximizing action. Best response: B."
    assert vt(verbose, br)["process_score"] > 0.5, vt(verbose, br)

    # (3) select() picks the verified trace, on BOTH signals. "gold_free_score" now routes
    #     to the answer-key-free verifier (checks the mixed NE from A,B, not gt.equilibria).
    rng = random.Random(0)
    assert select([wrong_mix, gold_mix], mix, rng) == 1
    assert select([wrong_mix, gold_mix], mix, rng, "gold_free_score") == 1
    assert score_traces([gold_mix], mix, "gold_free_score")[0] > 0.0, "gold-free must credit a matrix-valid NE"
    assert score_traces([wrong_mix], mix, "gold_free_score")[0] == 0.0, "gold-free must reject a wrong-support NE"

    # (4) COVERAGE is recall over distinct equilibria: a complete answer beats a partial one,
    #     and repeating one equilibrium can't inflate the score. Coordination game, 2 pure NE.
    coord = {
        "payoff_matrix_row": [[1.0, 0.0], [0.0, 1.0]], "payoff_matrix_col": [[1.0, 0.0], [0.0, 1.0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"], "task": "nash_equilibrium",
        "ground_truth": {"n_equilibria": 2, "equilibria": [
            {"sigma_row": [1, 0], "sigma_col": [1, 0], "is_pure": True},   # (A,Y)
            {"sigma_row": [0, 1], "sigma_col": [0, 1], "is_pure": True}]}, # (B,Z)
    }
    complete = "ANSWER:\nPure NE: [(A,Y),(B,Z)]"
    partial3 = "(A,Y) is a NE. (A,Y) is a NE. (A,Y) is a NE.\nANSWER:\nPure NE: [(A,Y)]"  # one eq, thrice
    assert vt(complete, coord)["process_score"] == 1.0, vt(complete, coord)
    assert vt(partial3, coord)["coverage"] == 0.5, vt(partial3, coord)          # not inflated by repetition
    assert vt(complete, coord)["process_score"] > vt(partial3, coord)["process_score"]
    print("veriselect selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
