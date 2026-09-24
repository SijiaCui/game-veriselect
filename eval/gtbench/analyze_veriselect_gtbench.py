"""
Offline analysis of the gold-free PROCESS verifier on GTBench n=8 CoT data.

For each move-state with N candidate (reasoning, move) pairs, score every candidate's
reasoning with `verify_trace_gtbench`, then compare selection rules on move QUALITY:

  pass@1   = mean quality over the N candidates          (== random pick, the baseline)
  maj@N    = quality of the majority-voted move token     (self-consistency; ties -> mean)
  vsel@N   = quality of the argmax-verifier-score cand   (the method; ties -> mean)
  oracle@N = max quality over the N candidates           (ceiling any selector could reach)

Move QUALITY is the expected return against the actual (uniform-random) opponent, computed
gold (this is the EVALUATION oracle — separate from the gold-free verifier):
  nim (misère), tictactoe : exact expectimax vs random   -> win-prob in [0,1]
  connect4                : 1 if the move is non-blunder (take win / forced block / no self-loss)
  prisoners_dilemma       : Testify=1, Silent=0 (dominant)
  kuhn_poker              : expectimax vs uniform-random opponent, normalized to [0,1]
  first_sealed_auction    : EV(bid)/EV(best legal bid) vs the random opponent, in [0,1]
  liars_dice              : no cheap gold oracle -> COVERAGE-ONLY (report verifier stats)

Also reports verifier COVERAGE: %candidates with >=1 verifiable claim, score separation
(mean verifier score of good vs bad moves), and per-type claim correct/incorrect counts.
This is the iterate-offline signal: no GPU re-gen needed to retune the verifier.

Usage: python3 eval/gtbench/analyze_veriselect_gtbench.py [results_dir]
"""
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "veriselect"))
from gtbench_gold_free_verifier import (
    verify_trace_gtbench, _ttt_cells, _ttt_place, _ttt_wins, _me_symbol,
    _c4_board, _c4_drop_wins, _nim_mover_wins,
)

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                           "gtbench_veriselect", "q25_7b-cot-n8")
_RNG = random.Random(0)

# env_name -> action regex (from regex_and_format.py); last hit = the played move
MOVE_RE = {
    "tictactoe": r"C[1-3]R[1-3]",
    "connect4": r"C[1-7]",
    "nim": r"pile:\s*\d+,?\s*take:\s*\d+",
    "python_iterated_prisoners_dilemma": r"Testify|Silent",
    "kuhn_poker": r"Pass|Bet",
    "first_sealed_auction": r"<(\d+)>",
    "liars_dice": r"\d\s*dice",
}


def move_of(env, text):
    """The played move token from one candidate generation (last regex hit)."""
    if not text:
        return None
    hits = re.findall(MOVE_RE.get(env, r"$^"), text)
    return hits[-1] if hits else None


# ───────────────────────── evaluation oracles (vs random opponent) ─────────────────────────
@lru_cache(maxsize=None)
def _nim_win_random(piles, my_turn):
    """P(I win) in misère nim: I play optimal, opponent uniform-random. Empty board => the
    mover-to-act wins (opponent took the last match)."""
    piles = tuple(sorted(p for p in piles if p > 0))
    if not piles:
        return 1.0 if my_turn else 0.0
    moves = [(i, t) for i, p in enumerate(piles) for t in range(1, p + 1)]
    vals = []
    for i, t in moves:
        nxt = list(piles); nxt[i] -= t
        vals.append(_nim_win_random(tuple(sorted(x for x in nxt if x > 0)), not my_turn))
    return max(vals) if my_turn else mean(vals)


def _q_nim(obs, mv):
    piles = [int(x) for x in obs.get("piles", [])]
    g = re.search(r"pile:\s*(\d+),?\s*take:\s*(\d+)", mv or "")
    if not g:
        return None
    p, t = int(g.group(1)), int(g.group(2))
    if not (1 <= p <= len(piles) and 1 <= t <= piles[p - 1]):
        return None
    piles[p - 1] -= t
    return _nim_win_random(tuple(sorted(x for x in piles if x > 0)), False)   # opp's turn next


@lru_cache(maxsize=None)
def _ttt_val_random(cells, to_move, me):
    """Expected return for `me` (win=1, draw=.5, loss=0): me optimal, opp uniform-random."""
    opp = "O" if me == "X" else "X"
    if _ttt_wins(cells, me):
        return 1.0
    if _ttt_wins(cells, opp):
        return 0.0
    empties = [i for i in range(9) if cells[i] == "."]
    if not empties:
        return 0.5
    if to_move == me:
        return max(_ttt_val_random(_ttt_place(cells, i, me), opp, me) for i in empties)
    return mean(_ttt_val_random(_ttt_place(cells, i, opp), me, me) for i in empties)


def _q_ttt(obs, mv):
    cells, me = _ttt_cells(obs.get("self_moves", []), obs.get("opponent_moves", []))
    g = re.search(r"C(\d)R(\d)", mv or "")
    if not g:
        return None
    i = (int(g.group(1)) - 1) + (int(g.group(2)) - 1) * 3
    if not (0 <= i < 9) or cells[i] != ".":
        return None
    opp = "O" if me == "X" else "X"
    return _ttt_val_random(_ttt_place(cells, i, me), opp, me)


def _q_c4(obs, mv):
    """Non-blunder in {0,1}: take an available win; else block a forced loss; else any move
    that doesn't hand the opponent an immediate win."""
    heights, grid, me = _c4_board(obs.get("self_moves", []), obs.get("opponent_moves", []))
    opp = "O" if me == "X" else "X"
    g = re.search(r"C(\d)", mv or "")
    if not g:
        return None
    col = int(g.group(1)) - 1
    if not (0 <= col < 7) or heights[col] >= 6:
        return None
    wins = [c for c in range(7) if heights[c] < 6 and _c4_drop_wins(heights, grid, c, me)]
    if wins:
        return 1.0 if col in wins else 0.0
    threats = [c for c in range(7) if heights[c] < 6 and _c4_drop_wins(heights, grid, c, opp)]
    if threats:
        return 1.0 if col in threats else 0.0            # must block (single-threat case)
    # else: avoid giving opponent an immediate win on top of my move
    h2 = heights[:]; g2 = [r[:] for r in grid]
    g2[h2[col]][col] = me; h2[col] += 1
    gives = any(h2[c] < 6 and _c4_drop_wins(h2, g2, c, opp) for c in range(7))
    return 0.0 if gives else 1.0


def _q_pd(obs, mv):
    if not mv:
        return None
    return 1.0 if "testify" in mv.lower() else 0.0


# --- auction: closed-form EV vs a uniform-random opponent (valuation U{1..M}, bid U{0..v-1}) ---
def _auc_opp_pmf(M=10):
    pmf = [0.0] * M                       # P(opp bid = k), k in 0..M-1
    for k in range(M):
        pmf[k] = sum(1.0 / vo for vo in range(k + 1, M + 1)) / M
    return pmf


_AUC_PMF = _auc_opp_pmf(10)


def _auc_ev(v, b):
    pwin = sum(_AUC_PMF[k] for k in range(0, min(b, len(_AUC_PMF)))) \
        + (0.5 * _AUC_PMF[b] if b < len(_AUC_PMF) else 0.0)   # tie split
    return (v - b) * pwin


def _q_auction(obs, mv):
    """Quality = EV(bid)/EV(best legal bid) vs the random opponent, in [0,1]."""
    v = int(float(obs.get("valuation", 0)))
    g = re.search(r"(\d+)", mv or "")
    if not g:
        return None
    b = int(g.group(1))
    legal = [int(re.search(r"\d+", x).group()) for x in obs.get("legal_moves", []) if re.search(r"\d+", x)]
    if not legal or b not in legal:
        return None
    evs = [_auc_ev(v, x) for x in legal]
    mx = max(evs)
    return (_auc_ev(v, b) / mx) if mx > 0 else 0.5


ORACLE = {"nim": _q_nim, "tictactoe": _q_ttt, "connect4": _q_c4,
          "python_iterated_prisoners_dilemma": _q_pd, "first_sealed_auction": _q_auction}


# --- kuhn poker: expectimax vs uniform-random opponent, enumerated over the 2 hidden cards ---
import pyspiel  # noqa: E402


def _expectimax(state, me, memo):
    """Expected return for `me`: me plays optimal, opponent uniform-random, chance uniform."""
    if state.is_terminal():
        return state.returns()[me]
    key = (str(state), me)
    if key in memo:
        return memo[key]
    if state.is_chance_node():
        v = sum(p * _expectimax(_child(state, a), me, memo) for a, p in state.chance_outcomes())
    else:
        vals = [_expectimax(_child(state, a), me, memo) for a in state.legal_actions()]
        v = max(vals) if state.current_player() == me else sum(vals) / len(vals)
    memo[key] = v
    return v


def _child(state, a):
    s = state.clone(); s.apply_action(a); return s


def _kuhn_state(my_card, opp_card, me, moves):
    g = pyspiel.load_game("kuhn_poker"); s = g.new_initial_state()
    deals = [my_card, opp_card] if me == 0 else [opp_card, my_card]
    s.apply_action(deals[0]); s.apply_action(deals[1])
    seq = moves if isinstance(moves, (list, tuple)) else (list(moves) if moves else [])
    for m in seq:
        s.apply_action(1 if str(m).lower().startswith("b") else 0)
    return s


def _q_kuhn(obs, mv):
    try:
        my = int(obs["card"])
    except (TypeError, ValueError, KeyError):
        return None
    me = int(obs.get("player_idx", 0))
    a = 1 if "bet" in (mv or "").lower() else (0 if "pass" in (mv or "").lower() else None)
    if a is None:
        return None
    opps = [c for c in (0, 1, 2) if c != my]
    memo = {}

    def val(action):
        tot = 0.0
        for oc in opps:
            s = _kuhn_state(my, oc, me, obs.get("moves"))
            if s.current_player() != me or action not in s.legal_actions():
                return None
            tot += _expectimax(_child(s, action), me, memo)
        return tot / len(opps)

    vals = {x: val(x) for x in (0, 1)}
    if vals[a] is None or any(v is None for v in vals.values()):
        return None
    mn, mx = min(vals.values()), max(vals.values())
    return 0.5 if mx == mn else (vals[a] - mn) / (mx - mn)


ORACLE["kuhn_poker"] = _q_kuhn


# ───────────────────────── analysis ─────────────────────────
def _candidates(step):
    """(reasoning, move) list for a CoTAgent step: raw_reasoning[i] (fallback llm_output[i])."""
    q = step["queries"][0] if step.get("queries") else None
    if not q:
        return []
    outs = q.get("llm_output") or []
    raws = q.get("raw_reasoning") or outs
    env = step["observation"].get("env_name", "")
    return [(raws[i] if i < len(raws) else outs[i], move_of(env, outs[i])) for i in range(len(outs))]


def analyze_game(files):
    rows = []   # per state: dict(scores=[...], quals=[...], env)
    cov_any = cov_pos = n_cand = 0
    by_type = {}
    good_scores, bad_scores = [], []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec.get("matches", []):
                for st in m.get("steps", []):
                    if st.get("agent") != "CoTAgent":
                        continue
                    obs = st["observation"]; env = obs.get("env_name", "")
                    cands = _candidates(st)
                    if not cands:
                        continue
                    scores, quals, moves = [], [], []
                    for reasoning, mv in cands:
                        v = verify_trace_gtbench(reasoning, obs)
                        n_cand += 1
                        if v["n_correct"] + v["n_incorrect"] > 0:
                            cov_any += 1
                        if v["gold_free_score"] > 0:
                            cov_pos += 1
                        for t, (ok, bad) in v["by_type"].items():
                            if ok or bad:
                                a = by_type.setdefault(t, [0, 0]); a[0] += ok; a[1] += bad
                        scores.append(v["gold_free_score"])
                        q = ORACLE[env](obs, mv) if env in ORACLE else None
                        quals.append(q)
                        moves.append(mv)
                        if q is not None:
                            (good_scores if q >= 0.5 else bad_scores).append(v["gold_free_score"])
                    rows.append({"scores": scores, "quals": quals, "moves": moves, "env": env,
                                 "judge": (st["queries"][0].get("judge_rewards")
                                           if st.get("queries") else None)})
    return rows, cov_any, cov_pos, n_cand, by_type, good_scores, bad_scores


def _argmax_set(scores):
    mx = max(scores)
    return [i for i, s in enumerate(scores) if s >= mx - 1e-12]


def _vote_key(mv):
    """Canonical vote token: collapse whitespace and drop commas so surface-form variants of
    the SAME logical move (e.g. nim 'pile:1, take:1' vs 'pile:1,take:1') count as one vote.
    None stays None (never enters the vote — filtered upstream by q is not None)."""
    if mv is None:
        return None
    return re.sub(r"\s+", "", mv).replace(",", "").lower()


def _majority_quality(moves, quals):
    """Self-consistency: vote over played-move tokens (canonicalized via _vote_key), return the
    modal move's quality. Ties (tokens sharing the top vote count) -> mean quality over the tied
    tokens, which equals the expected quality under a uniform random tie-break: tied tokens have
    EQUAL counts (they are the argmax of the Counter) so candidate-mean == token-mean, and quality
    is a pure function of the move so all candidates of one token share it. Voted over the SAME
    evaluable-candidate pool used by pass@1/vsel/oracle, so all selectors choose from an identical
    set of moves."""
    keys = [_vote_key(m) for m in moves]
    cnt = Counter(keys)
    top = max(cnt.values())
    modal = {tok for tok, c in cnt.items() if c == top}
    return mean(q for k, q in zip(keys, quals) if k in modal)


def selection_metrics(rows):
    """pass@1 / maj@N / vsel@N / oracle@N (and judge@N when judge_rewards are present) over states
    that HAVE an oracle quality. vsel/judge = EXPECTED quality over the argmax-score set
    (deterministic — no tie-break RNG), which equals pass@1 when the signal can't discriminate
    (all cands tie); maj@N votes over the played-move token (see _majority_quality)."""
    p1, mj, vs, orc, jd, n, njud = [], [], [], [], [], 0, 0
    for r in rows:
        pairs = [(i, q) for i, q in enumerate(r["quals"]) if q is not None]
        if not pairs:
            continue
        n += 1
        idxs = [i for i, _ in pairs]
        quals = [q for _, q in pairs]
        scores = [r["scores"][i] for i in idxs]
        moves = [r["moves"][i] for i in idxs]
        p1.append(mean(quals))
        mj.append(_majority_quality(moves, quals))
        vs.append(mean(quals[k] for k in _argmax_set(scores)))
        orc.append(max(quals))
        jr = r.get("judge")
        if jr and len(jr) == len(r["scores"]):
            jvals = [jr[i] for i in idxs]
            jd.append(mean(quals[k] for k in _argmax_set(jvals)))
            njud += 1
    if not n:
        return None
    out = {"n_states": n, "pass@1": mean(p1), "maj@N": mean(mj),
           "vsel@N": mean(vs), "oracle@N": mean(orc)}
    if njud:
        out["judge@N"] = mean(jd)
        out["n_judge"] = njud
    return out


def _selfcheck():
    # single modal token -> its own quality
    assert _majority_quality(["A", "A", "B"], [1.0, 1.0, 0.0]) == 1.0
    # tie A(q=1)×2 vs B(q=0)×2 -> mean of the two tied tokens = 0.5
    assert _majority_quality(["A", "A", "B", "B"], [1.0, 1.0, 0.0, 0.0]) == 0.5
    # single modal token B (count 2) beats A (count 1), regardless of qualities
    assert _majority_quality(["A", "B", "B"], [1.0, 0.0, 0.0]) == 0.0
    # DISTINCT tokens sharing a quality are NOT merged: A,A (count 2, q=1) is the sole mode,
    # even though B and C also each have q=0 -> result is A's quality, not a 3-way average
    assert _majority_quality(["A", "A", "B", "C"], [1.0, 1.0, 0.0, 0.0]) == 1.0
    # 3-way tie -> mean over all three tokens
    assert abs(_majority_quality(["A", "B", "C"], [0.9, 0.6, 0.0]) - 0.5) < 1e-9
    # canonicalization: surface-form variants of one logical move vote together (count 3 -> modal)
    assert _vote_key("pile:1, take:1") == _vote_key("pile:1,take:1") == _vote_key("PILE:1 TAKE:1")
    assert _majority_quality(["pile:1, take:1", "pile:1,take:1", "C2"], [1.0, 1.0, 0.0]) == 1.0
    print("analyze_veriselect_gtbench majority selfcheck OK")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _selfcheck()
        return
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    games = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"# veriselect gtbench analysis  ({root})\n")
    print(f"{'game':<22}{'states':>7}{'cand':>6}{'cov_any':>8}{'cov>0':>7}"
          f"{'pass@1':>8}{'maj@N':>8}{'vsel@N':>8}{'judge@N':>8}{'orac@N':>8}"
          f"{'Δmaj':>7}{'Δvs':>7}{'Δjdg':>7}{'sep':>7}")
    deltas = []
    mdeltas = []
    jdeltas = []
    for g in games:
        files = glob.glob(os.path.join(root, g, "*.jsonl"))
        if not files:
            continue
        rows, cov_any, cov_pos, n_cand, by_type, gs, bs = analyze_game(files)
        sel = selection_metrics(rows)
        sep = (mean(gs) - mean(bs)) if gs and bs else float("nan")
        has_j = bool(sel) and "judge@N" in sel
        p1 = f"{sel['pass@1']:.3f}" if sel else "  -"
        mj = f"{sel['maj@N']:.3f}" if sel else "  -"
        vs = f"{sel['vsel@N']:.3f}" if sel else "  -"
        jdg = f"{sel['judge@N']:.3f}" if has_j else "  -"
        orc = f"{sel['oracle@N']:.3f}" if sel else "  -"
        dmaj = f"{sel['maj@N'] - sel['pass@1']:+.3f}" if sel else "  -"
        dl = f"{sel['vsel@N'] - sel['pass@1']:+.3f}" if sel else "  -"
        djdg = f"{sel['judge@N'] - sel['pass@1']:+.3f}" if has_j else "  -"
        if sel:
            deltas.append(sel["vsel@N"] - sel["pass@1"])
            mdeltas.append(sel["maj@N"] - sel["pass@1"])
        if has_j:
            jdeltas.append(sel["judge@N"] - sel["pass@1"])
        ns = sel["n_states"] if sel else 0
        print(f"{g:<22}{ns:>7}{n_cand:>6}{cov_any/max(1,n_cand):>8.2f}{cov_pos/max(1,n_cand):>7.2f}"
              f"{p1:>8}{mj:>8}{vs:>8}{jdg:>8}{orc:>8}{dmaj:>7}{dl:>7}{djdg:>7}{sep:>7.2f}")
        if by_type:
            bt = " ".join(f"{t}:{c}/{c+b}" for t, (c, b) in sorted(by_type.items()) if c + b)
            print(f"{'  claims(correct/total):':<22}{bt}")
    if mdeltas:
        print(f"\n  mean Δ(maj-pass@1)   over oracle games: {mean(mdeltas):+.3f}")
    if deltas:
        print(f"  mean Δ(vsel-pass@1)  over oracle games: {mean(deltas):+.3f}")
    if jdeltas:
        print(f"  mean Δ(judge-pass@1) over judged oracle games: {mean(jdeltas):+.3f}")
    print("\n(maj@N/vsel@N/judge@N > pass@1 => that selector picks better moves than a random draw; "
          "oracle@N = ceiling. judge@N blank => run judge_reward_gtbench.py first. "
          "liars_dice has no cheap oracle -> selection blank, coverage only.)")


if __name__ == "__main__":
    main()
