"""
Gold-FREE process verifier for GTBench games (all 7: nim / tictactoe / connect4 /
prisoners_dilemma / first_sealed_auction / kuhn_poker / liars_dice).

The GTBench analog of `gold_free_verifier.py`: given one move's `observation` (the exact
state the model saw) it scores the model's natural-language REASONING by extracting
typed, formally-checkable claims and verifying each against facts computable from that
observation alone — never "who won".

GOLD-FREE INVARIANT (the one admissibility rule): a claim is credited/penalized only if
its truth is a decidable function of (the model's own observation, the fixed rules) —
WITHOUT the opponent's hidden type (card/die/valuation), WITHOUT any assumed opponent
strategy or equilibrium, and WITHOUT the outcome. Anything else is *not extracted* (the
refusal rule) — neither credited nor penalized. This is what keeps the selector sound on
imperfect-information games (kuhn / liars_dice / auction), where the strategically
interesting facts need the hidden state and so must be left alone.

Claim types (shared schema; a game uses the subset it can check gold-free):
  TACTICAL/POSITION (solved perfect-info): nim-sum & misère value, ttt minimax, c4 tactics
  DOMINANCE : one action's payoff beats another's in every contingency (PD, auction, kuhn)
  PAYOFF    : recompute a stated deterministic payoff (auction profit = valuation − bid)
  PROBABILITY: recompute a stated probability from KNOWN info (kuhn 3-card, liars own die)

Score: gold_free_score = correct_w / (correct_w + incorrect_w + 1); distinct facts deduped
so restating a true claim can't inflate; +1 smoothing keeps an empty/truncated trace at 0.
No coverage/recall term (needs a full solver enumeration we won't fake). Stdlib only.
"""
import re
from functools import lru_cache, reduce
from operator import xor

# harder / rarer claims weighted higher; binary uses all 1.0. ponytail: weights are a
# tunable selector-signal heuristic, not a calibrated reward — flatten to BINARY if unsure.
TYPE_WEIGHTS = {"WINNING_MOVE": 3.0, "TAKE_WIN": 3.0, "BLOCK": 3.0,
                "POSITION": 2.0, "FORK": 2.0, "OPTIMAL": 2.0, "DOMINANCE": 2.0,
                "NIM_SUM": 1.0, "THREAT": 1.5, "PAYOFF": 1.5, "PROBABILITY": 1.5}
BINARY_WEIGHTS = {k: 1.0 for k in TYPE_WEIGHTS}

# (game, claim) pairs that are DECIDABLE gold-free yet empirically NOT predictive of move
# quality against the actual (random) opponent — extracted so `by_type` still reports them
# (for the ablation table) but NOT scored, so they can't misguide the selector.
# ponytail: from the within-state ablation + online100 A/B (2026-09-16):
#   ("nim","NIM_SUM")      — the (normal) nim-sum is correct arithmetic but points at the
#                            normal-nim move, which is misère-/vs-random-suboptimal
#                            (between-state align -0.109; online nim -0.140 with it on).
#   ("kuhn_poker","DOMINANCE") — "I hold the best/worst card" is stated ~49% correct (a
#                            coin flip) and doesn't map to bet/pass quality vs random.
# The gold-free PROBABILITY/POSITION/WINNING_MOVE signals for these games are kept.
UNSCORED = {("nim", "NIM_SUM"), ("kuhn_poker", "DOMINANCE")}

# numeric tolerance for PROBABILITY / PAYOFF claims (matches bench/gamesolve/reward.py)
_EPS_REL, _EPS_ABS = 0.02, 0.01


def _num_ok(pred, gt):
    return abs(pred - gt) <= _EPS_REL * abs(gt) + _EPS_ABS


# ───────────────────────── nim primitives (misère) ─────────────────────────
@lru_cache(maxsize=None)
def _nim_mover_wins(piles):
    """True iff the player to move wins misère nim (take last match -> lose) with optimal
    play. Empty board => the opponent just took the last match => mover WINS."""
    piles = tuple(sorted(p for p in piles if p > 0))
    if not piles:
        return True
    for i, p in enumerate(piles):
        for take in range(1, p + 1):
            nxt = list(piles); nxt[i] = p - take
            if not _nim_mover_wins(tuple(nxt)):
                return True
    return False


def _nim_sum(piles):
    return reduce(xor, piles, 0)


# ───────────────────────── grid primitives (ttt / c4) ─────────────────────────
def _me_symbol(self_moves, opp_moves):
    """Whose mark is the model's, from move-list parity (player 0 = X moves first)."""
    return "X" if len(self_moves) == len(opp_moves) else "O"


def _ttt_cells(self_moves, opp_moves):
    """3x3 board as a 9-char string of X/O/. (cell = (C-1)+(R-1)*3). Cells are distinct,
    so occupancy needs no move interleaving — just paint own vs opponent marks."""
    me = _me_symbol(self_moves, opp_moves)
    opp = "O" if me == "X" else "X"
    cells = list("." * 9)
    for mv, sym in [(self_moves, me), (opp_moves, opp)]:
        for m in mv:
            g = re.search(r"C(\d)R(\d)", m)
            if g:
                idx = (int(g.group(1)) - 1) + (int(g.group(2)) - 1) * 3
                if 0 <= idx < 9:
                    cells[idx] = sym
    return "".join(cells), me


_TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
              (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _ttt_wins(cells, sym):
    return any(all(cells[i] == sym for i in ln) for ln in _TTT_LINES)


def _ttt_place(cells, idx, sym):
    return cells[:idx] + sym + cells[idx + 1:]


def _ttt_threats(cells, sym):
    """Empty cells where placing `sym` immediately completes a line."""
    return {i for i in range(9) if cells[i] == "." and _ttt_wins(_ttt_place(cells, i, sym), sym)}


@lru_cache(maxsize=None)
def _ttt_value(cells, me):
    """Negamax game value in {+1,0,-1} for `me` to move (no 3-in-a-row present yet)."""
    empties = [i for i in range(9) if cells[i] == "."]
    if not empties:
        return 0
    opp = "O" if me == "X" else "X"
    best = -2
    for i in empties:
        nc = _ttt_place(cells, i, me)
        v = 1 if _ttt_wins(nc, me) else -_ttt_value(nc, opp)
        if v > best:
            best = v
    return best


def _c4_board(self_moves, opp_moves):
    """(heights[7], grid) where grid[row][col] in {X,O,.}, row 0 = bottom. Colors depend on
    drop order, so replay the interleaved sequence (X = first mover)."""
    me = _me_symbol(self_moves, opp_moves)
    x_moves, o_moves = (self_moves, opp_moves) if me == "X" else (opp_moves, self_moves)
    seq = []
    for i in range(max(len(x_moves), len(o_moves))):
        if i < len(x_moves): seq.append((x_moves[i], "X"))
        if i < len(o_moves): seq.append((o_moves[i], "O"))
    grid = [["."] * 7 for _ in range(6)]
    heights = [0] * 7
    for m, sym in seq:
        g = re.search(r"C?(\d)", m)
        if not g:
            continue
        col = int(g.group(1)) - 1
        if 0 <= col < 7 and heights[col] < 6:
            grid[heights[col]][col] = sym
            heights[col] += 1
    return heights, grid, me


def _c4_drop_wins(heights, grid, col, sym):
    """True iff dropping `sym` in `col` completes 4-in-a-row (col must be non-full)."""
    if not (0 <= col < 7) or heights[col] >= 6:
        return False
    r, c = heights[col], col
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        run = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < 6 and 0 <= cc < 7 and grid[rr][cc] == sym:
                run += 1; rr += dr * s; cc += dc * s
        if run >= 4:
            return True
    return False


def _c4_threats_after(heights, grid, col, sym):
    """Count winning follow-ups for `sym` after it drops in `col` (its own threats)."""
    if heights[col] >= 6:
        return 0
    h2 = heights[:]; g2 = [row[:] for row in grid]
    g2[h2[col]][col] = sym; h2[col] += 1
    return sum(1 for c in range(7) if _c4_drop_wins(h2, g2, c, sym))


# ───────────────────────── extraction helpers ─────────────────────────
_WIN_RE = re.compile(r"\b(win|wins|winning|victor)", re.IGNORECASE)
# TAKE_WIN: a concrete "this move wins NOW" claim — excludes vague "winning position/strategy"
# ("winning" -ing is dropped; require "win/wins" or "complete a row/line/four").
_TAKEWIN_RE = re.compile(r"\bwins?\b|win the (?:game|round)|"
                         r"complete\w*\s+(?:the\s+)?(?:row|line|four|three|column|diagonal)",
                         re.IGNORECASE)
# a real block claim: the word "block", or "opponent ... win" (an explicit threat), not bare "opponent"
_BLOCK_RE = re.compile(r"\bblock\w*|opponent[^.\n]{0,20}\bwin", re.IGNORECASE)
_FORK_RE = re.compile(r"\b(fork|double threat|two threats)", re.IGNORECASE)
_BEST_RE = re.compile(r"\b(best|optimal|strongest|should play|draw)", re.IGNORECASE)
_OPP_RE = re.compile(r"(opponent|opp\b|enemy|they|you|user)", re.IGNORECASE)
_DOM_RE = re.compile(r"(dominat|dominant|strictly better|always better|best (?:strategy|choice|move)|"
                     r"optimal|never (?:fold|concede)|should always)", re.IGNORECASE)
# a probability literal: 0.5, .5, 50%, 1/6, or a bare 0/1 (but not a "1 dice"/"3 value" count)
_PROB_RE = re.compile(r"(\d+(?:\.\d+)?%|\d*\.\d+|\d+\s*/\s*\d+|\b[01]\b(?!\s*(?:dice|value)))")


def _parse_prob(tok):
    """Parse a probability literal -> float in [0,1] (percent / decimal / fraction)."""
    tok = tok.strip()
    try:
        if tok.endswith("%"):
            return float(tok[:-1]) / 100.0
        if "/" in tok:
            a, b = tok.split("/"); return float(a) / float(b)
        return float(tok)
    except (ValueError, ZeroDivisionError):
        return None


def _first_prob(seg):
    m = _PROB_RE.search(seg)
    return _parse_prob(m.group(0)) if m else None


def _moves_near(text, keyword_re, move_re, window=35, guard_re=None):
    """Move tokens (via `move_re`, group 0) found within `window` chars of a keyword hit,
    order-preserving & deduped. Both directions, so 'C2R2 wins' and 'win at C2R2' match.
    A keyword hit is skipped if `guard_re` matches the 18 chars just before it — used to
    drop opponent-attributed wins ('block X or the opponent wins') from a self TAKE_WIN."""
    out = []
    for k in keyword_re.finditer(text):
        if guard_re and guard_re.search(text[max(0, k.start() - 18): k.start()]):
            continue
        seg = text[max(0, k.start() - window): k.end() + window]
        for m in move_re.finditer(seg):
            out.append(m.group(0))
    return list(dict.fromkeys(out))


def _nim_moves_near(text, window=50):
    """(pile, take) pairs near a win keyword. Handles 'pile P, take T', 'pile:P take:T',
    and 'take/remove T from pile P'. Skips opponent-attributed wins."""
    out = []
    for k in _WIN_RE.finditer(text):
        if _OPP_RE.search(text[max(0, k.start() - 18): k.start()]):
            continue
        seg = text[max(0, k.start() - window): k.end() + window]
        for g in re.finditer(r"pile[:\s]*(\d+)[,\s]*take[:\s]*(\d+)", seg, re.IGNORECASE):
            out.append((int(g.group(1)), int(g.group(2))))
        for g in re.finditer(r"(?:tak|remov)\w*\s+(\d+)\s+(?:\w+\s+){0,3}?from\s+pile[:\s]*(\d+)",
                             seg, re.IGNORECASE):
            out.append((int(g.group(2)), int(g.group(1))))
    return list(dict.fromkeys(out))


def _idx_ttt(tok):
    g = re.search(r"C(\d)R(\d)", tok)
    if not g:
        return None
    i = (int(g.group(1)) - 1) + (int(g.group(2)) - 1) * 3
    return i if 0 <= i < 9 else None


def _col_c4(tok):
    g = re.search(r"(\d)", tok)
    if not g:
        return None
    c = int(g.group(1)) - 1
    return c if 0 <= c < 7 else None


# ───────────────────────── per-game checkers ─────────────────────────
# each: (text, observation, add) -> None ; add(type, ok, key) is the dedup closure.
def _check_nim(text, obs, add):
    piles = tuple(int(x) for x in obs.get("piles", []))
    # NIM_SUM: trigger on "nim-sum"/"grundy" (NOT bare "xor" — that sits between operands)
    # and take the RESULT: the number after the last '=', else after "is/equals".
    for m in re.finditer(r"nim[\s-]?sum|grundy", text, re.IGNORECASE):
        seg = re.split(r"[\n.]", text[m.end(): m.end() + 80], 1)[0]
        eqs = re.findall(r"=\s*(-?\d+)", seg)
        if eqs:
            v = int(eqs[-1])
        else:
            im = re.search(r"\b(?:is|equals?)\s+(?:to\s+)?(-?\d+)", seg, re.IGNORECASE)
            v = int(im.group(1)) if im else None
        if v is not None:
            add("NIM_SUM", v == _nim_sum(piles), v)
    if re.search(r"\b(i (can|will)?\s*win|winning position|n-position)\b", text, re.IGNORECASE):
        add("POSITION", _nim_mover_wins(piles), "win")
    if re.search(r"\b(i (will|'ll)?\s*los|losing position|p-position)\b", text, re.IGNORECASE):
        add("POSITION", not _nim_mover_wins(piles), "lose")
    for p, t in _nim_moves_near(text):
        ok = 1 <= p <= len(piles) and 1 <= t <= piles[p - 1]
        if ok:
            nxt = list(piles); nxt[p - 1] -= t
            ok = not _nim_mover_wins(tuple(nxt))
        add("WINNING_MOVE", ok, (p, t))


def _check_ttt(text, obs, add):
    cells, me = _ttt_cells(obs.get("self_moves", []), obs.get("opponent_moves", []))
    opp = "O" if me == "X" else "X"
    move_re = re.compile(r"C[1-3]R[1-3]")
    pos_val = _ttt_value(cells, me)
    opp_threats = _ttt_threats(cells, opp)
    for tok in _moves_near(text, _FORK_RE, move_re):
        i = _idx_ttt(tok)
        ok = i is not None and cells[i] == "." and len(_ttt_threats(_ttt_place(cells, i, me), me)) >= 2
        add("FORK", ok, i)
    for tok in _moves_near(text, _BLOCK_RE, move_re):
        i = _idx_ttt(tok)
        add("BLOCK", i is not None and i in opp_threats, i)
    for tok in _moves_near(text, _TAKEWIN_RE, move_re, guard_re=_OPP_RE):
        i = _idx_ttt(tok)
        ok = i is not None and cells[i] == "." and _ttt_wins(_ttt_place(cells, i, me), me)
        add("TAKE_WIN", ok, i)
    for tok in _moves_near(text, _BEST_RE, move_re):
        i = _idx_ttt(tok)
        if i is None or cells[i] != ".":
            add("OPTIMAL", False, i); continue
        mv_val = 1 if _ttt_wins(_ttt_place(cells, i, me), me) else -_ttt_value(_ttt_place(cells, i, me), opp)
        add("OPTIMAL", mv_val == pos_val, i)


def _check_c4(text, obs, add):
    heights, grid, me = _c4_board(obs.get("self_moves", []), obs.get("opponent_moves", []))
    opp = "O" if me == "X" else "X"
    move_re = re.compile(r"C[1-7]|[Cc]olumn\s*[1-7]")
    for tok in _moves_near(text, _BLOCK_RE, move_re):
        c = _col_c4(tok)
        add("BLOCK", c is not None and _c4_drop_wins(heights, grid, c, opp), c)
    for tok in _moves_near(text, _TAKEWIN_RE, move_re, guard_re=_OPP_RE):
        c = _col_c4(tok)
        add("TAKE_WIN", c is not None and _c4_drop_wins(heights, grid, c, me), c)
    for tok in _moves_near(text, _FORK_RE, move_re):
        c = _col_c4(tok)
        # ponytail: THREAT ok if the drop leaves >=1 winning follow-up; require >=2 only if precision matters
        add("THREAT", c is not None and _c4_threats_after(heights, grid, c, me) >= 1, c)


def _check_pd(text, obs, add):
    """Prisoner's Dilemma (years: T/S=(0,3), TT=(2,2), SS=(1,1)). Testify strictly
    dominates Silent — a decidable fact of the stated payoffs. (Long-run cooperation needs
    the opponent's future strategy -> refused.) The dominance keyword and the named action
    can be ~40 chars apart ("the dominant strategy ... is to testify"), so use a wide window
    and credit by whichever action ("testify"/"defect" vs "silent"/"cooperate") is nearest."""
    for m in _DOM_RE.finditer(text):
        seg = text[max(0, m.start() - 20): m.end() + 70].lower()
        i_t = min([p for p in (seg.find("testif"), seg.find("defect")) if p >= 0], default=-1)
        i_s = min([p for p in (seg.find("silent"), seg.find("cooperat")) if p >= 0], default=-1)
        if i_t >= 0 and (i_s < 0 or i_t <= i_s):
            add("DOMINANCE", True, "testify_dom")
        elif i_s >= 0:
            add("DOMINANCE", False, "silent_dom")   # Silent is dominated, not dominant
    # PAYOFF: correct testify-advantage reasoning without the word "dominant" (high recall).
    # Credit when an advantage phrase sits within ~45 chars of "testify"/"defect" but is
    # NOT closer to "silent"/"cooperate" (avoid crediting "stay silent to go free").
    low = text.lower()
    for adv, key in ((r"(?:go|walk|set|get out|be)\s+free|\b0\s*years?|no (?:jail|prison|time)", "free"),
                     (r"(?:few|less|reduc|lower|minimi|shorter|light)\w*\s*(?:\w+\s+){0,3}(?:sentence|years?|time|prison)", "less")):
        for a in re.finditer(adv, low):
            ctx = low[max(0, a.start() - 45): a.end() + 45]
            near_t = "testif" in ctx or "defect" in ctx
            near_s = "silent" in ctx or "cooperat" in ctx
            if near_t and not near_s:
                add("PAYOFF", True, "testify_" + key)
                break


def _check_auction(text, obs, add):
    """First-price sealed-bid. Legal bids are always < valuation (overbidding is illegal
    here), so a "don't overbid" claim is vacuous. The gold-free AND outcome-relevant fact:
    bidding 0 never wins (weakly dominated) and any bid >= valuation earns 0 profit (weakly
    dominated) — only an INTERIOR bid 0<b<val can both win and profit. Credit the reasoning's
    recommended bid on that; also check a stated profit = valuation - bid."""
    val = float(obs.get("valuation", 0.0))
    bids = re.findall(r"<(\d+)>", text)
    if bids and val > 0:
        b = float(bids[-1])                      # the recommended/played bid (last token)
        add("DOMINANCE", 0 < b < val, ("bid", int(b)))
    # PAYOFF: "if I win with bid b, profit = valuation - b"
    for m in re.finditer(r"(?:profit|utilit\w*|payoff|net|gain)[^\d\n]{0,25}(-?\d+(?:\.\d+)?)", text, re.IGNORECASE):
        seg = text[max(0, m.start() - 40): m.end()]
        bm = re.search(r"bid[^\d\n]{0,10}(\d+(?:\.\d+)?)", seg, re.IGNORECASE)
        if bm:
            profit = float(m.group(1)); bid = float(bm.group(1))
            add("PAYOFF", _num_ok(profit, val - bid), round(profit, 3))


def _check_kuhn(text, obs, add):
    """Kuhn poker (cards 0=J,1=Q,2=K; opponent uniform over the 2 unheld). Showdown odds
    are exact from the 3-card deck. Bet/call/bluff EV needs opponent frequencies -> refused."""
    c = int(obs.get("card", -1))
    if c not in (0, 1, 2):
        return
    p_opp_higher = (2 - c) / 2.0
    p_win = c / 2.0
    for m in re.finditer(r"(prob\w*|chance|odds|likel\w*|[0-9.]+%)([^.\n]{0,70})", text, re.IGNORECASE):
        seg = m.group(0)
        val = _first_prob(seg)
        if val is None:
            continue
        if re.search(r"(high|better|strong|opponent.*(win|beat)|lose|behind)", seg, re.IGNORECASE):
            add("PROBABILITY", _num_ok(val, p_opp_higher), ("opp_higher", round(val, 3)))
        elif re.search(r"(i .*win|win.*(showdown|hand)|showdown|ahead|better card)", seg, re.IGNORECASE):
            add("PROBABILITY", _num_ok(val, p_win), ("win", round(val, 3)))
    # DOMINANCE: K best / unbeatable ; J worst / can't win
    if re.search(r"\b(best|strongest|unbeatable|nuts|highest|can'?t lose|cannot lose|never fold)\b", text, re.IGNORECASE):
        add("DOMINANCE", c == 2, "best_hand")
    if re.search(r"\b(worst|lowest|weakest|can'?t win|cannot win|never wins?)\b", text, re.IGNORECASE):
        add("DOMINANCE", c == 0, "worst_hand")


def _check_liars(text, obs, add):
    """Liar's Dice (2 dice, no wild, own die known, opp uniform{1..6}). P(bid ⟨q,v⟩ true)
    is exact from own die. Liar-call correctness / who holds a face need the hidden die
    -> refused (never read opponent_dice_face_value)."""
    try:
        d = int(obs.get("self_dice_face_value"))
    except (TypeError, ValueError):
        return

    def p_true(q, v):
        if q == 1:
            return 1.0 if d == v else 1 / 6
        if q == 2:
            return 1 / 6 if d == v else 0.0
        return None

    # "<q dices, v value> ... probability P" or "probability ... q dice v value ... P"
    for m in re.finditer(r"(\d)\s*dice[s]?[,\s]*(?:value\s*)?(\d)|(\d)\s*value", text, re.IGNORECASE):
        # collect (q,v) then look for a nearby probability
        g = re.search(r"(\d)\s*dice[s]?[,\s]*(?:value\s*)?(\d)", m.group(0), re.IGNORECASE)
        if not g:
            continue
        q, v = int(g.group(1)), int(g.group(2))
        if q not in (1, 2) or not (1 <= v <= 6):
            continue
        seg = text[max(0, m.start() - 40): m.end() + 40]
        val = _first_prob(seg)
        exp = p_true(q, v)
        if val is not None and exp is not None and re.search(r"(prob\w*|chance|odds|likel|%)", seg, re.IGNORECASE):
            add("PROBABILITY", _num_ok(val, exp), (q, v, round(val, 3)))


CHECKERS = {
    "nim": _check_nim,
    "tictactoe": _check_ttt,
    "connect4": _check_c4,
    "python_iterated_prisoners_dilemma": _check_pd,
    "first_sealed_auction": _check_auction,
    "kuhn_poker": _check_kuhn,
    "liars_dice": _check_liars,
}


# ───────────────────────── verification ─────────────────────────
def verify_trace_gtbench(response, observation, weights=None):
    """Score one move's reasoning `response` against the gold-free facts of `observation`
    (a GTBench step observation with env_name + game state). Same return schema as
    `gold_free_verifier.verify_trace_gold_free`. Dispatches on env_name; unknown games and
    empty/None reasoning score 0."""
    weights = weights or BINARY_WEIGHTS
    text = response or ""
    env = (observation or {}).get("env_name", "")

    by_type = {k: [0, 0] for k in TYPE_WEIGHTS}
    seen_ok, seen_bad = set(), set()
    correct_w = incorrect_w = 0.0

    def add(t, ok, key):
        nonlocal correct_w, incorrect_w
        by_type[t][0 if ok else 1] += 1
        if (env, t) in UNSCORED:
            return  # extracted for the ablation table, but not scored (not move-quality-predictive)
        bucket = seen_ok if ok else seen_bad
        if (t, key) in bucket:
            return
        bucket.add((t, key))
        if ok:
            correct_w += weights[t]
        else:
            incorrect_w += weights[t]

    checker = CHECKERS.get(env)
    if checker and text:
        try:
            checker(text, observation, add)
        except Exception:
            pass  # one malformed step must not crash a run (matches reward.py)

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
    V = verify_trace_gtbench
    # ── nim (misère 1 3 5 7): first mover LOSES; nim-sum 0. ──
    nim = {"env_name": "nim", "piles": ["1", "3", "5", "7"]}
    assert _nim_sum((1, 3, 5, 7)) == 0
    assert V("The nim-sum is 0, so I'm in a losing position.", nim)["accuracy"] == 1.0
    assert V("The XOR is 4.", nim)["accuracy"] == 0.0
    assert V("This is a winning position for me.", nim)["accuracy"] == 0.0
    assert V("I'm in a losing position.", nim)["accuracy"] == 1.0
    # NIM_SUM is extracted (recorded) but UNSCORED — a bare nim-sum claim no longer moves score.
    r = V("The nim-sum is 4.", nim)                       # 4 != true nim-sum 0
    assert r["by_type"]["NIM_SUM"] == [0, 1] and r["accuracy"] == 0.0, r
    assert V("Take 2 from pile 2 to win.", {"env_name": "nim", "piles": ["1", "2", "3"]})["accuracy"] == 0.0
    win2 = {"env_name": "nim", "piles": ["1", "2", "4"]}
    r = V("I take 1 from pile 3, which wins.", win2)
    assert r["accuracy"] == 1.0 and r["by_type"]["WINNING_MOVE"] == [1, 0], r

    # ── tictactoe: I am X, top-row C1R1,C2R1 -> C3R1 wins; opp (O) C1R2,C2R2 threatens C3R2. ──
    ttt = {"env_name": "tictactoe", "self_moves": ["C1R1", "C2R1"], "opponent_moves": ["C1R2", "C2R2"]}
    assert V("I can win by playing C3R1.", ttt)["accuracy"] == 1.0
    assert V("Playing C3R2 wins the game.", ttt)["accuracy"] == 0.0
    assert V("I must block C3R2 or the opponent wins.", ttt)["accuracy"] == 1.0
    assert V("I must block C3R3.", ttt)["accuracy"] == 0.0

    # ── connect4: I am X, C4 x3 stacked -> C4 completes vertical 4. ──
    c4 = {"env_name": "connect4", "self_moves": ["C4", "C4", "C4"], "opponent_moves": ["C1", "C2", "C1"]}
    assert V("Dropping in C4 wins by completing four in a column.", c4)["accuracy"] == 1.0
    assert V("Playing C7 wins.", c4)["accuracy"] == 0.0

    # ── prisoner's dilemma: Testify dominates Silent. ──
    pd = {"env_name": "python_iterated_prisoners_dilemma"}
    assert V("Testify strictly dominates Silent, so I testify.", pd)["accuracy"] == 1.0
    assert V("Staying Silent is the dominant strategy.", pd)["accuracy"] == 0.0

    # ── first-price auction (val 8): interior bid non-dominated; bid 0 dominated; profit=val-bid. ──
    auc = {"env_name": "first_sealed_auction", "valuation": 8.0}
    assert V("After weighing it, I will bid <5>.", auc)["accuracy"] == 1.0     # 0<5<8 (undominated)
    assert V("To be safe I bid <0>.", auc)["accuracy"] == 0.0                  # bid 0 never wins
    assert V("If I bid 3 and win, my profit is 5.", auc)["accuracy"] == 1.0    # PAYOFF 8-3
    assert V("If I bid 3 and win, my profit is 2.", auc)["accuracy"] == 0.0

    # ── kuhn: K (card 2) -> P(opp higher)=0, best hand; Q (card 1) -> 0.5. ──
    kK = {"env_name": "kuhn_poker", "card": "2"}
    kQ = {"env_name": "kuhn_poker", "card": "1"}
    assert V("The probability the opponent has a higher card is 0.", kK)["accuracy"] == 1.0
    # DOMINANCE ("best/worst hand") is extracted but UNSCORED (stated ~49% correct vs random):
    r = V("K is the best hand and cannot lose.", kK)
    assert r["by_type"]["DOMINANCE"] == [1, 0] and r["accuracy"] == 0.0, r   # recorded, not scored
    assert V("K is the worst hand.", kK)["by_type"]["DOMINANCE"] == [0, 1]   # wrong, still unscored
    assert V("The chance the opponent holds a higher card is 50%.", kQ)["accuracy"] == 1.0
    assert V("The chance the opponent holds a higher card is 0.", kQ)["accuracy"] == 0.0

    # ── liar's dice: my die is 6. P(<1 dice, 6 value> true)=1; P(<1 dice, 3 value> true)=1/6. ──
    ld = {"env_name": "liars_dice", "self_dice_face_value": "6"}
    assert V("The probability of <1 dice, 6 value> being true is 1.", ld)["accuracy"] == 1.0
    assert V("The probability that 1 dice 3 value holds is 1/6.", ld)["accuracy"] == 1.0
    assert V("The probability of <1 dice, 3 value> is 0.9.", ld)["accuracy"] == 0.0

    # ── invariants: repetition can't inflate; empty/None -> 0. ──
    assert V("I can win by playing C3R1.", ttt)["gold_free_score"] == \
        V("Win at C3R1. Win at C3R1. Win at C3R1.", ttt)["gold_free_score"]
    assert V("", ttt)["gold_free_score"] == 0.0
    assert V(None, nim)["gold_free_score"] == 0.0
    print("gtbench_gold_free_verifier selfcheck OK (7 games)")


if __name__ == "__main__":
    _selfcheck()
