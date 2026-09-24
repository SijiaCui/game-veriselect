"""
Game-Theoretic Process Reward (GT-PRM) verifier.

Extracts formally-verifiable intermediate reasoning claims from an LLM's free-form
game-solving trace and checks each against the ground-truth game structure (payoff
matrices + solver ground truth). Zero human annotation, zero rollouts.

Verifiable claim types:
  - DOMINANCE   : "<X> dominates <Y>"  /  "<Y> is dominated"   → check payoff matrix
  - EQUILIBRIUM : "(<row>, <col>) is a Nash equilibrium"        → mutual best response
  - BR_ACTION   : "best response is <action>"                   → vs ground-truth BR
  - EXP_PAYOFF  : "expected payoff for <action> is <v>"         → vs ground-truth EUs

Returns per-trace statistics and a scalar process score in [0, 1] that rewards
BOTH accuracy (fraction of claims correct) AND coverage (covering real solution facts),
so it cannot be trivially gamed by emitting one correct claim or many wrong ones.
"""
import json
import re
import numpy as np

# weights for the "typed" variant (NE hardest → highest); binary uses all 1.0
TYPE_WEIGHTS = {"EQUILIBRIUM": 3.0, "BR_ACTION": 2.0, "EXP_PAYOFF": 1.5,
                "DOMINANCE": 1.0}
BINARY_WEIGHTS = {k: 1.0 for k in TYPE_WEIGHTS}

_SUPP_THR = 1e-6   # an action is "in support" if its prob exceeds this; gt probs are exact,
                   # so this is just a nonzero test (matches reward.py SUPP_THR)


# ───────────────────────── game-theory primitives ─────────────────────────
def _row_weakly_dominates(A, i, k):
    """Row strategy i weakly dominates k for the ROW player (chooses rows)."""
    ge = all(A[i][j] >= A[k][j] for j in range(len(A[0])))
    gt = any(A[i][j] > A[k][j] for j in range(len(A[0])))
    return ge and gt


def _col_weakly_dominates(B, j, l):
    """Col strategy j weakly dominates l for the COL player (chooses cols)."""
    ge = all(B[i][j] >= B[i][l] for i in range(len(B)))
    gt = any(B[i][j] > B[i][l] for i in range(len(B)))
    return ge and gt


def _is_pure_ne(A, B, i, j):
    nrow, ncol = len(A), len(A[0])
    row_br = A[i][j] >= max(A[r][j] for r in range(nrow)) - 1e-9
    col_br = B[i][j] >= max(B[i][c] for c in range(ncol)) - 1e-9
    return row_br and col_br


def _support_of(vec):
    v = np.asarray(vec, float)
    if v.sum() > 0:
        v = v / v.sum()
    return frozenset(np.where(v > _SUPP_THR)[0].tolist())


def _is_mixed_ne(sigma_r, sigma_c, equilibria):
    """True iff the claimed profile randomises over the SAME actions as some true
    equilibrium — the benchmark's PRIMARY correctness criterion (`exact` in reward.py is
    pure + mixed-SUPPORT, NOT the exact probabilities). Support is discrete, so
    this is robust to the model's probability rounding (1-/2-decimal answers still match)
    yet cannot be gamed by nudging mass onto a non-support action (that changes the
    support). Supports come from the game's solved equilibria and are matched with the same
    near-zero (nonzero) threshold reward.py uses, so verifier and benchmark agree exactly.

    (An indifference-gap tolerance cannot do this job: a correct-but-rounded profile and a
    close-but-wrong one produce overlapping gaps, so any single threshold both admits wrong
    profiles and rejects coarsely-rounded correct ones. Support-matching sidesteps that.)"""
    if not equilibria:
        return False
    sr = np.asarray(sigma_r, float); sc = np.asarray(sigma_c, float)
    if sr.sum() <= 0 or sc.sum() <= 0:
        return False
    claim = (_support_of(sr), _support_of(sc))
    return any(claim == (_support_of(e["sigma_row"]), _support_of(e["sigma_col"]))
               for e in equilibria)


# ───────────────────────── claim extraction ─────────────────────────
def _label_alt(labels):
    return "|".join(re.escape(l) for l in labels)


def _answer_json(text):
    """The answer object from the last ```json fenced block (or last bare {...}), else {}.
    Same schema/extraction as bench/gamesolve/reward.py — so the verifier reads the eval
    answer the same way the reward does. Free-text reasoning is still parsed by the regex
    extractors below; this only adds the claims stated in the final JSON ANSWER block."""
    if not text:
        return {}
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        i, j = text.rfind("{"), text.rfind("}")     # fallback: last flat object (answers are flat)
        blocks = [text[i:j + 1]] if -1 < i < j else []
    for b in reversed(blocks):
        try:
            obj = json.loads(b)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def extract_dominance_claims(text, row_labels, col_labels):
    """Return list of (dominant_label, dominated_label)."""
    claims = []
    alll = _label_alt(row_labels + col_labels)
    # "X (strictly|weakly) dominates Y" — labels matched case-sensitively (single-char
    # labels A-E/V-Z would otherwise collide with English words under IGNORECASE).
    for m in re.finditer(rf"\b(?-i:({alll}))\b\s+(?:strictly\s+|weakly\s+)?dominat\w*\s+(?:over\s+)?\b(?-i:({alll}))\b",
                         text, re.IGNORECASE):
        claims.append((m.group(1), m.group(2)))
    return claims


def extract_ne_claims(text, row_labels, col_labels):
    """Return list of (row_label, col_label) claimed as a (pure) Nash equilibrium.
    Only counts pairs appearing within ~60 chars of an NE keyword."""
    claims = []
    rl, cl = _label_alt(row_labels), _label_alt(col_labels)
    pair_pat = rf"\(\s*(?-i:({rl}))\s*,\s*(?-i:({cl}))\s*\)"
    for m in re.finditer(pair_pat, text, re.IGNORECASE):
        window = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "nash" in window or re.search(r"\bne\b", window) or "equilibri" in window:
            claims.append((m.group(1), m.group(2)))
    # ANSWER-block JSON: {"pure_ne": [["A","Y"], ...]}
    row_of = {l.lower(): l for l in row_labels}
    col_of = {l.lower(): l for l in col_labels}
    for pair in _answer_json(text).get("pure_ne") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            r, c = row_of.get(str(pair[0]).strip().lower()), col_of.get(str(pair[1]).strip().lower())
            if r and c:
                claims.append((r, c))
    return claims


def extract_mixed_ne_claims(text, row_labels, col_labels):
    """Return list of (sigma_row, sigma_col) probability-vector pairs claimed as a MIXED
    Nash equilibrium, in the answer format `([p,..],[q,..])`. Gated on an NE/mixed keyword
    appearing from the start of the pair's line, so ALL equilibria in a one-line
    'Mixed NE: [(..),(..),..]' list are captured (not just the first)."""
    claims = []
    pat = r"\(\s*\[([-\d.,\s]+)\]\s*,\s*\[([-\d.,\s]+)\]\s*\)"
    for m in re.finditer(pat, text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        ctx = (text[line_start:m.start()] + text[m.start(): m.end() + 60]).lower()
        if not ("nash" in ctx or re.search(r"\bne\b", ctx) or "equilibri" in ctx or "mixed" in ctx):
            continue
        try:
            sr = [float(x) for x in m.group(1).split(",") if x.strip()]
            sc = [float(x) for x in m.group(2).split(",") if x.strip()]
        except ValueError:
            continue
        if sr and sc:
            claims.append((sr, sc))
    # ANSWER-block JSON: {"mixed_ne": [[[p..],[q..]], ...]}
    for eq in _answer_json(text).get("mixed_ne") or []:
        if (isinstance(eq, (list, tuple)) and len(eq) == 2
                and isinstance(eq[0], list) and isinstance(eq[1], list)):
            try:
                jsr = [float(x) for x in eq[0]]
                jsc = [float(x) for x in eq[1]]
            except (TypeError, ValueError):
                continue
            if jsr and jsc:
                claims.append((jsr, jsc))
    return claims


def extract_br_action_claims(text, row_labels):
    """Return list of claimed best-response action labels (row player).
    Handles: 'best response is/: X', 'best response: X', 'Action(s): {X}'."""
    claims = []
    rl = _label_alt(row_labels)
    # 'best response' then the first case-sensitive label within a short window. The
    # keyword match does NOT consume the window, so a following 'best response' is not
    # skipped; labels are case-sensitive so the article 'a' can't be read as label 'A'.
    for m in re.finditer(r"best\s+response", text, re.IGNORECASE):
        seg = text[m.end(): m.end() + 40]
        lm = re.search(rf"\b(?-i:({rl}))\b", seg)
        if lm:
            claims.append(lm.group(1))
    # 'Action(s): {X}' restatement
    for m in re.finditer(rf"Action\(?s?\)?\s*[:=]\s*[\[{{]?\s*\b(?-i:({rl}))\b", text, re.IGNORECASE):
        claims.append(m.group(1))
    # ANSWER-block JSON: {"best_response_actions": ["A", ...]}
    row_of = {l.lower(): l for l in row_labels}
    for a in _answer_json(text).get("best_response_actions") or []:
        lab = row_of.get(str(a).strip().lower())
        if lab:
            claims.append(lab)
    return list(dict.fromkeys(claims))   # a BR action is a distinct fact; don't double-count


def extract_exp_payoff_claims(text, row_labels):
    """Return list of (action_label, value) for per-action expected-payoff claims.
    Handles 'EU(X) = ... = v' and 'expected payoff for X ... v'."""
    claims = []
    rl = _label_alt(row_labels)
    # Match only the 'EU(X)'/'expected payoff for X' head (non-consuming), then take the
    # RESULT number from this action's own segment: rest of the LINE, cut before the next
    # EU head, so 'EU(A)=1.2 and EU(B)=3.4' scores A against 1.2 and a long multi-term
    # 'EU(A) = w1 + w2 + ... = 2.31' line still scores A against its final result 2.31.
    for m in re.finditer(rf"(?:EU|expected\s+payoff)\s*(?:for\s+|\(\s*)?(?-i:({rl}))\b", text, re.IGNORECASE):
        seg = text[m.end():].split("\n", 1)[0]
        seg = re.split(r"(?:EU|expected\s+payoff)", seg, flags=re.IGNORECASE)[0]
        nums = re.findall(r"-?\d+\.?\d*", seg)
        if nums:
            claims.append((m.group(1), float(nums[-1])))
    # ANSWER-block JSON: {"expected_payoffs": [v0, v1, ...]} in row order
    eu = _answer_json(text).get("expected_payoffs")
    if isinstance(eu, list):
        for i, v in enumerate(eu):
            if i < len(row_labels):
                try:
                    claims.append((row_labels[i], float(v)))
                except (TypeError, ValueError):
                    pass
    return claims


def _label_idx(label, labels):
    for i, l in enumerate(labels):
        if l.lower() == label.lower():
            return i
    return None


# ───────────────────────── verification ─────────────────────────
def verify_trace(response, game, weights=None):
    """
    game: dict with payoff_matrix_row (A), payoff_matrix_col (B), row_labels,
          col_labels, task, ground_truth(dict).
    Returns dict with counts and process_score in [0,1].
    """
    weights = weights or BINARY_WEIGHTS
    A = game["payoff_matrix_row"]
    B = game["payoff_matrix_col"]
    row_labels = game["row_labels"]
    col_labels = game["col_labels"]
    task = game.get("task", "nash_equilibrium")
    gt = game.get("ground_truth", {})

    correct_w = 0.0
    incorrect_w = 0.0
    by_type = {k: [0, 0] for k in TYPE_WEIGHTS}  # type -> [correct, incorrect]

    def record(t, ok):
        by_type[t][0 if ok else 1] += 1
        return weights[t] if ok else 0.0, 0.0 if ok else weights[t]

    # DOMINANCE
    for dom, dnt in extract_dominance_claims(response, row_labels, col_labels):
        if dom in row_labels and dnt in row_labels:
            i, k = _label_idx(dom, row_labels), _label_idx(dnt, row_labels)
            ok = i is not None and k is not None and i != k and _row_weakly_dominates(A, i, k)
        elif dom in col_labels and dnt in col_labels:
            j, l = _label_idx(dom, col_labels), _label_idx(dnt, col_labels)
            ok = j is not None and l is not None and j != l and _col_weakly_dominates(B, j, l)
        else:
            continue  # cross-set claim is ill-typed; skip
        c, w = record("DOMINANCE", ok)
        correct_w += c; incorrect_w += w

    # EQUILIBRIUM (Nash task)
    matched = set()   # distinct true-equilibrium supports the trace correctly identifies (recall)
    if task == "nash_equilibrium":
        gt_supports = {(_support_of(e["sigma_row"]), _support_of(e["sigma_col"]))
                       for e in gt.get("equilibria", [])}
        for (rlab, clab) in extract_ne_claims(response, row_labels, col_labels):
            i, j = _label_idx(rlab, row_labels), _label_idx(clab, col_labels)
            ok = i is not None and j is not None and _is_pure_ne(A, B, i, j)
            if ok:
                matched.add((frozenset([i]), frozenset([j])))
            c, w = record("EQUILIBRIUM", ok)
            correct_w += c; incorrect_w += w
        # mixed-strategy NE claims '([p,..],[q,..])' — support-matched vs solved equilibria
        for (sr, sc) in extract_mixed_ne_claims(response, row_labels, col_labels):
            ok = _is_mixed_ne(sr, sc, gt.get("equilibria", []))
            if ok:
                matched.add((_support_of(sr), _support_of(sc)))
            c, w = record("EQUILIBRIUM", ok)
            correct_w += c; incorrect_w += w
        matched &= gt_supports   # count only genuine equilibria

    # BEST RESPONSE (BR task)
    if task == "best_response":
        gt_actions = set(gt.get("best_response_actions", []))
        for lab in extract_br_action_claims(response, row_labels):
            idx = _label_idx(lab, row_labels)
            ok = idx is not None and idx in gt_actions
            c, w = record("BR_ACTION", ok)
            correct_w += c; incorrect_w += w
        # per-action expected payoff claims: 'EU(label) = ... = v'
        eus = gt.get("expected_payoffs", [])
        if eus:
            for lab, v in extract_exp_payoff_claims(response, row_labels):
                idx = _label_idx(lab, row_labels)
                if idx is None or idx >= len(eus):
                    continue
                ok = abs(v - eus[idx]) < 0.10 * (abs(eus[idx]) + 1)
                c, w = record("EXP_PAYOFF", ok)
                correct_w += c; incorrect_w += w

    # ── aggregate: accuracy × coverage ──
    n_correct = sum(v[0] for v in by_type.values())
    n_incorrect = sum(v[1] for v in by_type.values())
    n_total = n_correct + n_incorrect
    accuracy = correct_w / (correct_w + incorrect_w) if (correct_w + incorrect_w) > 0 else 0.0

    # coverage = recall over the game's facts
    if task == "nash_equilibrium":
        # fraction of the game's equilibria correctly identified (mirrors reward.py's recall:
        # per-equilibrium, so a support shared by two equilibria covers both, and repeating
        # one can't inflate). Set-based `matched` means repetition is free of effect.
        eqs = gt.get("equilibria", [])
        n_eq = max(1, gt.get("n_equilibria", len(eqs)) or 1)
        covered = sum(1 for e in eqs
                      if (_support_of(e["sigma_row"]), _support_of(e["sigma_col"])) in matched)
        coverage = covered / n_eq
    else:
        target = max(1.0, float(1 + len(gt.get("expected_payoffs", []))))  # BR action + EUs
        coverage = correct_w / (target * np.mean(list((weights or BINARY_WEIGHTS).values())))
    coverage = min(1.0, coverage)

    process_score = 0.5 * accuracy + 0.5 * coverage

    # Selector signal: blends accuracy with the absolute count of correct claims (no
    # ground-truth normalization). DOMINANCE and pure-NE EQUILIBRIUM are checked purely
    # against the payoff matrix; mixed-NE EQUILIBRIUM is support-matched against the solved
    # equilibria (structure only, not the outcome/exact-match reward).
    gold_free_score = correct_w / (correct_w + incorrect_w + 1.0)

    return {
        "process_score": float(process_score),
        "gold_free_score": float(gold_free_score),
        "accuracy": float(accuracy),
        "coverage": float(coverage),
        "correct_w": float(correct_w), "incorrect_w": float(incorrect_w),
        "n_correct": n_correct, "n_incorrect": n_incorrect, "n_total": n_total,
        "by_type": by_type,
    }


def _selfcheck():
    # Claims stated in the JSON ANSWER block are read the same as the old free-text block,
    # and checked against ground_truth. (Free-text reasoning claims still work too.)
    mix = {
        "payoff_matrix_row": [[5.0, 1.0], [2.0, 4.0]],
        "payoff_matrix_col": [[-2.0, -1.0], [4.0, -3.0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"], "task": "nash_equilibrium",
        "ground_truth": {"equilibria": [{"sigma_row": [0.875, 0.125], "sigma_col": [0.5, 0.5],
                                         "is_pure": False}], "n_equilibria": 1},
    }
    js_ok = 'reason\nANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.88,0.12],[0.5,0.5]]]}\n```'
    txt_ok = 'ANSWER:\nMixed NE: [([0.88,0.12],[0.50,0.50])]'
    js_bad = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[1,0],[0,1]]]}\n```'
    for r in (js_ok, txt_ok):
        v = verify_trace(r, mix)
        assert v["accuracy"] == 1.0 and v["coverage"] == 1.0, (r, v)
    assert verify_trace(js_bad, mix)["accuracy"] == 0.0, verify_trace(js_bad, mix)

    # thin support (0.03) matches reward.py's 1e-6: keeping the tiny action matches, dropping it does not
    thin = dict(mix, ground_truth={"equilibria": [{"sigma_row": [0.97, 0.03], "sigma_col": [0.5, 0.5],
                                                    "is_pure": False}], "n_equilibria": 1})
    keep = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.9, 0.1], [0.5, 0.5]]]}\n```'
    dropt = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[1.0, 0.0], [0.5, 0.5]]]}\n```'
    assert verify_trace(keep, thin)["accuracy"] == 1.0, verify_trace(keep, thin)
    assert verify_trace(dropt, thin)["accuracy"] == 0.0, verify_trace(dropt, thin)

    br = {
        "payoff_matrix_row": [[-5.0, 3.0], [2.0, -1.0]], "payoff_matrix_col": [[0, 0], [0, 0]],
        "row_labels": ["A", "B"], "col_labels": ["Y", "Z"], "task": "best_response",
        "ground_truth": {"best_response_actions": [0], "expected_payoffs": [2.9072, -0.9652]},
    }
    js_br = ('ANSWER:\n```json\n{"best_response_actions": ["A"], '
             '"expected_payoffs": [2.91, -0.97], "best_response_value": 2.91}\n```')
    assert verify_trace(js_br, br)["accuracy"] == 1.0, verify_trace(js_br, br)
    assert verify_trace(js_br.replace('["A"]', '["B"]'), br)["accuracy"] < 1.0
    # no code fence + an earlier prose brace: still reads the last flat object
    nofence = ('best options are {A, B}.\nANSWER:\n{"best_response_actions": ["A"], '
               '"expected_payoffs": [2.9072, -0.9652]}')
    assert verify_trace(nofence, br)["accuracy"] == 1.0, verify_trace(nofence, br)
    print("gt_prm_verifier selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
