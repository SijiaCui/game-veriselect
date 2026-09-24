"""
GameSolve — outcome reward / accuracy metric (JSON-answer version).
===================================================================
The eval prompt requires the model to end with a fenced ```json block whose shape
mirrors the ground truth, so scoring is a direct *consistency* check between the
parsed answer and the ground truth — no free-text regex parsing, no weighted
partial credit.

  best_response : {"best_response_actions": [<row label>, ...],
                   "expected_payoffs":      [<float per row action, in row order>],
                   "best_response_value":   <float>}
  nash          : {"pure_ne":  [[<row label>, <col label>], ...],
                   "mixed_ne": [[[<sigma_row...>], [<sigma_col...>]], ...]}

Public API is unchanged (reward_breakdown / compute_reward / exact_match /
parse_nash_response / parse_br_response) so callers need no edits.

  exact       — the answer matches the ground truth: BR action set + expected
                payoffs + value; NE pure-NE set + mixed-NE *support* (which
                actions are randomised over).
  exact_full  — additionally the exact mixed probability vectors (NE only;
                identical to ``exact`` for BR).
  reward      — mean of the exact-level indicators (graded consistency in [0, 1]).

Numbers are NOT comparable to the previous weighted reward.
"""
from __future__ import annotations

import json
import re

import numpy as np

EPS_REL = 0.02       # numeric tolerance: |pred - gt| <= EPS_REL*|gt| + EPS_ABS
EPS_ABS = 0.01
SUPP_THR = 1e-6      # an action is "in the support" if its probability exceeds this
                     # (ground-truth probs are exact, so this is just a nonzero test)


def _num_ok(pred, gt):
    return bool(np.isclose(pred, gt, rtol=EPS_REL, atol=EPS_ABS))


# ───────────────────────── parsing (JSON answer block) ─────────────────────────
def extract_answer_json(text):
    """The answer object from the last ```json fenced block (or, failing that, the
    last bare {...} in the text). Returns a dict, or {} if nothing parses."""
    if not text:
        return {}
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not candidates:
        i, j = text.rfind("{"), text.rfind("}")     # fallback: last flat object (answers are flat)
        candidates = [text[i:j + 1]] if -1 < i < j else []
    for block in reversed(candidates):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def parse_br_response(text, row_labels):
    obj = extract_answer_json(text)
    label_of = {l.lower(): l for l in row_labels}
    acts = obj.get("best_response_actions") or []
    br_actions = ([label_of[str(a).strip().lower()] for a in acts
                   if str(a).strip().lower() in label_of] if isinstance(acts, list) else [])
    eu = obj.get("expected_payoffs") or []
    try:
        expected = [float(x) for x in eu] if isinstance(eu, list) else []
    except Exception:
        expected = []
    val = obj.get("best_response_value")
    try:
        br_value = float(val) if val is not None else None
    except Exception:
        br_value = None
    return {"br_actions": br_actions, "expected_payoffs": expected, "br_value": br_value}


def parse_nash_response(text, row_labels, col_labels):
    obj = extract_answer_json(text)
    row_of = {l.lower(): l for l in row_labels}
    col_of = {l.lower(): l for l in col_labels}
    pure = []
    for pair in obj.get("pure_ne") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            r, c = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
            if r in row_of and c in col_of and (row_of[r], col_of[c]) not in pure:
                pure.append((row_of[r], col_of[c]))
    mixed = []
    for eq in obj.get("mixed_ne") or []:
        if (isinstance(eq, (list, tuple)) and len(eq) == 2
                and isinstance(eq[0], list) and isinstance(eq[1], list)):
            try:
                mixed.append(([float(x) for x in eq[0]], [float(x) for x in eq[1]]))
            except Exception:
                pass
    return {"pure_ne": pure, "mixed_ne": mixed}


# ───────────────────────── best-response scoring ─────────────────────────
def _br_components(text, gt, row_labels):
    p = parse_br_response(text, row_labels)
    gt_actions = {row_labels[i] for i in gt["best_response_actions"]}
    gt_eu = gt["expected_payoffs"]
    action = 1.0 if set(p["br_actions"]) == gt_actions else 0.0
    eu = 1.0 if (len(p["expected_payoffs"]) == len(gt_eu)
                 and all(_num_ok(a, b) for a, b in zip(p["expected_payoffs"], gt_eu))) else 0.0
    value = 1.0 if (p["br_value"] is not None
                    and _num_ok(p["br_value"], gt["best_response_value"])) else 0.0
    return {"action": action, "eu": eu, "value": value}


# ───────────────────────── nash scoring ─────────────────────────
def _support(vec):
    return frozenset(i for i, pr in enumerate(vec) if abs(pr) > SUPP_THR)


def _match_mixed(pred, gt_mixed, with_vectors):
    """Greedy 1-1 matching of predicted mixed equilibria to ground-truth ones.
    Requires equal count (no missing, no extra). ``with_vectors`` additionally
    checks the probabilities within tolerance; otherwise only the support."""
    if len(pred) != len(gt_mixed):
        return False
    used = [False] * len(pred)
    for g in gt_mixed:
        for i, pm in enumerate(pred):
            if used[i]:
                continue
            if (len(pm[0]) == len(g[0]) and len(pm[1]) == len(g[1])
                    and _support(pm[0]) == _support(g[0]) and _support(pm[1]) == _support(g[1])
                    and (not with_vectors
                         or (all(_num_ok(a, b) for a, b in zip(pm[0], g[0]))
                             and all(_num_ok(a, b) for a, b in zip(pm[1], g[1]))))):
                used[i] = True
                break
        else:
            return False
    return True


def _ne_components(text, gt, row_labels, col_labels):
    p = parse_nash_response(text, row_labels, col_labels)
    gt_eqs = gt["equilibria"]
    gt_pure = {(row_labels[e["sigma_row"].index(max(e["sigma_row"]))],
                col_labels[e["sigma_col"].index(max(e["sigma_col"]))])
               for e in gt_eqs if e["is_pure"]}
    gt_mixed = [(e["sigma_row"], e["sigma_col"]) for e in gt_eqs if not e["is_pure"]]
    pure = 1.0 if set(p["pure_ne"]) == gt_pure else 0.0
    support = 1.0 if _match_mixed(p["mixed_ne"], gt_mixed, with_vectors=False) else 0.0
    vec = 1.0 if _match_mixed(p["mixed_ne"], gt_mixed, with_vectors=True) else 0.0
    return {"pure": pure, "support": support, "vec": vec}


# ───────────────────────── public API ─────────────────────────
def reward_breakdown(response_text, ground_truth, task, row_labels, col_labels):
    """Per-component scores, graded reward, and two exact flags (see module docstring)."""
    if not response_text or not response_text.strip():
        return {"reward": 0.0, "exact": False, "exact_full": False, "components": {}}
    try:
        if task == "nash_equilibrium":
            c = _ne_components(response_text, ground_truth, row_labels, col_labels)
            exact = c["pure"] == 1.0 and c["support"] == 1.0
            exact_full = exact and c["vec"] == 1.0
            reward = float(np.mean([c["pure"], c["support"]]))
        elif task == "best_response":
            c = _br_components(response_text, ground_truth, row_labels)
            exact = c["action"] == 1.0 and c["eu"] == 1.0 and c["value"] == 1.0
            exact_full = exact
            reward = float(np.mean([c["action"], c["eu"], c["value"]]))
        else:
            raise ValueError(f"unknown task: {task}")
        return {"reward": reward, "exact": bool(exact),
                "exact_full": bool(exact_full), "components": c}
    except Exception:
        # defensive: one malformed sample must not crash a long eval run
        return {"reward": 0.0, "exact": False, "exact_full": False, "components": {}}


def compute_reward(response_text, ground_truth, task, row_labels, col_labels):
    """Graded reward in [0, 1] (drop-in replacement for the original signature)."""
    return reward_breakdown(response_text, ground_truth, task, row_labels, col_labels)["reward"]


def exact_match(response_text, ground_truth, task, row_labels, col_labels):
    """PRIMARY 0/1 correctness: BR fully correct, or NE pure+mixed-support correct."""
    return 1.0 if reward_breakdown(
        response_text, ground_truth, task, row_labels, col_labels)["exact"] else 0.0


# ───────────────────────── self-check ─────────────────────────
def _selfcheck():
    rl, cl = ["A", "B"], ["Y", "Z"]

    # best response: correct answer, trailing prose after the block is ignored
    gt_br = {"best_response_actions": [0], "expected_payoffs": [2.9072, -0.9652],
             "best_response_value": 2.9072}
    good = ('Reasoning...\nANSWER:\n```json\n{"best_response_actions": ["A"], '
            '"expected_payoffs": [2.91, -0.97], "best_response_value": 2.91}\n```')
    bd = reward_breakdown(good, gt_br, "best_response", rl, cl)
    assert bd["exact"] and bd["reward"] == 1.0, bd
    assert reward_breakdown(good + "\n\nSo A is best.", gt_br, "best_response", rl, cl)["exact"]
    assert not reward_breakdown(good.replace('["A"]', '["B"]'), gt_br, "best_response", rl, cl)["exact"]
    assert not reward_breakdown(good.replace("-0.97", "0.5"), gt_br, "best_response", rl, cl)["exact"]

    # nash pure: right set is exact; a spurious mixed claim breaks consistency
    gt_pure = {"equilibria": [{"sigma_row": [1.0, 0.0], "sigma_col": [0.0, 1.0], "is_pure": True}],
               "equilibrium_class": "pure"}
    ne_ok = 'ANSWER:\n```json\n{"pure_ne": [["A", "Z"]], "mixed_ne": []}\n```'
    bd = reward_breakdown(ne_ok, gt_pure, "nash_equilibrium", rl, cl)
    assert bd["exact"] and bd["exact_full"], bd
    ne_spurious = 'ANSWER:\n```json\n{"pure_ne": [["A", "Z"]], "mixed_ne": [[[0.5,0.5],[0.5,0.5]]]}\n```'
    assert not reward_breakdown(ne_spurious, gt_pure, "nash_equilibrium", rl, cl)["exact"]

    # nash mixed: support (incl. a genuine tiny 0.045 support prob) vs exact vectors
    gt_mix = {"equilibria": [{"sigma_row": [0.955, 0.045], "sigma_col": [0.5, 0.5], "is_pure": False}],
              "equilibrium_class": "mixed"}
    supp_only = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.9, 0.1], [0.5, 0.5]]]}\n```'
    bd = reward_breakdown(supp_only, gt_mix, "nash_equilibrium", rl, cl)
    assert bd["exact"] and not bd["exact_full"], bd
    exact_v = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.955, 0.045], [0.5, 0.5]]]}\n```'
    assert reward_breakdown(exact_v, gt_mix, "nash_equilibrium", rl, cl)["exact_full"]
    # dropping the tiny-support action is now a genuine support mismatch
    drop = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[1.0, 0.0], [0.5, 0.5]]]}\n```'
    assert not reward_breakdown(drop, gt_mix, "nash_equilibrium", rl, cl)["exact"]
    # a wrong-dimensionality vector must NOT match on support alone (len guard)
    wrongdim = 'ANSWER:\n```json\n{"pure_ne": [], "mixed_ne": [[[0.955, 0.045, 0.0], [0.5, 0.5]]]}\n```'
    assert not reward_breakdown(wrongdim, gt_mix, "nash_equilibrium", rl, cl)["exact"]

    # no code fence: extract the last flat object, ignoring an earlier prose brace
    nofence = ('the set {A, B} matters.\nANSWER:\n{"best_response_actions": ["A"], '
               '"expected_payoffs": [2.91, -0.97], "best_response_value": 2.91}')
    assert reward_breakdown(nofence, gt_br, "best_response", rl, cl)["exact"], "no-fence fallback"

    # garbage / no JSON scores 0 without raising
    assert reward_breakdown("no answer here", gt_br, "best_response", rl, cl)["reward"] == 0.0
    print("reward.py self-check: OK")


if __name__ == "__main__":
    _selfcheck()
