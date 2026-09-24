"""
GameSolve-Hard evaluation engine (shared; --enable_thinking toggles Qwen3 thinking).

Series-specific entry points set only their own defaults and call run():
  eval_qwen25.py  — thinking off,  8192/10240 token budget
  eval_qwen3.py   — thinking flag, 16384/18432 token budget
(and baselines/gamesolve_run.py + eval_think_diag.py reuse the pure helpers here.)

For one model: load the ID + OOD hard benchmarks, take a deterministic stratified
subset (identical across models for a fixed seed/per_group), sample n traces per
prompt with vLLM (single call, shared prefill), score each trace with the NEW
reward (bench/gamesolve/reward.py) using both strict exact_match (PRIMARY) and the
graded compute_reward (secondary), and compute per (split x tier x task):

  pass@1     = mean over prompts of (mean over the n draws of exact_match)
  majority@n = fraction of prompts whose modal canonical answer is exact-correct
               (modal canon via canon_answer, random tiebreak; project analyze.py rule)
  oracle@n   = fraction of prompts with >= 1 exact-correct draw (headroom ceiling)
  pass@1(graded) = mean over prompts of (mean over draws of compute_reward)

Per prompt also records truncation diagnostics (mirrors eval_think_diag.py), so a low
score can be told apart from a truncated / unparseable generation. Aggregated per group:
  trunc_frac     = draws that hit max_tokens (finish_reason=="length") OR have no ANSWER:
  no_answer_frac = draws with no "ANSWER:" marker
  length_frac    = draws that hit the token cap
  mean_gen_toks / max_gen_toks = generated tokens per draw (how close to the cap)
(no </think> diagnostic: this is the non-thinking run.)

GPU pinning: pass --gpu 0|1 ; CUDA_VISIBLE_DEVICES is set before importing vllm.

Prompt format / canon are copied from the project's veriselect/generate.py so the
protocol matches; parsers/scorer come from the NEW bench reward.py.
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# ---- paths ----
# bench/gamesolve of THIS checkout (eval/gamesolve/ -> <repo>/bench/gamesolve), overridable
# with GAMESOLVE_BENCH for a bench tree kept elsewhere.
BENCH_DIR = os.environ.get("GAMESOLVE_BENCH") or str(
    Path(__file__).resolve().parents[2] / "bench" / "gamesolve"
)
sys.path.insert(0, BENCH_DIR)
import reward as R  # NEW reward.py: exact_match, compute_reward, parse_nash_response, parse_br_response

import numpy as np

# ─────────────────────────── prompt format (from generate.py) ───────────────────────────
SYSTEM_PROMPT = """You are a game-theory expert. Show your reasoning, then give the answer as JSON.

Reason step by step and SHOW it: write each expected payoff as "EU(<action>) = <value>", note any "<X> dominates <Y>", and state each Nash equilibrium and why (no profitable deviation; for a mixed NE, the indifference condition).

Then end your response with a line "ANSWER:" followed by ONE fenced ```json block containing a SINGLE JSON object in the exact format given in the task instruction, and nothing after it. Do not output any other format or label. Use the exact action labels and valid JSON.
"""


def format_prompt(sample, style):
    desc = sample["descriptions"][style]
    if sample["task"] == "nash_equilibrium":
        instruction = (
            "Find all Nash Equilibria of this game (state pure vs mixed; for pure NE give the "
            "strategy pair, for mixed NE the probability distribution over each player's actions).\n"
            "Format the ANSWER as exactly this single JSON object (no labels, no other keys):\n"
            '{"pure_ne": [["<row_action>", "<col_action>"], ...], '
            '"mixed_ne": [[[<row probs>], [<col probs>]], ...]}\n'
            "pure_ne: list of [row_action, col_action] label pairs, or [] if none. "
            "mixed_ne: one [row_probs, col_probs] per mixed equilibrium, each probability vector "
            "in action order and summing to 1, or [] if none.")
    else:
        instruction = (
            "Compute the best response for the row player given the column player's strategy: "
            "(1) the expected payoff of each row action, (2) the best response action(s), and "
            "(3) the best response expected payoff value.\n"
            "Format the ANSWER as exactly this single JSON object (no labels, no other keys):\n"
            '{"best_response_actions": ["<row_action>", ...], '
            '"expected_payoffs": [<eu per row action, in order>], "best_response_value": <number>}\n'
            "best_response_actions: the row action label(s) with the maximum expected payoff. "
            "expected_payoffs: the expected payoff of each row action, in the given row order.")
    return f"{desc}\n\n{instruction}"


def canon_answer(response, sample):
    """Canonical answer key for self-consistency / majority vote (structure only)."""
    try:
        if sample["task"] == "nash_equilibrium":
            p = R.parse_nash_response(response, sample["row_labels"], sample["col_labels"])
            return json.dumps({"pure": sorted(map(list, p["pure_ne"])), "mixed": len(p["mixed_ne"]) > 0}, sort_keys=True)
        else:
            p = R.parse_br_response(response, sample["row_labels"])
            return json.dumps({"br": sorted(p["br_actions"])}, sort_keys=True)
    except Exception:
        return "__parse_error__"


# ─────────────────────────── subset selection ───────────────────────────
def load_split(path):
    return [json.loads(l) for l in open(path)]


def stratified_subset(samples, group_key, per_group, seed):
    """Deterministic: group by group_key, sort each group by id, seeded-shuffle, take per_group.
    Depends only on (data, group_key, per_group, seed) -> identical across models."""
    groups = defaultdict(list)
    for s in samples:
        groups[group_key(s)].append(s)
    picked = []
    for gk in sorted(groups.keys(), key=lambda x: str(x)):
        g = sorted(groups[gk], key=lambda s: s["id"])
        rng = random.Random(f"{seed}|{gk}")
        rng.shuffle(g)
        picked.extend(g[:per_group])
    return picked


def assign_style(sample):
    keys = sorted(sample["descriptions"].keys())
    h = int(hashlib.md5(str(sample["id"]).encode()).hexdigest(), 16)
    return keys[h % len(keys)]


def tier_of(s):
    return s.get("metadata", {}).get("difficulty", "?")


def build_prompt_set(bench_path, per_group, seed):
    """Returns list of samples tagged with split/_style. Loads ID and sibling OOD."""
    p = Path(bench_path)
    out = []
    if p.name.endswith("_ood.jsonl"):
        id_path, ood_path = None, p
    else:
        id_path = p
        ood_cand = p.with_name(p.name.replace(".jsonl", "_ood.jsonl"))
        ood_path = ood_cand if ood_cand.exists() else None

    if id_path is not None:
        ids = load_split(id_path)
        sub = stratified_subset(ids, lambda s: (s["task"], tier_of(s)), per_group, seed)
        for s in sub:
            s["_split"] = "ID"
            s["_style"] = assign_style(s)
        out.extend(sub)
    if ood_path is not None:
        oods = load_split(ood_path)
        sub = stratified_subset(oods, lambda s: s["metadata"].get("ood_category"), per_group, seed)
        for s in sub:
            s["_split"] = "OOD"
            s["_style"] = assign_style(s)
        out.extend(sub)
    return out


# ─────────────────────────── metrics ───────────────────────────
def per_prompt_metrics(exacts, exacts_full, rewards, canons, rng):
    """exacts/exacts_full/rewards: lists over draws. canons: list of str.
    exact       = PRIMARY structural correctness (NE: pure+support; BR: full).
    exact_full  = additionally requires exact mixed probability vectors (NE)."""
    exacts = np.asarray(exacts, dtype=float)
    exacts_full = np.asarray(exacts_full, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    pass1 = float(exacts.mean())
    pass1_full = float(exacts_full.mean())
    pass1_graded = float(rewards.mean())
    oracle = float(exacts.max()) if len(exacts) else 0.0
    oracle_full = float(exacts_full.max()) if len(exacts_full) else 0.0
    # majority: modal canon (random tiebreak), mean exact over draws with that canon
    cnt = Counter(canons)
    top = max(cnt.values())
    cands = sorted([c for c, v in cnt.items() if v == top])
    pick = cands[rng.integers(len(cands))]
    maj = float(np.mean([e for e, c in zip(exacts, canons) if c == pick]))
    return {"pass1": pass1, "pass1_full": pass1_full, "pass1_graded": pass1_graded,
            "oracle": oracle, "oracle_full": oracle_full, "majority": maj}


def aggregate(records):
    """records: list of dicts with keys split, task, tier, ood_category, the per-prompt
    metrics (pass1, majority, oracle, pass1_graded, ...) and truncation diagnostics
    (trunc_frac, no_answer_frac, length_frac, mean_gen_toks).
    Returns nested summary: by (split,task,tier), by split overall, by ood_category."""
    keys4 = ("pass1", "pass1_full", "majority", "oracle", "oracle_full", "pass1_graded",
             "trunc_frac", "no_answer_frac", "length_frac")   # averaged, rounded to 4 dp

    def agg(rows):
        if not rows:
            return None
        m = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys4}
        return {"n_prompts": len(rows), **m,
                "mean_gen_toks": round(float(np.mean([r["mean_gen_toks"] for r in rows])), 1)}

    summary = {"by_group": {}, "by_split": {}, "by_ood_category": {}}
    # by split x task x tier
    grp = defaultdict(list)
    for r in records:
        grp[(r["split"], r["task"], r["tier"])].append(r)
    for (sp, tk, ti), rows in grp.items():
        summary["by_group"].setdefault(sp, {}).setdefault(tk, {})[ti] = agg(rows)
    # by split overall + split x task
    for sp in sorted(set(r["split"] for r in records)):
        rows = [r for r in records if r["split"] == sp]
        summary["by_split"][sp] = {"overall": agg(rows)}
        for tk in sorted(set(r["task"] for r in rows)):
            summary["by_split"][sp][tk] = agg([r for r in rows if r["task"] == tk])
    # OOD by category
    for cat in sorted(set(r.get("ood_category") for r in records if r.get("ood_category"))):
        summary["by_ood_category"][cat] = agg([r for r in records if r.get("ood_category") == cat])
    return summary


# ─────────────────────────── main ───────────────────────────
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bench_path", required=True,
                    help="ID jsonl (sibling *_ood.jsonl auto-loaded) or an *_ood.jsonl")
    ap.add_argument("--out", required=True, help="per-model results JSON path")
    ap.add_argument("--gpu", required=True, help="GPU index to pin (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--per_group", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=10240)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="pass enable_thinking to the chat template (Qwen3 thinking mode)")
    ap.add_argument("--gold_check", action="store_true",
                    help="also score dataset chain_of_thought (gold) to sanity-check the scorer")
    ap.add_argument("--save_traces", action=argparse.BooleanOptionalAction, default=True,
                    help="save raw model responses per prompt/draw into per_prompt (default: on)")
    return ap


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from vllm import LLM, SamplingParams  # import AFTER pinning GPU

    model_name = Path(args.model_path).name
    samples = build_prompt_set(args.bench_path, args.per_group, args.seed)
    n_id = sum(1 for s in samples if s["_split"] == "ID")
    n_ood = sum(1 for s in samples if s["_split"] == "OOD")
    print(f"[{model_name}] gpu={args.gpu} prompts: ID={n_id} OOD={n_ood} total={len(samples)} n={args.n}", flush=True)

    # optional gold sanity check (no GPU needed)
    if args.gold_check:
        e = ef = tot = 0
        for s in samples:
            cot = s.get("chain_of_thought", "")
            if not cot:
                continue
            bd = R.reward_breakdown(cot, s["ground_truth"], s["task"], s["row_labels"], s["col_labels"])
            e += 1 if bd["exact"] else 0
            ef += 1 if bd["exact_full"] else 0
            tot += 1
        print(f"[{model_name}] GOLD-CHECK on chain_of_thought: exact={e/max(tot,1):.3f} "
              f"exact_full={ef/max(tot,1):.3f} (n={tot})", flush=True)

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              dtype="bfloat16", max_model_len=args.max_model_len, trust_remote_code=True, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, seed=args.seed)

    prompts = []
    for s in samples:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_prompt(s, s["_style"])}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=args.enable_thinking))

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[{model_name}] generated {len(prompts)}x{args.n} in {dt:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    records = []
    per_prompt = []
    for s, o in zip(samples, outs):
        exacts, exacts_full, rewards, canons, responses = [], [], [], [], []
        gen_toks = []
        trunc = no_answer = length_hit = 0
        for comp in o.outputs:
            resp = comp.text
            responses.append(resp)
            bd = R.reward_breakdown(resp, s["ground_truth"], s["task"], s["row_labels"], s["col_labels"])
            exacts.append(1.0 if bd["exact"] else 0.0)
            exacts_full.append(1.0 if bd["exact_full"] else 0.0)
            rewards.append(bd["reward"])
            canons.append(canon_answer(resp, s))
            # truncation diagnostics (defs match eval_think_diag.py; no </think> here — non-thinking run)
            gen_toks.append(len(comp.token_ids))
            has_ans = "ANSWER:" in resp.upper()
            hit_len = (comp.finish_reason == "length")
            no_answer += 0 if has_ans else 1
            length_hit += 1 if hit_len else 0
            trunc += 1 if (hit_len or not has_ans) else 0
        m = per_prompt_metrics(exacts, exacts_full, rewards, canons, rng)
        n = len(o.outputs)
        diag = {"trunc_frac": trunc / n, "no_answer_frac": no_answer / n,
                "length_frac": length_hit / n,
                "mean_gen_toks": float(np.mean(gen_toks)), "max_gen_toks": int(max(gen_toks))}
        rec = {"id": s["id"], "split": s["_split"], "task": s["task"], "tier": tier_of(s),
               "ood_category": s.get("metadata", {}).get("ood_category"), **m, **diag}
        records.append(rec)
        pp = {**rec, "style": s["_style"],
              "exacts": exacts, "exacts_full": exacts_full,
              "rewards": [round(r, 4) for r in rewards]}
        if args.save_traces:
            pp["responses"] = responses
        per_prompt.append(pp)

    summary = aggregate(records)
    result = {
        "model": model_name, "model_path": args.model_path,
        "config": {"n": args.n, "per_group": args.per_group, "seed": args.seed,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
                   "gpu_mem": args.gpu_mem, "tp": args.tp, "enable_thinking": args.enable_thinking},
        "n_prompts": {"ID": n_id, "OOD": n_ood, "total": len(samples)},
        "gen_seconds": round(dt, 1),
        "summary": summary,
        "per_prompt": per_prompt,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{model_name}] wrote {args.out}", flush=True)

    # compact console summary
    for sp_ in sorted(summary["by_split"].keys()):
        ov = summary["by_split"][sp_]["overall"]
        print(f"[{model_name}] {sp_} overall: pass@1={ov['pass1']:.3f} full={ov['pass1_full']:.3f} "
              f"maj={ov['majority']:.3f} oracle={ov['oracle']:.3f} oracle_full={ov['oracle_full']:.3f} "
              f"graded={ov['pass1_graded']:.3f} | trunc={ov['trunc_frac']:.2f} no_ans={ov['no_answer_frac']:.2f} "
              f"len_hit={ov['length_frac']:.2f} mean_toks={ov['mean_gen_toks']:.0f} (n={ov['n_prompts']})", flush=True)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
