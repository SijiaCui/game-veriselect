"""
THINKING diagnostic for NE-medium/hard/expert: does enabling Qwen3 thinking
un-floor the Nash-equilibrium tiers that were 0 pass@1 AND 0 oracle@8 non-thinking?

Reuses the EXACT same NE prompt ids as the non-thinking run (eval_core's deterministic
stratified subset, seed 42, per_group=40), restricted to task==nash_equilibrium and
tiers {medium,hard,expert}, taking the first --per_tier of each tier's block (a subset
of the non-thinking ids -> apples-to-apples).

THINKING ON: apply_chat_template(enable_thinking=True); unified sampling (T=0.6,
top_p=0.95, no top_k), big token budget (max_tokens=16384, max_model_len=18432).

Reports per (model x NE-tier): pass@1 (exact structural), pass@1 (exact_full), oracle@8
(exact), plus TRUNCATION diagnostics:
  - trunc_frac      : draws that hit max_tokens (finish_reason=="length") OR have no ANSWER block
  - no_answer_frac  : draws with no parseable "ANSWER:" block
  - length_frac     : draws that hit the token cap
  - no_thinkclose   : draws where </think> never appeared
  - mean_gen_toks   : mean generated tokens/draw (how close to the cap)
Saves NE-hard raw traces for a few prompts for the qualitative indifference-condition note.
"""
import os
import sys
import json
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = os.environ.get("GAMESOLVE_BENCH") or str(HERE.parents[1] / "bench" / "gamesolve")
EVAL_DIR = str(HERE)
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, EVAL_DIR)
import reward as R
import eval_core as E  # reuse SYSTEM_PROMPT, format_prompt, canon_answer, build_prompt_set, tier_of
import numpy as np

ID_BENCH = f"{BENCH_DIR}/gamesolve_hard.jsonl"
NE_TIERS = ["medium", "hard", "expert"]


def ne_subset(per_tier, seed):
    """Same ids as the non-thinking run: build the per_group=40 subset, keep NE tiers,
    take first per_tier of each tier (deterministic order preserved)."""
    full = E.build_prompt_set(ID_BENCH, 40, seed)  # sets _split, _style; deterministic order
    by_tier = defaultdict(list)
    for s in full:
        if s["_split"] == "ID" and s["task"] == "nash_equilibrium" and E.tier_of(s) in NE_TIERS:
            by_tier[E.tier_of(s)].append(s)
    picked = []
    for t in NE_TIERS:
        picked.extend(by_tier[t][:per_tier])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_tier", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--max_model_len", type=int, default=18432)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from vllm import LLM, SamplingParams

    model_name = Path(args.model_path).name
    samples = ne_subset(args.per_tier, args.seed)
    print(f"[{model_name}] gpu={args.gpu} NE prompts={len(samples)} "
          f"({args.per_tier}/tier x {len(NE_TIERS)}) n={args.n} thinking=ON "
          f"max_tokens={args.max_tokens}", flush=True)

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              dtype="bfloat16", max_model_len=args.max_model_len, trust_remote_code=True, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    prompts = []
    for s in samples:
        msgs = [{"role": "system", "content": E.SYSTEM_PROMPT},
                {"role": "user", "content": E.format_prompt(s, s["_style"])}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True))

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[{model_name}] generated {len(prompts)}x{args.n} thinking traces in {dt:.1f}s", flush=True)

    records = []
    raw_examples = []  # NE-hard example traces for qualitative note
    for s, o in zip(samples, outs):
        tier = E.tier_of(s)
        exacts, exacts_full, canons = [], [], []
        trunc = no_answer = length_hit = no_close = 0
        gen_toks = []
        for comp in o.outputs:
            resp = comp.text
            bd = R.reward_breakdown(resp, s["ground_truth"], s["task"], s["row_labels"], s["col_labels"])
            exacts.append(1.0 if bd["exact"] else 0.0)
            exacts_full.append(1.0 if bd["exact_full"] else 0.0)
            canons.append(E.canon_answer(resp, s))
            has_ans = "ANSWER:" in resp.upper()
            hit_len = (comp.finish_reason == "length")
            no_answer += 0 if has_ans else 1
            length_hit += 1 if hit_len else 0
            no_close += 0 if ("</think>" in resp) else 1
            trunc += 1 if (hit_len or not has_ans) else 0
            gen_toks.append(len(comp.token_ids))
        n = len(o.outputs)
        rec = {"id": s["id"], "tier": tier,
               "eq_class": s["ground_truth"]["equilibrium_class"], "dims": s["dimensions"],
               "pass1": float(np.mean(exacts)), "pass1_full": float(np.mean(exacts_full)),
               "oracle": float(max(exacts)), "oracle_full": float(max(exacts_full)),
               "trunc_frac": trunc / n, "no_answer_frac": no_answer / n,
               "length_frac": length_hit / n, "no_thinkclose_frac": no_close / n,
               "mean_gen_toks": float(np.mean(gen_toks)), "max_gen_toks": int(max(gen_toks))}
        records.append(rec)
        if tier == "hard" and len(raw_examples) < 3:
            raw_examples.append({"id": s["id"], "dims": s["dimensions"],
                                 "eq_class": s["ground_truth"]["equilibrium_class"],
                                 "gt": s["ground_truth"]["equilibria"],
                                 "draw0_finish": o.outputs[0].finish_reason,
                                 "draw0_resp": o.outputs[0].text[:6000]})

    # aggregate by tier
    def agg(rows):
        keys = ["pass1", "pass1_full", "oracle", "oracle_full", "trunc_frac",
                "no_answer_frac", "length_frac", "no_thinkclose_frac", "mean_gen_toks"]
        return {"n_prompts": len(rows), **{k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}}
    summary = {t: agg([r for r in records if r["tier"] == t]) for t in NE_TIERS}

    result = {"model": model_name, "mode": "thinking",
              "config": {"n": args.n, "per_tier": args.per_tier, "seed": args.seed,
                         "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
                         "temperature": args.temperature, "top_p": args.top_p},
              "gen_seconds": round(dt, 1), "summary": summary,
              "per_prompt": records, "raw_examples": raw_examples}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[{model_name}] wrote {args.out}", flush=True)
    for t in NE_TIERS:
        a = summary[t]
        print(f"[{model_name}] NE-{t}: pass@1={a['pass1']:.3f} full={a['pass1_full']:.3f} "
              f"oracle@8={a['oracle']:.3f} | trunc={a['trunc_frac']:.2f} no_ans={a['no_answer_frac']:.2f} "
              f"len_hit={a['length_frac']:.2f} mean_toks={a['mean_gen_toks']:.0f} (n={a['n_prompts']})", flush=True)


if __name__ == "__main__":
    main()
