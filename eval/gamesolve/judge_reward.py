"""
LLM-as-judge holistic process scorer for GameSolve-Hard (gold-free baseline).

The paper's "claim-level" verifier vs a HOLISTIC judge: this scores each candidate's
whole trace with a judge LLM that sees ONLY the game problem + the candidate response
(never ground_truth -> gold-free), then selects best-of-N by judge score. Directly
comparable to the stored pass1/oracle/majority and to veriselect_gf.

Reads a per-model results JSON (from eval_core.run: per_prompt[].responses/.exacts/.style)
and writes back IN-PLACE (atomic):
  per_prompt[i]["judge_rewards"] = [score per candidate, 0..scale]
  per_prompt[i]["judge"]         = exacts[argmax judge_rewards]   (best-of-N correctness)
  top-level  ["judge_config"], ["judge_summary"] = {ID/OOD: judge@N vs pass@1 vs oracle}
Selection tiebreak: random among argmax (seed from result config), matching veriselect_at8.py.

Usage (one judge model, many result files -> load the judge once):
  python3 eval/gamesolve/judge_reward.py --gpu 0 --tp 1 \
      --judge_model_path /path/to/JudgeModel \
      --result eval/results/gamesolve/Qwen2.5-1.5B-Instruct.json [more.json ...]
Re-running RESUMES (skips prompts already judged) unless --overwrite forces a redo.
Progress is checkpointed to disk every --save_every prompts (default 50), so a crash
mid-run loses at most that many prompts; just re-run the same command to continue.
Parser self-check (no GPU):  python3 eval/gamesolve/judge_reward.py --selfcheck
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_core import format_prompt  # reuse the exact problem text the solver saw (no ground_truth)

BENCH_DIR = os.environ.get("GAMESOLVE_BENCH") or str(
    Path(__file__).resolve().parents[2] / "bench" / "gamesolve"
)
RESP_CAP = 16000   # ponytail: char cap on a candidate trace in the judge prompt; token-based if it bites
DEFAULT_SCALE = 10


def judge_messages(problem, response, scale):
    half = scale // 2
    system = (
        "You are a strict game-theory grader. You are given a game-theory PROBLEM and ONE "
        "candidate SOLUTION (its reasoning and final answer). You are NOT given the correct "
        "answer -- work it out yourself from the problem, then judge the candidate.\n\n"
        f"Rate how CORRECT and well-justified the candidate's reasoning and final answer are, "
        f"as an integer from 0 to {scale}:\n"
        f"  0  = reasoning absent, irrelevant, or clearly wrong; wrong answer.\n"
        f"  {half} = partially correct: some steps right but a material error or an "
        f"unjustified/incorrect final answer.\n"
        f"  {scale} = every step correct and the final answer right and fully justified.\n"
        "Judge only game-theoretic correctness of the process and answer, not style or length.\n"
        f'Think briefly, then end with exactly one line: "SCORE: <integer 0-{scale}>".'
    )
    if len(response) > RESP_CAP:
        keep = RESP_CAP // 2                       # keep head AND tail so the final ANSWER survives
        response = response[:keep] + "\n...[truncated]...\n" + response[-keep:]
    user = (f"PROBLEM:\n{problem}\n\nCANDIDATE SOLUTION:\n{response}\n\n"
            f'Now grade it. End with "SCORE: <integer 0-{scale}>".')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_score(text, scale):
    """Score from the final 'SCORE: <int>' line (the judge is told to end with one). Rejects
    range/fraction echoes of the rubric like 'SCORE: 0-10' or '10/10'. Unparseable -> None
    (caller maps to 0.0 for selection but counts it, so a truncated judge is visible)."""
    text = text or ""
    pat = r"SCORE:\s*(\d+)(?!\s*[-/]\s*\d)(?!\d)"   # a bare integer, not a "0-10"/"10/10" echo
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        m = re.search(pat, line, re.I)
        if m:
            return max(0.0, min(float(scale), float(m.group(1))))
    ms = re.findall(pat, text, re.I)
    return max(0.0, min(float(scale), float(ms[-1]))) if ms else None


def select(scores, rng):
    mx = max(scores)
    idx = [i for i, v in enumerate(scores) if v == mx]
    return idx[rng.integers(len(idx))]


def load_games():
    by_id = {}
    for fn in ("gamesolve_hard.jsonl", "gamesolve_hard_ood.jsonl"):
        p = os.path.join(BENCH_DIR, fn)
        if os.path.exists(p):
            for line in open(p):
                s = json.loads(line)
                by_id[s["id"]] = s
    return by_id


def write_atomic(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)   # atomic: a crash mid-write can't corrupt the input


def _pp_rng(seed, pid):
    """Per-prompt tie-break RNG: deterministic and independent of run/resume/chunk order."""
    return np.random.default_rng(int(hashlib.md5(f"{seed}|{pid}".encode()).hexdigest()[:8], 16))


def judge_summary(d, tag):
    """<tag>@N vs pass@1 vs oracle per split, over prompts that already have a <tag> score."""
    def agg(split):
        rows = [pp for pp in d["per_prompt"] if pp["split"] == split and tag in pp]
        if not rows:
            return None
        return {"n": len(rows),
                "judge": round(float(np.mean([pp[tag] for pp in rows])), 4),
                "pass1": round(float(np.mean([pp["pass1"] for pp in rows])), 4),
                "oracle": round(float(np.mean([pp["oracle"] for pp in rows])), 4)}
    return {s: agg(s) for s in ("ID", "OOD")}


def score_file(path, llm, tok, sp, games, scale, overwrite, limit, save_every, tag):
    """Add <tag>_rewards/<tag> to one result JSON, checkpointing every `save_every` prompts so a
    crash loses at most that many. Resumable: prompts already carrying <tag>_rewards are skipped
    unless overwrite. `tag` namespaces the fields so multiple judges coexist. Returns
    (n_scored_prompts, n_calls, summary)."""
    d = json.load(open(path))
    seed = d.get("config", {}).get("seed", 42)
    enable_thinking = sp["_enable_thinking"]
    rw_key = f"{tag}_rewards"

    pending = []
    for pi, pp in enumerate(d["per_prompt"]):
        if not pp.get("responses") or pp["id"] not in games:
            continue
        if len(pp.get("exacts", [])) != len(pp["responses"]):
            continue                              # malformed record: can't align selection to exacts
        if rw_key in pp and not overwrite:
            continue
        pending.append(pi)
        if limit and len(pending) >= limit:
            break

    if not pending:
        return 0, 0, judge_summary(d, tag)   # nothing to do; report from existing scores, no rewrite

    from vllm import SamplingParams
    params = SamplingParams(n=1, temperature=0.0, max_tokens=sp["max_new_tokens"], seed=seed)

    def checkpoint():
        d[f"{tag}_config"] = {"judge_model": sp["judge_model"], "scale": scale,
                              "max_new_tokens": sp["max_new_tokens"],
                              "enable_thinking": enable_thinking, "temperature": 0.0}
        d[f"{tag}_summary"] = judge_summary(d, tag)
        write_atomic(path, d)

    ncalls = nfail = 0
    for c in range(0, len(pending), save_every):
        chunk = pending[c:c + save_every]
        todo = []          # (pp_idx, cand_idx, prompt_text)
        for pi in chunk:
            pp = d["per_prompt"][pi]
            s = games[pp["id"]]
            problem = format_prompt(s, pp.get("style") or sorted(s["descriptions"])[0])
            for ci, resp in enumerate(pp["responses"]):
                msgs = judge_messages(problem, resp, scale)
                todo.append((pi, ci, tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking)))
        outs = llm.generate([t for _, _, t in todo], params)
        ncalls += len(todo)
        per = {}
        for (pi, ci, _), o in zip(todo, outs):
            sc = parse_score(o.outputs[0].text, scale)
            nfail += sc is None
            per.setdefault(pi, {})[ci] = 0.0 if sc is None else sc
        for pi in chunk:
            pp = d["per_prompt"][pi]
            jr = [per.get(pi, {}).get(ci, 0.0) for ci in range(len(pp["responses"]))]
            pp[rw_key] = jr
            pp[tag] = float(pp["exacts"][select(jr, _pp_rng(seed, pp["id"]))])
        checkpoint()       # persist this chunk before starting the next
        print(f"  .. {min(c + save_every, len(pending))}/{len(pending)} prompts judged "
              f"(checkpointed)", flush=True)
    if nfail:
        print(f"  !! {nfail}/{ncalls} judge outputs had no parseable SCORE (scored 0.0) — "
              f"raise --max_new_tokens or drop --enable_thinking if this is high", flush=True)
    return len(pending), ncalls, d[f"{tag}_summary"]


def selfcheck():
    assert parse_score("blah\nSCORE: 7", 10) == 7.0
    assert parse_score("SCORE: 3\nrethink\nSCORE: 9", 10) == 9.0        # last line wins
    assert parse_score("SCORE: 42", 10) == 10.0                         # clamp high
    assert parse_score("SCORE: -1", 10) is None                        # no digit after -> no match
    assert parse_score("verdict SCORE: 0-10", 10) is None              # reject rubric range echo
    assert parse_score("SCORE: 8\nfinal: reconsider SCORE: 0-10", 10) == 8.0  # skip echo, take 8
    assert parse_score("SCORE:6", 10) == 6.0                           # no space
    assert parse_score("I think it is a 5 but won't commit", 10) is None
    assert parse_score("", 10) is None
    rng = np.random.default_rng(0)
    assert select([1, 3, 3, 2], rng) in (1, 2)                          # argmax set
    assert select([0, 0, 5], rng) == 2
    assert select([0, 0, 0], np.random.default_rng(0)) in (0, 1, 2)     # no signal -> random pick
    assert _pp_rng(42, "x").integers(1 << 30) == _pp_rng(42, "x").integers(1 << 30)  # resume-stable
    assert _pp_rng(42, "x").integers(1 << 30) != _pp_rng(42, "y").integers(1 << 30)
    print("judge_reward selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", nargs="+", help="per-model result JSON(s), edited in place")
    ap.add_argument("--judge_model_path")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (e.g. 0 or 0,1,2,3)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    ap.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--overwrite", action="store_true", help="re-judge prompts already scored")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: judge at most N prompts THIS run (0 = all; resume-friendly)")
    ap.add_argument("--save_every", type=int, default=50,
                    help="checkpoint to disk every N prompts (crash loses at most this many)")
    ap.add_argument("--tag", default="judge",
                    help="field-name prefix so multiple judges coexist (e.g. judge_llm72b, judge_self)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.result or not args.judge_model_path:
        ap.error("--result and --judge_model_path are required (or use --selfcheck)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from vllm import LLM  # after GPU pin
    games = load_games()
    llm = LLM(model=args.judge_model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=args.max_model_len, trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = {"max_new_tokens": args.max_new_tokens, "judge_model": Path(args.judge_model_path).name,
          "_enable_thinking": args.enable_thinking}

    for path in args.result:
        t0 = time.time()
        npp, ncalls, summ = score_file(path, llm, tok, sp, games, args.scale,
                                       args.overwrite, args.limit, args.save_every, args.tag)
        dt = time.time() - t0
        name = Path(path).name
        if ncalls == 0:
            print(f"[{name}] already judged for tag={args.tag} (use --overwrite to redo)", flush=True)
        for split, a in summ.items():
            if a:
                print(f"[{name}] {split}: {args.tag}@N={a['judge']:.3f} pass@1={a['pass1']:.3f} "
                      f"oracle={a['oracle']:.3f} (n={a['n']})", flush=True)
        print(f"[{name}] tag={args.tag} +{npp} prompts, {ncalls} judge calls, {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
