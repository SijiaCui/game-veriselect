"""
LLM-as-judge holistic process scorer for GTBench (gold-free baseline) -- the sequential-game
analog of eval/gamesolve/judge_reward.py.

For each move-state, score every candidate reasoning+move with a judge LLM that sees ONLY the
per-move prompt the player saw (query.messages: game rules + current position + legal moves)
plus ONE candidate response -- never the outcome or the opponent's hidden info, so it is
gold-free by construction. Writes back IN-PLACE into the JSONL, per query:
  query["judge_rewards"] = [score per candidate, 0..scale]      (parallel to llm_output)
The best-of-N SELECTION metric (judge@N vs pass@1/vsel@N/oracle@N) is computed offline by
analyze_veriselect_gtbench.py from these stored scores + its move-quality oracles.

Chunked + checkpointed per file (a crash loses <= --save_every states) + resumable
(queries already carrying judge_rewards are skipped unless --overwrite).

Usage:
  python3 eval/gtbench/judge_reward_gtbench.py --gpu 0 --tp 1 \
      --judge_model_path /path/to/JudgeModel \
      --root eval/results/gtbench_veriselect/q25_7b-cot-n8 [--games nim tictactoe] [--limit 20]
Self-check (no GPU):  python3 eval/gtbench/judge_reward_gtbench.py --selfcheck
"""
import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gamesolve"))
from judge_reward import parse_score   # reuse the tested "SCORE: <int>" parser (rejects range echoes)

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "results", "gtbench_veriselect", "q25_7b-cot-n8")
RESP_CAP = 16000
DEFAULT_SCALE = 10


def judge_messages_gt(problem, response, scale):
    half = scale // 2
    system = (
        "You are a strict game-playing grader. You are given the exact SITUATION shown to a "
        "player in a turn-based game (the rules, the current position, and the legal moves) and "
        "ONE candidate RESPONSE (the player's reasoning and chosen move). You do NOT know the "
        "outcome or the opponent's hidden information -- reason about the position yourself, then "
        "judge the candidate.\n\n"
        f"Rate how SOUND the candidate's reasoning and chosen move are for this position, as an "
        f"integer from 0 to {scale}:\n"
        f"  0  = reasoning absent/irrelevant/wrong, or an illegal or clearly losing move.\n"
        f"  {half} = partially sound: a reasonable move but flawed or incomplete reasoning.\n"
        f"  {scale} = correct, well-justified reasoning and a best (or clearly good) move.\n"
        "Judge only game-playing correctness, not style or length.\n"
        f'Think briefly, then end with exactly one line: "SCORE: <integer 0-{scale}>".'
    )
    if len(response) > RESP_CAP:
        keep = RESP_CAP // 2                       # keep head AND tail so the chosen move survives
        response = response[:keep] + "\n...[truncated]...\n" + response[-keep:]
    user = (f"SITUATION SHOWN TO THE PLAYER:\n{problem}\n\nCANDIDATE RESPONSE:\n{response}\n\n"
            f'Now grade it. End with "SCORE: <integer 0-{scale}>".')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_text(q, ci):
    """Full candidate response to score: raw_reasoning[ci] (pre-strip CoT) else llm_output[ci]."""
    outs = q["llm_output"]
    raws = q.get("raw_reasoning") or outs
    return (raws[ci] if ci < len(raws) and raws[ci] else outs[ci]) or ""


def problem_text(q):
    """The exact per-move prompt the player saw (gold-free: no outcome, no hidden info)."""
    return "\n\n".join(m.get("content", "") for m in q.get("messages", []))


def pending_queries(recs, overwrite):
    """First query of every move-state with candidates that isn't judged yet (in file order)."""
    out = []
    for rec in recs:
        for m in rec.get("matches", []):
            for st in m.get("steps", []):
                qs = st.get("queries")
                if not qs or not qs[0].get("llm_output"):
                    continue
                if "judge_rewards" in qs[0] and not overwrite:
                    continue
                out.append(qs[0])
    return out


def write_jsonl_atomic(path, recs):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)   # atomic: a crash mid-write can't corrupt the input file


def score_file(fpath, llm, tok, sp, scale, overwrite, max_states, save_every):
    """Add judge_rewards to each query in one game JSONL, checkpointing every `save_every`
    states. Returns (n_states_scored, n_calls, n_parse_failures)."""
    recs = [json.loads(l) for l in open(fpath) if l.strip()]
    pending = pending_queries(recs, overwrite)
    if max_states is not None:
        pending = pending[:max_states]
    if not pending:
        return 0, 0, 0

    from vllm import SamplingParams
    params = SamplingParams(n=1, temperature=0.0, max_tokens=sp["max_new_tokens"], seed=0)
    et = sp["_enable_thinking"]
    ncalls = nfail = 0
    for c in range(0, len(pending), save_every):
        chunk = pending[c:c + save_every]
        todo = []                          # (query_ref, cand_idx, prompt_text)
        for q in chunk:
            prob = problem_text(q)
            for ci in range(len(q["llm_output"])):
                msgs = judge_messages_gt(prob, candidate_text(q, ci), scale)
                todo.append((q, ci, tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=et)))
        outs = llm.generate([t for _, _, t in todo], params)
        ncalls += len(todo)
        buf = {}                           # id(query) -> {cand_idx: score}
        for (q, ci, _), o in zip(todo, outs):
            sc = parse_score(o.outputs[0].text, scale)
            nfail += sc is None
            buf.setdefault(id(q), {})[ci] = 0.0 if sc is None else sc
        for q in chunk:
            per = buf.get(id(q), {})
            q["judge_rewards"] = [per.get(ci, 0.0) for ci in range(len(q["llm_output"]))]
        write_jsonl_atomic(fpath, recs)    # persist this chunk before starting the next
    return len(pending), ncalls, nfail


def selfcheck():
    q = {"llm_output": ["OUT0", "OUT1"], "raw_reasoning": ["RAW0", None]}
    assert candidate_text(q, 0) == "RAW0"
    assert candidate_text(q, 1) == "OUT1"                       # None raw -> fall back to output
    assert candidate_text({"llm_output": ["A", "B"]}, 1) == "B"  # no raw_reasoning key
    assert "C2R2" in problem_text({"messages": [{"role": "user", "content": "play C2R2?"}]})
    recs = [{"matches": [{"steps": [
        {"queries": [{"llm_output": ["a"]}]},                             # pending
        {"queries": [{"llm_output": ["b"], "judge_rewards": [1.0]}]},     # already judged
        {"queries": [{"llm_output": []}]},                                # no candidates
        {"no_queries": 1},                                                # no queries
    ]}]}]
    assert len(pending_queries(recs, overwrite=False)) == 1
    assert len(pending_queries(recs, overwrite=True)) == 2      # overwrite re-includes the judged
    print("judge_reward_gtbench selfcheck OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--games", nargs="*", help="subset of game dirs (default: all under root)")
    ap.add_argument("--judge_model_path")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (e.g. 0 or 0,1,2,3)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    ap.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--overwrite", action="store_true", help="re-judge states already scored")
    ap.add_argument("--limit", type=int, default=0, help="smoke: judge at most N states THIS run")
    ap.add_argument("--save_every", type=int, default=50,
                    help="checkpoint to disk every N states (crash loses at most this many)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.judge_model_path:
        ap.error("--judge_model_path is required (or use --selfcheck)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from vllm import LLM  # after GPU pin
    llm = LLM(model=args.judge_model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=args.max_model_len, trust_remote_code=True)
    tok = llm.get_tokenizer()
    sp = {"max_new_tokens": args.max_new_tokens, "_enable_thinking": args.enable_thinking}

    games = args.games or sorted(d for d in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, d)))
    json.dump({"judge_model": Path(args.judge_model_path).name, "scale": args.scale,
               "max_new_tokens": args.max_new_tokens, "enable_thinking": args.enable_thinking},
              open(os.path.join(args.root, "judge_config.json"), "w"), indent=2)

    budget = args.limit or None
    for g in games:
        gdir = os.path.join(args.root, g)
        if not os.path.isdir(gdir):
            print(f"[{g}] missing, skip", flush=True)
            continue
        for fp in sorted(glob.glob(os.path.join(gdir, "*.jsonl"))):
            if budget is not None and budget <= 0:
                break
            t0 = time.time()
            npp, ncalls, nfail = score_file(fp, llm, tok, sp, args.scale, args.overwrite,
                                            budget, args.save_every)
            if budget is not None:
                budget -= npp
            tag = os.path.join(g, os.path.basename(fp))
            if ncalls:
                warn = f"  !! {nfail} unparseable SCORE" if nfail else ""
                print(f"[{tag}] +{npp} states, {ncalls} calls, {time.time() - t0:.1f}s{warn}",
                      flush=True)
            elif npp == 0:
                print(f"[{tag}] already judged", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
