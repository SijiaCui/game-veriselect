#!/usr/bin/env bash
# GameSolve judge@8 baselines for the Qwen2.5 series. Two arms, both = gold-free HOLISTIC
# LLM-judge best-of-8 selection (eval/gamesolve/judge_reward.py), namespaced by --tag so they
# coexist in each result JSON:
#   arm1 llm-judge : Qwen2.5-72B judges ALL 7 result files       -> fields "judge_llm72b*"
#   arm2 self-judge: each model judges its OWN file (72B skipped; -> fields "judge_self*"
#                    72B-self == judge_llm72b on the 72B file, aliased in summarize_judge.py)
# In-process vLLM, TP=4 on GPU0-3 (0.5B needs TP=2: 14 heads not /4). Checkpointed + resumable:
# re-run to continue; LIMIT=5 for a smoke; OVERWRITE=1 to redo a tag. No pkill (in-process only).
set -u
# Paths derived from THIS script's location. Env overrides: MODEL_ROOT (required), PY, LIMIT, OVERWRITE.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
PY=${PY:-python3}
MDIR=$MODEL_ROOT
RES=$HERE/../results/gamesolve
JR=$HERE/judge_reward.py
LOG=$RES/logs; mkdir -p "$LOG"
LIMIT=${LIMIT:-0}
OW=${OVERWRITE:+--overwrite}

GMEM=${GMEM:-0.90}; MML=${MML:-16384}   # MML 16384: judge prompt is problem + response(<=16000 chars) + wrapper
VLLM="--gpu_mem $GMEM --max_model_len $MML"

MODELS=(Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct \
        Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct)

tp_for(){ case "$1" in *0.5B*) echo "2 0,1";; *) echo "4 0,1,2,3";; esac; }   # 0.5B: 14 heads not /4

# ---- arm 1: llm-judge (Qwen2.5-72B over all files) ----
read jtp jgpu <<<"$(tp_for Qwen2.5-72B-Instruct)"
RESULTS=(); for m in "${MODELS[@]}"; do RESULTS+=("$RES/$m.json"); done
echo "[$(date +%T)] ===== ARM1 llm-judge: Qwen2.5-72B judges all ${#MODELS[@]} files (TP=$jtp) ====="
"$PY" "$JR" --tag judge_llm72b --judge_model_path "$MDIR/Qwen2.5-72B-Instruct" \
    --tp $jtp --gpu $jgpu $VLLM --limit $LIMIT $OW --result "${RESULTS[@]}" \
    > "$LOG/judge_llm72b.log" 2>&1 \
  && echo "[$(date +%T)] ARM1 DONE" || echo "[$(date +%T)] ARM1 FAIL rc=$? (see $LOG/judge_llm72b.log)"

# ---- arm 2: self-judge (each model on its own file; 72B skipped = dup of arm1) ----
echo "[$(date +%T)] ===== ARM2 self-judge: each model judges its own traces ====="
for m in "${MODELS[@]}"; do
  [ "$m" = "Qwen2.5-72B-Instruct" ] && { echo "[$(date +%T)] skip $m self (== judge_llm72b)"; continue; }
  read tp gpu <<<"$(tp_for "$m")"
  echo "[$(date +%T)] self-judge $m (TP=$tp GPU$gpu)"
  "$PY" "$JR" --tag judge_self --judge_model_path "$MDIR/$m" \
      --tp $tp --gpu $gpu $VLLM --limit $LIMIT $OW --result "$RES/$m.json" \
      > "$LOG/judge_self_$m.log" 2>&1 \
    && echo "[$(date +%T)] self $m DONE" || echo "[$(date +%T)] self $m FAIL rc=$? (see $LOG/judge_self_$m.log)"
done
echo "[$(date +%T)] ===== ALL DONE ====="
"$PY" "$HERE/summarize_judge.py" || true
