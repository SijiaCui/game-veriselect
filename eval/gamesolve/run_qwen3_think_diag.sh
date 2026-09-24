#!/usr/bin/env bash
# Qwen3 THINKING diagnostic on NE medium/hard/expert, ONE MODEL AT A TIME, TP=4 on GPU0-3.
# Does enabling thinking un-floor the NE tiers that were 0 non-thinking? vLLM runs
# in-process and frees the GPUs on exit — no PID tracking, no pkill.
# Sampling + token budget (16384/18432) come from eval_think_diag.py defaults.
set -u
# Paths derived from THIS script's location. Env overrides: MODEL_ROOT (required), PY.
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
PY=${PY:-python3}
MDIR=$MODEL_ROOT
OUT=$HERE/../results/gamesolve/thinking_diag
LOG=$OUT/logs
DIR=$HERE
mkdir -p "$OUT" "$LOG"
PER=24; N=8; SEED=42

MODELS=(Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B)

for m in "${MODELS[@]}"; do
  echo "[$(date +%T)] ===== START $m (TP=4 on GPU0-3) ====="
  "$PY" "$DIR/eval_think_diag.py" --model_path "$MDIR/$m" --gpu 0,1,2,3 --tp 4 \
      --out "$OUT/$m.json" --per_tier $PER --n $N --seed $SEED \
      > "$LOG/$m.log" 2>&1 \
    && echo "[$(date +%T)] DONE $m" \
    || echo "[$(date +%T)] FAIL $m rc=$? (see $LOG/$m.log)"
done
echo "[$(date +%T)] ===== THINKING DIAG DONE ====="
