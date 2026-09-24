#!/usr/bin/env bash
# GameSolve-Hard eval, ONE MODEL AT A TIME, TP=4 on GPU0-3 (in-process vLLM, frees GPUs on
# exit — no PID tracking, no pkill). Three passes, cheapest first:
#   1. Qwen2.5  nothink  -> Qwen2.5-*.json         (eval_qwen25.py, 8192/10240 budget)
#   2. Qwen3    nothink  -> Qwen3-*.json           (eval_qwen3.py,  16384/18432 budget)
#   3. Qwen3    think    -> Qwen3-*-think.json      (eval_qwen3.py --enable_thinking)  <- slowest
#
# TP must divide the model's attention-head count: TP=4 everywhere EXCEPT models whose heads
# aren't divisible by 4 (Qwen2.5-0.5B, 14 heads) -> TP=2 on GPU0,1. Add such models below.
# Env overrides: PER (per_group, default 40), N (draws, default 8), SEED (42).
set -u
# Everything below is derived from THIS script's location — no absolute paths.
# Env overrides: PER (per_group, 40), N (draws, 8), SEED (42),
#                MODEL_ROOT (model weights dir; required), PY (interpreter).
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
PY=${PY:-python3}
MDIR=$MODEL_ROOT
BENCH=$REPO/bench/gamesolve/gamesolve_hard.jsonl
OUT=$HERE/../results/gamesolve
LOG=$OUT/logs
DIR=$HERE
mkdir -p "$OUT" "$LOG"
PER=${PER:-40}; N=${N:-8}; SEED=${SEED:-42}

QWEN25=(Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct \
        Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct)
QWEN3=(Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B)

# build the run list: "model|entry|out_name|extra_args"
RUNS=()
for m in "${QWEN25[@]}"; do RUNS+=("$m|eval_qwen25.py|$m|"); done
for m in "${QWEN3[@]}";  do RUNS+=("$m|eval_qwen3.py|$m|"); done
for m in "${QWEN3[@]}";  do RUNS+=("$m|eval_qwen3.py|$m-think|--enable_thinking"); done

echo "[$(date +%T)] ===== full eval: ${#RUNS[@]} runs (PER=$PER N=$N SEED=$SEED) ====="
for spec in "${RUNS[@]}"; do
  IFS='|' read -r m entry out extra <<< "$spec"
  case "$m" in
    Qwen2.5-0.5B-Instruct) tp=2; gpus=0,1 ;;   # 14 heads, not divisible by 4
    *)                     tp=4; gpus=0,1,2,3 ;;
  esac
  echo "[$(date +%T)] ===== START $out ($entry $extra, TP=$tp on GPU$gpus) ====="
  "$PY" "$DIR/$entry" --model_path "$MDIR/$m" --bench_path "$BENCH" \
      --out "$OUT/$out.json" --gpu "$gpus" --tp $tp --n $N --per_group $PER --seed $SEED $extra \
      > "$LOG/$out.log" 2>&1 \
    && echo "[$(date +%T)] DONE $out" \
    || echo "[$(date +%T)] FAIL $out rc=$? (see $LOG/$out.log)"
done
echo "[$(date +%T)] ===== ALL DONE ====="
