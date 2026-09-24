#!/bin/bash
# Per-model serving: serve ONE model with tensor parallelism across BOTH allowed
# cards (GPU2 + GPU3) at high utilization. This is the go-forward pattern —
# each model gets the full available compute, models are run one at a time.
# (Contrast with serve.sh, which co-locates all 6 small servers for a single pass.)
#
# Usage: bash serve_one.sh <served_name>    e.g. bash serve_one.sh q32b
# Ports match serve.sh so run_gtbench.sh works unchanged.
set -u
SERVED=${1:?served name required (q0_6b|q1_7b|q4b|q8b|q14b|q32b)}
HERE="$(cd "$(dirname "$0")" && pwd)"
# Model weights live OUTSIDE the repo. Override with MODEL_ROOT=<dir> in the env.
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
LOG="$HERE/logs"; mkdir -p "$LOG"
PIDFILE="$LOG/server_pids.txt"
GPUS=2,3          # the two allowed cards
MAXLEN=32768      # ample context (TP=2 gives plenty of KV cache); avoids think-mode overflow

declare -A DIR=(
  [q0_6b]=Qwen3-0.6B [q1_7b]=Qwen3-1.7B [q4b]=Qwen3-4B
  [q8b]=Qwen3-8B     [q14b]=Qwen3-14B   [q32b]=Qwen3-32B
)
declare -A PORT=(
  [q32b]=8001 [q14b]=8002 [q8b]=8003 [q4b]=8004 [q1_7b]=8005 [q0_6b]=8006
)
[ -n "${DIR[$SERVED]:-}" ] || { echo "unknown model $SERVED"; exit 1; }
port=${PORT[$SERVED]}

echo ">> serving $SERVED (${DIR[$SERVED]}) TP=2 on GPU$GPUS, port $port, max-len $MAXLEN"
CUDA_VISIBLE_DEVICES=$GPUS nohup vllm serve "$MODEL_ROOT/${DIR[$SERVED]}" \
  --served-model-name "$SERVED" --port "$port" \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len $MAXLEN \
  > "$LOG/serve_${SERVED}_${port}.log" 2>&1 &
pid=$!

while ! curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "!! $SERVED (pid $pid) died before healthy; last log:"; tail -20 "$LOG/serve_${SERVED}_${port}.log"; exit 1
  fi
  sleep 5
done
# record PID for precise teardown (eval/stop.sh); append so multiple serve_one's coexist
grep -qs " $SERVED\$" "$PIDFILE" 2>/dev/null && sed -i "/ ${SERVED}\$/d" "$PIDFILE"
echo "$pid $SERVED" >> "$PIDFILE"
echo "READY: $SERVED on $port (pid $pid). Stop with: bash eval/stop.sh"
wait
