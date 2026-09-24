#!/bin/bash
# Serve the 6 Qwen3 chat models as local vLLM OpenAI-compatible servers, all on
# GPU2 + GPU3 (the only two cards allowed for this experiment). Every model fits
# comfortably; we co-locate 3 per card, balanced by weight so KV cache is even.
#
#   CUDA_VISIBLE_DEVICES=2 : q32b(8001) q8b(8003) q1_7b(8005)
#   CUDA_VISIBLE_DEVICES=3 : q14b(8002) q4b(8004) q0_6b(8006)
#
# Thinking on/off is a per-REQUEST parameter (chat_template_kwargs.enable_thinking),
# so a single server per model covers both decode modes and both benchmarks.
# Servers on the same card are launched sequentially with a health-gate so each
# vLLM profiles free memory only after the previous one has claimed its share.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Model weights live OUTSIDE the repo. Override with MODEL_ROOT=<dir> in the env.
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
LOG="$HERE/logs"; mkdir -p "$LOG"
MAXLEN=16384
PIDFILE="$LOG/server_pids.txt"
: > "$PIDFILE"   # fresh record of the PIDs we start, for precise teardown (eval/stop.sh)

declare -A DIR=(
  [q0_6b]=Qwen3-0.6B [q1_7b]=Qwen3-1.7B [q4b]=Qwen3-4B
  [q8b]=Qwen3-8B     [q14b]=Qwen3-14B   [q32b]=Qwen3-32B
)

# name gpu port util
launch () {
  local name=$1 gpu=$2 port=$3 util=$4
  echo ">> launching $name on GPU$gpu port $port (util $util)"
  CUDA_VISIBLE_DEVICES=$gpu nohup vllm serve "$MODEL_ROOT/${DIR[$name]}" \
    --served-model-name "$name" --port "$port" \
    --gpu-memory-utilization "$util" --max-model-len $MAXLEN \
    > "$LOG/serve_${name}_${port}.log" 2>&1 &
  local pid=$!
  # health-gate before the next co-located server profiles memory
  while ! curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "!! $name (pid $pid) died before becoming healthy; last log lines:"
      tail -15 "$LOG/serve_${name}_${port}.log"
      exit 1
    fi
    sleep 5
  done
  echo "$pid $name" >> "$PIDFILE"
  echo "   $name ready on $port (pid $pid)"
}

# GPU2: heavy + medium + tiny
launch q32b  2 8001 0.45
launch q8b   2 8003 0.25
launch q1_7b 2 8005 0.15
# GPU3: medium + small + tiny
launch q14b  3 8002 0.35
launch q4b   3 8004 0.25
launch q0_6b 3 8006 0.15

echo "waiting for all 6 servers..."
for p in 8001 8002 8003 8004 8005 8006; do
  until curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1; do sleep 5; done
  echo "port $p ready"
done
echo "ALL 6 SERVERS READY"
# Stay alive so the servers (our child processes) keep running. To release the
# GPUs, run `bash eval/stop.sh` (kills ONLY these PIDs, from logs/server_pids.txt)
# — never a broad `pkill vllm`, which could hit other tasks' servers.
wait

