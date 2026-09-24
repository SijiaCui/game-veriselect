#!/usr/bin/env bash
# GTBench for the 7 Qwen2.5-*B-Instruct models, default / non-thinking mode only
# (Qwen2.5 has no thinking mode), candidate vs the built-in random_agent.
#
# GPU PLAN (per user request):
#   * 72B  -> GPU2 + GPU3, tensor-parallel=2   (dedicated server, port 8021)
#   * other 6 (32B,14B,7B,3B,1.5B,0.5B) -> GPU1, served ONE AT A TIME,
#     largest -> smallest (single server re-served per model on port 8022)
#   The two tracks run CONCURRENTLY (disjoint GPUs).
#
# SAFETY:
#   * We only ever kill vLLM PIDs THIS script started (recorded in PIDFILE).
#     Never a blanket `pkill vllm` — other tasks may share the box.
#   * GATE: we do NOT launch any vLLM until the sibling gamesolve run has fully
#     finished (all 7 JSONs + "ALL DONE") AND GPU1/2/3 are released (<8 GB used),
#     because gamesolve's wave 2 briefly borrows GPU0/1/2.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
CFGDIR="$GTBENCH/gamingbench/configs/model_configs"
RES="$EVAL/results/gtbench"
LOG="$RES/logs_qwen25"; mkdir -p "$LOG" "$RES"
PIDFILE="$LOG/qwen25_server_pids.txt"; : > "$PIDFILE"

NUM_MATCHES=${NUM_MATCHES:-20}
WORKERS=${WORKERS:-8}
MAXLEN=${MAXLEN:-16384}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# served-name -> model dir  (nothink only)
declare -A DIR=(
  [q25_72b]=Qwen2.5-72B-Instruct [q25_32b]=Qwen2.5-32B-Instruct
  [q25_14b]=Qwen2.5-14B-Instruct [q25_7b]=Qwen2.5-7B-Instruct
  [q25_3b]=Qwen2.5-3B-Instruct   [q25_1_5b]=Qwen2.5-1.5B-Instruct
  [q25_0_5b]=Qwen2.5-0.5B-Instruct
)
PORT_72B=8021; PORT_GPU1=8022
# single source of truth: 72B on its own port, the other 6 share the GPU1 port.
export LOCAL_LLM_ENDPOINTS="{\"q25_72b\":\"http://127.0.0.1:$PORT_72B/v1\",\"q25_32b\":\"http://127.0.0.1:$PORT_GPU1/v1\",\"q25_14b\":\"http://127.0.0.1:$PORT_GPU1/v1\",\"q25_7b\":\"http://127.0.0.1:$PORT_GPU1/v1\",\"q25_3b\":\"http://127.0.0.1:$PORT_GPU1/v1\",\"q25_1_5b\":\"http://127.0.0.1:$PORT_GPU1/v1\",\"q25_0_5b\":\"http://127.0.0.1:$PORT_GPU1/v1\"}"

# ---- 1. write nothink model configs into the GTBench submodule ----
for name in "${!DIR[@]}"; do
  cat > "$CFGDIR/${name}-nothink.yaml" <<YAML
llm_model_path: local/${name}:nothink
max_tokens: 1024
timeout: 180
temperature: 0.7
model_type: LLMModel
nick_name: ${name}-nothink
YAML
done
echo "[$(date +%T)] wrote 7 Qwen2.5 nothink configs"

# ---- helpers ----
descendants () { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
stop_pid () {  # kill one recorded server PID + its children
  local pid=$1 name=$2 all
  kill -0 "$pid" 2>/dev/null || { echo "[$(date +%T)] $name (pid $pid) already gone"; return; }
  all="$pid $(descendants "$pid")"
  echo "[$(date +%T)] stopping $name: $all"
  kill -TERM $all 2>/dev/null; sleep 5
  for p in $all; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
}
serve () {  # $1=served $2=gpus $3=port $4=tp $5=util ; status->stderr, pid->stdout
  local name=$1 gpus=$2 port=$3 tp=$4 util=$5 pid
  echo "[$(date +%T)] SERVE $name (${DIR[$name]}) GPU=$gpus tp=$tp port=$port util=$util" >&2
  CUDA_VISIBLE_DEVICES=$gpus nohup vllm serve "$MODEL_ROOT/${DIR[$name]}" \
    --served-model-name "$name" --port "$port" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization "$util" --max-model-len $MAXLEN \
    > "$LOG/serve_${name}.log" 2>&1 &
  pid=$!
  echo "$pid $name" >> "$PIDFILE"
  while ! curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(date +%T)] !! $name died before healthy; tail:" >&2; tail -20 "$LOG/serve_${name}.log" >&2; return 1
    fi
    sleep 5
  done
  echo "[$(date +%T)] READY $name on $port (pid $pid)" >&2; echo "$pid"
}
run_gt () {  # $1=served
  local name=$1
  echo "[$(date +%T)] GTBENCH START $name"
  ( cd "$GTBENCH" && python3 -m gamingbench.main \
      --num-matches "$NUM_MATCHES" \
      --exp-root "$RES/${name}-nothink" \
      --seed 0 --game-names $GAMES \
      --agent-configs gamingbench/configs/agent_configs/prompt_agent.yaml gamingbench/configs/agent_configs/random_agent.yaml \
      --model-configs "gamingbench/configs/model_configs/${name}-nothink.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
      --exchange-first-player --num-workers "$WORKERS" \
      --threshold-matches $((NUM_MATCHES + 10)) ) > "$LOG/gt_${name}.log" 2>&1
  echo "[$(date +%T)] GTBENCH DONE $name rc=$?"
}

# ---- 2. GATE: wait for the sibling gamesolve run to finish & free GPUs ----
gate () {
  local gdir="$EVAL/results/gamesolve"
  echo "[$(date +%T)] GATE: waiting for gamesolve JSONs + GPU1/2/3 release..."
  while :; do
    local have=1 s
    for s in 0.5B 1.5B 3B 7B 14B 32B 72B; do
      [ -f "$gdir/Qwen2.5-${s}-Instruct.json" ] || have=0
    done
    # GPU1/2/3 free? (used MiB < 8000 on each)
    local free=1 u g
    for g in 1 2 3; do
      u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
      [ -n "$u" ] && [ "$u" -lt 8000 ] || free=0
    done
    if [ "$have" = 1 ] && [ "$free" = 1 ]; then
      echo "[$(date +%T)] GATE open: gamesolve done, GPU1/2/3 free."; break
    fi
    sleep 20
  done
}
gate

# ---- 3. run both tracks concurrently ----
track_72b () {   # GPU2+3, tp=2
  local pid
  pid=$(serve q25_72b 2,3 $PORT_72B 2 0.90) || { echo "72B serve failed"; return 1; }
  run_gt q25_72b
  stop_pid "$pid" q25_72b
}
track_gpu1 () {  # GPU1 single card, largest -> smallest, one at a time
  local name pid
  for name in q25_32b q25_14b q25_7b q25_3b q25_1_5b q25_0_5b; do
    pid=$(serve "$name" 1 $PORT_GPU1 1 0.90) || { echo "$name serve failed, skip"; continue; }
    run_gt "$name"
    stop_pid "$pid" "$name"
  done
}

echo "==== GTBENCH QWEN2.5 START $(date +%T)  matches=$NUM_MATCHES games=[$GAMES] ===="
track_72b  > "$LOG/track_72b.log" 2>&1 &
T72=$!
track_gpu1 > "$LOG/track_gpu1.log" 2>&1 &
TG1=$!
wait "$T72"; echo "[$(date +%T)] track_72b finished rc=$?"
wait "$TG1"; echo "[$(date +%T)] track_gpu1 finished rc=$?"
echo "==== ALL GTBENCH DONE $(date +%T) ===="
