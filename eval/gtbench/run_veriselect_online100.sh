#!/usr/bin/env bash
# GATE on all 4 GPUs free, then run the 100-match ONLINE veriselect validation.
# 3 arms run in PARALLEL, one 7B server per GPU (single=GPU0, random@N=GPU1, veriselect@N=GPU2),
# so wall-clock ~= one n=N arm (~30 min) instead of 3x sequential.
#   single       : CoTAgent n=1               -> pass@1
#   random@N     : BaselineAgent selector=random,n=N
#   veriselect@N : BaselineAgent selector=veriselect,n=N  (gold-free process verifier)
# vs the built-in random_agent, all 7 games. Metric = (win+0.5*draw)/normal.
#
# SAFETY: only kills the vLLM PIDs THIS script started; never a blanket pkill vllm.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; EVAL="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
CFGDIR="$GTBENCH/gamingbench/configs/model_configs"; ACFGDIR="$GTBENCH/gamingbench/configs/agent_configs"
RES="$EVAL/results/gtbench_veriselect/${RESDIR:-online100}"; LOG="$RES/logs"; mkdir -p "$LOG" "$RES"

N=${N:-8}; NUM_MATCHES=${NUM_MATCHES:-100}; WORKERS=${WORKERS:-16}; MAXTOK=${MAXTOK:-2048}
FREE_MIB=${FREE_MIB:-8000}; POLL=${POLL:-30}; MAXWAIT=${MAXWAIT:-43200}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}
SERVED=q25_7b; DIR=Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# ---- 1. GATE: wait until ALL 4 GPUs are free (used < FREE_MIB) ----
echo "[$(date +%T)] GATE: waiting for GPU0-3 all < ${FREE_MIB} MiB used (poll ${POLL}s, max ${MAXWAIT}s)"
t0=$(date +%s)
while :; do
  free=1
  for g in 0 1 2 3; do
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt "$FREE_MIB" ] || free=0
  done
  [ "$free" = 1 ] && { echo "[$(date +%T)] GATE open: all 4 GPUs free."; break; }
  [ $(( $(date +%s) - t0 )) -ge "$MAXWAIT" ] && { echo "[$(date +%T)] GATE timeout after ${MAXWAIT}s; aborting."; exit 1; }
  sleep "$POLL"
done

# ---- 2. configs ----
cat > "$CFGDIR/${SERVED}-cot.yaml" <<YAML
llm_model_path: local/${SERVED}:nothink
max_tokens: ${MAXTOK}
timeout: 180
temperature: 0.7
model_type: LLMModel
nick_name: ${SERVED}-cot
YAML
cat > "$ACFGDIR/cot_single.yaml"                <<< $'agent_name: CoTAgent\nnum_generations: 1\nmajority_vote: False'
cat > "$ACFGDIR/baseline_random_n${N}.yaml"     <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: random\nseed: 0'
cat > "$ACFGDIR/baseline_veriselect_n${N}.yaml" <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: veriselect\nseed: 0'

PIDS=()
serve () {  # $1=gpu $2=port
  CUDA_VISIBLE_DEVICES=$1 nohup vllm serve "$MODEL_ROOT/$DIR" --served-model-name "$SERVED" --port "$2" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --max-model-len 16384 > "$LOG/serve_gpu$1.log" 2>&1 &
  local pid=$!; PIDS+=($pid)
  while ! curl -sf "http://127.0.0.1:$2/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "!! server gpu$1 died"; tail -20 "$LOG/serve_gpu$1.log"; return 1; }; sleep 5; done
  echo "[$(date +%T)] READY server gpu$1 port$2 (pid $pid)"
}
cleanup () { for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; sleep 4; for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done; }
trap cleanup EXIT

serve 0 8041 || exit 1
serve 1 8042 || exit 1
serve 2 8043 || exit 1

run_arm () {  # $1=label $2=agent-cfg $3=port
  ( export LOCAL_LLM_ENDPOINTS="{\"$SERVED\":\"http://127.0.0.1:$3/v1\"}"
    cd "$GTBENCH" && python3 -m gamingbench.main --num-matches "$NUM_MATCHES" --exp-root "$RES/$1" \
      --seed 0 --game-names $GAMES \
      --agent-configs "gamingbench/configs/agent_configs/$2" gamingbench/configs/agent_configs/random_agent.yaml \
      --model-configs "gamingbench/configs/model_configs/${SERVED}-cot.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
      --exchange-first-player --num-workers "$WORKERS" --threshold-matches $((NUM_MATCHES + 10)) ) > "$LOG/arm_$1.log" 2>&1
  echo "[$(date +%T)] ARM $1 done rc=$?"
}

echo "==== ONLINE100 START $(date +%T) N=$N matches=$NUM_MATCHES (3 arms in parallel) ===="
run_arm single         cot_single.yaml               8041 & A1=$!
run_arm random${N}     baseline_random_n${N}.yaml     8042 & A2=$!
run_arm veriselect${N} baseline_veriselect_n${N}.yaml 8043 & A3=$!
wait "$A1" "$A2" "$A3"        # wait ONLY on the arms, not the (never-exiting) servers
echo "==== ONLINE100 DONE $(date +%T) ===="
python3 "$HERE/summarize_online.py" "$RES" "$GAMES"