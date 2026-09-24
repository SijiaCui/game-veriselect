#!/usr/bin/env bash
# GTBench veriselect SCALING sweep over all 7 Qwen2.5-Instruct sizes (0.5B..72B), SERIAL.
# Each model is served ONCE with tensor parallelism across ALL 4 B200s (TP=4; TP=2 for
# 0.5B whose 14 attn heads aren't divisible by 4), and the 3 arms run CONCURRENTLY against
# that single endpoint (vLLM continuous-batches them) -> all 4 GPUs busy, lower latency.
#   single       : CoTAgent n=1                 -> pass@1
#   random8      : BaselineAgent selector=random -> random@8
#   veriselect8  : BaselineAgent selector=veriselect (fixed gold-free verifier) -> veriselect@8
# vs the built-in random_agent, all 7 games, 100 matches/arm/game. Resumable via per-model .done.
# SAFETY: only kills the vLLM PID THIS script started; per-model teardown + EXIT trap.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; EVAL="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
CFGDIR="$GTBENCH/gamingbench/configs/model_configs"; ACFGDIR="$GTBENCH/gamingbench/configs/agent_configs"
RES="$EVAL/results/gtbench_veriselect/scaling_q25"; LOG="$RES/logs"; mkdir -p "$LOG" "$RES"

N=${N:-8}; NUM_MATCHES=${NUM_MATCHES:-100}; WORKERS=${WORKERS:-24}; MAXTOK=${MAXTOK:-2048}
PORT=${PORT:-8041}; UTIL=${UTIL:-0.90}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# tag  dir  tensor-parallel-size  (0.5B -> TP=2; all others -> TP=4)
MODELS=(
  "q25_0_5b Qwen2.5-0.5B-Instruct 2"
  "q25_1_5b Qwen2.5-1.5B-Instruct 4"
  "q25_3b   Qwen2.5-3B-Instruct   4"
  "q25_7b   Qwen2.5-7B-Instruct   4"
  "q25_14b  Qwen2.5-14B-Instruct  4"
  "q25_32b  Qwen2.5-32B-Instruct  4"
  "q25_72b  Qwen2.5-72B-Instruct  4"
)

# agent configs are model-independent -> write once
cat > "$ACFGDIR/cot_single.yaml"                <<< $'agent_name: CoTAgent\nnum_generations: 1\nmajority_vote: False'
cat > "$ACFGDIR/baseline_random_n${N}.yaml"     <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: random\nseed: 0'
cat > "$ACFGDIR/baseline_veriselect_n${N}.yaml" <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: veriselect\nseed: 0'

SPID=""
cleanup () { [ -n "$SPID" ] && kill -KILL "$SPID" 2>/dev/null; }
trap cleanup EXIT

run_arm () {  # $1=served $2=label $3=agent-cfg
  ( export LOCAL_LLM_ENDPOINTS="{\"$1\":\"http://127.0.0.1:$PORT/v1\"}"
    cd "$GTBENCH" && python3 -m gamingbench.main --num-matches "$NUM_MATCHES" --exp-root "$RES/$1/$2" \
      --seed 0 --game-names $GAMES \
      --agent-configs "gamingbench/configs/agent_configs/$3" gamingbench/configs/agent_configs/random_agent.yaml \
      --model-configs "gamingbench/configs/model_configs/$1-cot.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
      --exchange-first-player --num-workers "$WORKERS" --threshold-matches $((NUM_MATCHES + 10)) ) > "$LOG/arm_${1}_$2.log" 2>&1
  echo "[$(date +%T)] ARM $1/$2 done rc=$?"
}

echo "==== SCALING START $(date +%T)  N=$N matches=$NUM_MATCHES workers=$WORKERS  models=${#MODELS[@]} ===="
for entry in "${MODELS[@]}"; do
  set -- $entry; SERVED=$1; DIR=$2; TP=$3
  [ -d "$MODEL_ROOT/$DIR" ] || { echo "!! missing $DIR, skip"; continue; }
  if [ -f "$RES/$SERVED/.done" ]; then echo "[$(date +%T)] $SERVED already done, skip"; continue; fi
  case "$TP" in 2) DEVS="0,1";; *) DEVS="0,1,2,3";; esac
  echo "==== MODEL $SERVED ($DIR, TP=$TP on GPU $DEVS) START $(date +%T) ===="
  cat > "$CFGDIR/${SERVED}-cot.yaml" <<YAML
llm_model_path: local/${SERVED}:nothink
max_tokens: ${MAXTOK}
timeout: 300
temperature: 0.7
model_type: LLMModel
nick_name: ${SERVED}-cot
YAML
  CUDA_VISIBLE_DEVICES=$DEVS nohup vllm serve "$MODEL_ROOT/$DIR" --served-model-name "$SERVED" --port "$PORT" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$UTIL" --max-model-len 16384 \
    > "$LOG/serve_${SERVED}.log" 2>&1 &
  SPID=$!
  ok=1
  while ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$SPID" 2>/dev/null || { echo "!! server $SERVED died"; tail -30 "$LOG/serve_${SERVED}.log"; ok=0; break; }
    sleep 5; done
  [ "$ok" = 1 ] || { SPID=""; continue; }
  echo "[$(date +%T)] READY $SERVED port$PORT (pid $SPID)"

  run_arm "$SERVED" single         cot_single.yaml               & A1=$!
  run_arm "$SERVED" random${N}     baseline_random_n${N}.yaml     & A2=$!
  run_arm "$SERVED" veriselect${N} baseline_veriselect_n${N}.yaml & A3=$!
  wait "$A1" "$A2" "$A3"

  kill -TERM "$SPID" 2>/dev/null; sleep 5; kill -KILL "$SPID" 2>/dev/null; SPID=""; sleep 3
  touch "$RES/$SERVED/.done"
  echo "==== MODEL $SERVED DONE $(date +%T) ===="
done
echo "==== SCALING ALL DONE $(date +%T) ===="
python3 "$HERE/summarize_scaling_q25.py" "$RES"
