#!/usr/bin/env bash
# Data-gen for the gold-free PROCESS-VERIFIER experiment (veriselect best-of-N).
# ONE model (Qwen2.5-7B-Instruct, nothink), CoT prompt (reasoning elicited + captured),
# num_generations=N (default 8) so every move records N candidate (reasoning, move) pairs
# in queries[].llm_output / .raw_reasoning. Opponent = built-in random_agent.
#
# The prompt template is FROZEN here (default CoT "Thought:/Action:"). Iterate the verifier
# OFFLINE on this fixed data with analyze_veriselect_gtbench.py — no GPU re-gen per tweak.
#
# SAFETY: only kills the vLLM PID this script started (recorded in PIDFILE); never `pkill vllm`.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; EVAL="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
CFGDIR="$GTBENCH/gamingbench/configs/model_configs"; ACFGDIR="$GTBENCH/gamingbench/configs/agent_configs"
RES="$EVAL/results/gtbench_veriselect"; LOG="$RES/logs"; mkdir -p "$LOG" "$RES"
PIDFILE="$LOG/server_pids.txt"; : > "$PIDFILE"

N=${N:-8}                       # candidates per move
NUM_MATCHES=${NUM_MATCHES:-20}
WORKERS=${WORKERS:-8}
MAXTOK=${MAXTOK:-2048}          # room for CoT "Thought:" + "Action:"
GPU=${GPU:-0}
PORT=${PORT:-8031}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}
SERVED=q25_7b; DIR=Qwen2.5-7B-Instruct

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export LOCAL_LLM_ENDPOINTS="{\"$SERVED\":\"http://127.0.0.1:$PORT/v1\"}"

# ---- configs: nothink model + CoT agent with N generations (majority_vote off => all N saved) ----
cat > "$CFGDIR/${SERVED}-cot.yaml" <<YAML
llm_model_path: local/${SERVED}:nothink
max_tokens: ${MAXTOK}
timeout: 180
temperature: 0.7
model_type: LLMModel
nick_name: ${SERVED}-cot
YAML
cat > "$ACFGDIR/cot_agent_n${N}.yaml" <<YAML
agent_name: CoTAgent
num_generations: ${N}
majority_vote: False
YAML
echo "[$(date +%T)] wrote configs (N=$N, max_tokens=$MAXTOK)"

# ---- serve ----
echo "[$(date +%T)] SERVE $SERVED ($DIR) GPU=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL_ROOT/$DIR" \
  --served-model-name "$SERVED" --port "$PORT" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --max-model-len 16384 \
  > "$LOG/serve_${SERVED}.log" 2>&1 &
SPID=$!; echo "$SPID $SERVED" >> "$PIDFILE"
trap 'echo "[$(date +%T)] stopping server $SPID"; kill -TERM $SPID 2>/dev/null; sleep 4; kill -KILL $SPID 2>/dev/null' EXIT
while ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { echo "!! server died; tail:"; tail -30 "$LOG/serve_${SERVED}.log"; exit 1; }
  sleep 5
done
echo "[$(date +%T)] READY $SERVED (pid $SPID)"

# ---- run: CoT(N) vs random, all games, save all N generations ----
echo "==== DATAGEN START $(date +%T) N=$N matches=$NUM_MATCHES games=[$GAMES] ===="
( cd "$GTBENCH" && python3 -m gamingbench.main \
    --num-matches "$NUM_MATCHES" \
    --exp-root "$RES/${SERVED}-cot-n${N}" \
    --seed 0 --game-names $GAMES \
    --agent-configs "gamingbench/configs/agent_configs/cot_agent_n${N}.yaml" gamingbench/configs/agent_configs/random_agent.yaml \
    --model-configs "gamingbench/configs/model_configs/${SERVED}-cot.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
    --exchange-first-player --num-workers "$WORKERS" \
    --threshold-matches $((NUM_MATCHES + 10)) ) > "$LOG/datagen.log" 2>&1
echo "==== DATAGEN DONE $(date +%T) rc=$? -> $RES/${SERVED}-cot-n${N} ===="