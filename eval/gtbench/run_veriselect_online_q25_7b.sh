#!/usr/bin/env bash
# ONLINE veriselect experiment: Qwen2.5-7B (CoT) vs the built-in random_agent, 7 games.
# Three arms share ONE server; each plays full games (per-move selection changes the
# trajectory, so this must be online, not offline re-scoring):
#   single      : CoTAgent, n=1                      -> pass@1 floor
#   random@8    : BaselineAgent selector=random, n=8 -> best-of-8 with a RANDOM pick
#   veriselect@8: BaselineAgent selector=veriselect,n=8 -> gold-free process-verifier pick
# veriselect@8 > single  = the method helps; veriselect@8 > random@8 = the verifier (not
# just more samples) is what helps. Metric = (win + 0.5*draw)/normal, printed at the end.
#
# SAFETY: kills only the vLLM PID this script started.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; EVAL="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
CFGDIR="$GTBENCH/gamingbench/configs/model_configs"; ACFGDIR="$GTBENCH/gamingbench/configs/agent_configs"
RES="$EVAL/results/gtbench_veriselect/online"; LOG="$RES/logs"; mkdir -p "$LOG" "$RES"

N=${N:-8}; NUM_MATCHES=${NUM_MATCHES:-20}; WORKERS=${WORKERS:-8}; MAXTOK=${MAXTOK:-2048}
GPU=${GPU:-0}; PORT=${PORT:-8032}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}
SERVED=q25_7b; DIR=Qwen2.5-7B-Instruct

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export LOCAL_LLM_ENDPOINTS="{\"$SERVED\":\"http://127.0.0.1:$PORT/v1\"}"

cat > "$CFGDIR/${SERVED}-cot.yaml" <<YAML
llm_model_path: local/${SERVED}:nothink
max_tokens: ${MAXTOK}
timeout: 180
temperature: 0.7
model_type: LLMModel
nick_name: ${SERVED}-cot
YAML
cat > "$ACFGDIR/cot_single.yaml"        <<< $'agent_name: CoTAgent\nnum_generations: 1\nmajority_vote: False'
cat > "$ACFGDIR/baseline_random_n${N}.yaml"     <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: random\nseed: 0'
cat > "$ACFGDIR/baseline_veriselect_n${N}.yaml" <<< $'agent_name: BaselineAgent\nnum_generations: '"$N"$'\nmajority_vote: False\nselector: veriselect\nseed: 0'
echo "[$(date +%T)] wrote configs"

echo "[$(date +%T)] SERVE $SERVED GPU=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL_ROOT/$DIR" \
  --served-model-name "$SERVED" --port "$PORT" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --max-model-len 16384 \
  > "$LOG/serve.log" 2>&1 &
SPID=$!
trap 'echo "[$(date +%T)] stop server $SPID"; kill -TERM $SPID 2>/dev/null; sleep 4; kill -KILL $SPID 2>/dev/null' EXIT
while ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { echo "!! server died"; tail -30 "$LOG/serve.log"; exit 1; }
  sleep 5
done
echo "[$(date +%T)] READY $SERVED (pid $SPID)"

run_arm () {  # $1=arm-label $2=agent-config
  echo "[$(date +%T)] ARM $1 start"
  ( cd "$GTBENCH" && python3 -m gamingbench.main \
      --num-matches "$NUM_MATCHES" --exp-root "$RES/$1" --seed 0 --game-names $GAMES \
      --agent-configs "gamingbench/configs/agent_configs/$2" gamingbench/configs/agent_configs/random_agent.yaml \
      --model-configs "gamingbench/configs/model_configs/${SERVED}-cot.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
      --exchange-first-player --num-workers "$WORKERS" --threshold-matches $((NUM_MATCHES + 10)) \
    ) > "$LOG/arm_$1.log" 2>&1
  echo "[$(date +%T)] ARM $1 done rc=$?"
}

echo "==== ONLINE START $(date +%T) N=$N matches=$NUM_MATCHES ===="
run_arm single       cot_single.yaml
run_arm random${N}   baseline_random_n${N}.yaml
run_arm veriselect${N} baseline_veriselect_n${N}.yaml
echo "==== ONLINE DONE $(date +%T) ===="

python3 "$HERE/summarize_online.py" "$RES" "$GAMES"