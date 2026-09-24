#!/usr/bin/env bash
# Re-run ONLY the veriselect@N arm (verifier-dependent) into online/veriselect${N};
# single/random${N} are verifier-independent and reused from the prior full run.
# Then re-summarize all three. Fast iteration loop for verifier changes.
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; EVAL="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model weights}"
RES="$EVAL/results/gtbench_veriselect/online"; LOG="$RES/logs"; mkdir -p "$LOG"
N=${N:-8}; NUM_MATCHES=${NUM_MATCHES:-20}; WORKERS=${WORKERS:-8}; GPU=${GPU:-2}; PORT=${PORT:-8034}
GAMES=${GAMES:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}
SERVED=q25_7b; DIR=Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export LOCAL_LLM_ENDPOINTS="{\"$SERVED\":\"http://127.0.0.1:$PORT/v1\"}"

echo "[$(date +%T)] SERVE $SERVED GPU=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL_ROOT/$DIR" --served-model-name "$SERVED" --port "$PORT" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --max-model-len 16384 > "$LOG/serve_rerun.log" 2>&1 &
SPID=$!
trap 'kill -TERM $SPID 2>/dev/null; sleep 4; kill -KILL $SPID 2>/dev/null' EXIT
while ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$SPID" 2>/dev/null || { echo "!! server died"; tail -20 "$LOG/serve_rerun.log"; exit 1; }; sleep 5; done
echo "[$(date +%T)] READY (pid $SPID)"

rm -rf "$RES/veriselect${N}"
echo "[$(date +%T)] veriselect${N} arm start"
( cd "$GTBENCH" && python3 -m gamingbench.main --num-matches "$NUM_MATCHES" --exp-root "$RES/veriselect${N}" \
    --seed 0 --game-names $GAMES \
    --agent-configs "gamingbench/configs/agent_configs/baseline_veriselect_n${N}.yaml" gamingbench/configs/agent_configs/random_agent.yaml \
    --model-configs "gamingbench/configs/model_configs/${SERVED}-cot.yaml" gamingbench/configs/model_configs/dummy-random.yaml \
    --exchange-first-player --num-workers "$WORKERS" --threshold-matches $((NUM_MATCHES + 10)) ) > "$LOG/arm_veriselect_rerun.log" 2>&1
echo "[$(date +%T)] veriselect${N} arm done rc=$?"
python3 "$HERE/summarize_online.py" "$RES" "$GAMES"