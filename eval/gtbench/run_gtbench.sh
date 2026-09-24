#!/bin/bash
# GTBench for one Qwen3 model+mode (prompt_agent) vs the built-in random_agent.
# Usage: bash run_gtbench.sh <served> <think|nothink> [num_matches] [games...]
#   served in: q0_6b q1_7b q4b q8b q14b q32b
set -u
SERVED=${1:?served name required (q0_6b|q1_7b|q4b|q8b|q14b|q32b)}
MODE=${2:?mode required (think|nothink)}
NUM_MATCHES=${3:-20}
shift 3 2>/dev/null || shift 2
GAMES=${@:-"tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"}

HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GTBENCH="$REPO/bench/GTBench"

export NO_PROXY="127.0.0.1,localhost"; export no_proxy="127.0.0.1,localhost"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export LOCAL_LLM_ENDPOINTS='{"q32b":"http://127.0.0.1:8001/v1","q14b":"http://127.0.0.1:8002/v1","q8b":"http://127.0.0.1:8003/v1","q4b":"http://127.0.0.1:8004/v1","q1_7b":"http://127.0.0.1:8005/v1","q0_6b":"http://127.0.0.1:8006/v1"}'

cd "$GTBENCH"
python3 -m gamingbench.main \
    --num-matches "${NUM_MATCHES}" \
    --exp-root "$EVAL/results/gtbench/${SERVED}-${MODE}" \
    --seed 0 \
    --game-names ${GAMES} \
    --agent-configs gamingbench/configs/agent_configs/prompt_agent.yaml gamingbench/configs/agent_configs/random_agent.yaml \
    --model-configs gamingbench/configs/model_configs/${SERVED}-${MODE}.yaml gamingbench/configs/model_configs/dummy-random.yaml \
    --exchange-first-player \
    --num-workers 8 \
    --threshold-matches $((NUM_MATCHES + 10))
echo "GTBENCH DONE: ${SERVED}-${MODE}"
