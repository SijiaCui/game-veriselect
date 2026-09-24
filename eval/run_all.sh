#!/bin/bash
# Full external eval: 6 Qwen3 models x {think,nothink} on GTBench.
# Runs 6 model-pipelines in parallel (one per vLLM server / GPU slice); within a
# pipeline the 2 modes run sequentially to keep per-server load bounded.
#
#   GTBench: candidate (model,mode) vs built-in random_agent
#
# Usage: bash run_all.sh [--smoke]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/logs"; mkdir -p "$LOG"

MODELS="q32b q14b q8b q4b q1_7b q0_6b"
MODES="nothink think"
GT_GAMES="tictactoe connect4 kuhn_poker nim prisoners_dilemma liars_dice first_sealed_auction"
N=20

if [ "${1:-}" = "--smoke" ]; then
  echo "### SMOKE run: 2 models, 1 game, 2 matches ###"
  MODELS="q0_6b q4b"
  GT_GAMES="tictactoe"
  N=2
fi

pipeline () {  # runs both modes for one model, sequentially
  local m=$1
  for mode in $MODES; do
    echo "[$m/$mode] GTBench ..."
    bash "$HERE/gtbench/run_gtbench.sh" "$m" "$mode" "$N" $GT_GAMES \
      > "$LOG/gt_${m}_${mode}.log" 2>&1
  done
  echo "PIPELINE DONE: $m"
}

for m in $MODELS; do
  pipeline "$m" &
done
wait
echo "ALL EXTERNAL EVAL COMPLETE"
