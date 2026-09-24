#!/bin/bash
# Precise teardown: stop ONLY the 6 vLLM servers this session started (recorded in
# logs/server_pids.txt), plus their EngineCore child processes. Never does a broad
# `pkill vllm` — other tasks may run their own vLLM servers on other GPUs.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$HERE/logs/server_pids.txt"
[ -f "$PIDFILE" ] || { echo "no $PIDFILE; nothing to stop"; exit 0; }

# collect a pid and all its descendants
descendants () {
  local p=$1 c
  for c in $(pgrep -P "$p" 2>/dev/null); do
    echo "$c"; descendants "$c"
  done
}

TO_KILL=""
while read -r pid name; do
  [ -n "${pid:-}" ] || continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping $name (pid $pid) + children"
    TO_KILL="$TO_KILL $pid $(descendants "$pid")"
  else
    echo "$name (pid $pid) not running"
  fi
done < "$PIDFILE"

[ -n "${TO_KILL// /}" ] || { echo "nothing alive to kill"; exit 0; }
kill -TERM $TO_KILL 2>/dev/null
sleep 5
# force any survivors
for p in $TO_KILL; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
echo "servers stopped (GPU2/GPU3 released)"
