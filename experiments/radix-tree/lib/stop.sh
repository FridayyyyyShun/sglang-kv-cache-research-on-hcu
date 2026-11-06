#!/usr/bin/env bash
# 停掉本目录起的服务：bash lib/stop.sh
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PORT="${1:-31000}"
pids=$(pgrep -f "sglang.launch_server.*--port ${PORT}" || true)
[[ -z "$pids" ]] && { echo "no server on port $PORT"; exit 0; }
echo "stopping: $pids"
kill $pids 2>/dev/null || true
for i in $(seq 1 30); do
  pgrep -f "sglang.launch_server.*--port ${PORT}" >/dev/null || break
  sleep 2
done
left=$(pgrep -f "sglang.launch_server.*--port ${PORT}" || true)
[[ -n "$left" ]] && { echo "force kill: $left"; kill -9 $left 2>/dev/null || true; }
echo "stopped"
