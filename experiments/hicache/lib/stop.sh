#!/usr/bin/env bash
# Stop only the PID recorded for this experiment, after checking its command line.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-$(<"$ROOT/.state/last_run")}"
PID="$(<"$RUN/server.pid")"
[[ "$PID" =~ ^[0-9]+$ && "$PID" -gt 1 ]] || exit 2
CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline")"
PORT="$(sed -n 's/^port=//p' "$RUN/axes.env")"
[[ "$CMD" == *sglang.launch_server* && "$CMD" == *"--port $PORT "* ]] || { echo 'PID no longer matches this run'; exit 1; }
kill -TERM "$PID"
echo "Sent TERM to run $RUN PID $PID"
