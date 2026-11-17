#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   P_NODE=S3 D_NODE=S4 bash router.sh
#   P_NODE=s1 D_NODE=s2 ROUTER_POLICY=round_robin bash router.sh
# Explicit PREFILL_URL / DECODE_URL override URLs derived from node aliases.

resolve_node_ip() {
  local node
  node="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${node}" in
    s1|1) printf '%s' '10.41.103.12' ;;
    s2|2) printf '%s' '10.41.103.10' ;;
    s3|3) printf '%s' '10.41.101.164' ;;
    s4|4) printf '%s' '10.41.103.4' ;;
    s7|7) printf '%s' '10.41.103.8' ;;
    *)
      printf 'Unknown node %q. Valid nodes: S1, S2, S3, S4, S7 (or 1,2,3,4,7).\n' "$1" >&2
      return 2
      ;;
  esac
}

normalize_node() {
  local node
  node="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${node}" in
    s1|1) printf '%s' 'S1' ;;
    s2|2) printf '%s' 'S2' ;;
    s3|3) printf '%s' 'S3' ;;
    s4|4) printf '%s' 'S4' ;;
    s7|7) printf '%s' 'S7' ;;
    *) resolve_node_ip "$1" >/dev/null; return 2 ;;
  esac
}

P_NODE="$(normalize_node "${P_NODE:-S1}")"
D_NODE="$(normalize_node "${D_NODE:-S2}")"
P_HOST="$(resolve_node_ip "${P_NODE}")"
D_HOST="$(resolve_node_ip "${D_NODE}")"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-30016}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30023}"
PREFILL_URL="${PREFILL_URL:-http://${P_HOST}:${PREFILL_PORT}}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8299}"
DECODE_URL="${DECODE_URL:-http://${D_HOST}:${DECODE_PORT}}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
PROMETHEUS_HOST="${PROMETHEUS_HOST:-0.0.0.0}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-29018}"
MODEL_PATH="${MODEL_PATH:-/model/GLM-5.3-Channel-INT4-w4a8}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-3600}"
WORKER_STARTUP_TIMEOUT_SECS="${WORKER_STARTUP_TIMEOUT_SECS:-1800}"
# Health-check tuning: GLM /health does a real forward (~3s); under heavy long
# prefill load the probe queues up and times out, tripping the circuit breaker
# even though P is actually alive. Loosen these so the router does not falsely
# mark a busy-but-healthy prefill worker as down.
HEALTH_CHECK_INTERVAL_SECS="${HEALTH_CHECK_INTERVAL_SECS:-60}"
HEALTH_CHECK_TIMEOUT_SECS="${HEALTH_CHECK_TIMEOUT_SECS:-120}"
HEALTH_FAILURE_THRESHOLD="${HEALTH_FAILURE_THRESHOLD:-10}"
HEALTH_SUCCESS_THRESHOLD="${HEALTH_SUCCESS_THRESHOLD:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROUTER_LOG_DIR:-${SCRIPT_DIR}/logs/$(date +%Y%m%d)}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/router-pd-${P_NODE}-${D_NODE}-${ROUTER_POLICY}_$(date +%Y%m%d_%H%M%S).log"

case "${ROUTER_POLICY}" in
  random|round_robin|cache_aware|power_of_two|bucket|manual|consistent_hashing|prefix_hash) ;;
  *) echo "Unsupported ROUTER_POLICY=${ROUTER_POLICY}" >&2; exit 2 ;;
esac

command -v sglang-router >/dev/null 2>&1 || {
  echo "sglang-router is not installed or not in PATH" >&2
  exit 127
}

ROUTER_CMD=(
  sglang-router launch
  --host "${ROUTER_HOST}"
  --port "${ROUTER_PORT}"
  --pd-disaggregation
  --prefill "${PREFILL_URL}" "${PREFILL_BOOTSTRAP_PORT}"
  --decode "${DECODE_URL}"
  --policy "${ROUTER_POLICY}"
  --model-path "${MODEL_PATH}"
  --reasoning-parser glm45
  --tool-call-parser glm47_moe
  --request-id-headers x-request-id x-session-id
  --request-timeout-secs "${REQUEST_TIMEOUT_SECS}"
  --worker-startup-timeout-secs "${WORKER_STARTUP_TIMEOUT_SECS}"
  --health-check-interval-secs "${HEALTH_CHECK_INTERVAL_SECS}"
  --health-check-timeout-secs "${HEALTH_CHECK_TIMEOUT_SECS}"
  --health-failure-threshold "${HEALTH_FAILURE_THRESHOLD}"
  --health-success-threshold "${HEALTH_SUCCESS_THRESHOLD}"
  --prometheus-host "${PROMETHEUS_HOST}"
  --prometheus-port "${PROMETHEUS_PORT}"
  --log-level info
)

printf 'router log: %s\n' "${LOG_FILE}"
printf 'nodes: P_NODE=%s (%s), D_NODE=%s (%s)\n' "${P_NODE}" "${P_HOST}" "${D_NODE}" "${D_HOST}"
printf 'listen: http://%s:%s\n' "${ROUTER_HOST}" "${ROUTER_PORT}"
printf 'prefill: %s (bootstrap=%s)\n' "${PREFILL_URL}" "${PREFILL_BOOTSTRAP_PORT}"
printf 'decode: %s\n' "${DECODE_URL}"
printf 'policy: %s\n' "${ROUTER_POLICY}"

if [[ "${ROUTER_DRY_RUN:-0}" == "1" ]]; then
  printf 'command:'
  printf ' %q' "${ROUTER_CMD[@]}"
  printf '\n'
  exit 0
fi

export PYTHONUNBUFFERED=1
"${ROUTER_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
