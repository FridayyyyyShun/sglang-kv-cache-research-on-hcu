#!/usr/bin/env bash
# Independent TP8 GLM HiCache launcher. plan mode has no filesystem mutations.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
NODE=s4; PAGE=64; KV=bf16; CAP=8192; CTX=65536; OBS=1; MODE=plan; PORT=32000
RATIO=2; STORAGE=none; WRITE_POLICY=write_through; PREFETCH_POLICY=wait_complete
HOST_MODE=cache; RUNROOT=/tmp/m1-hicache; TIER=host; IO=direct; LAYOUT=layer_first
while [[ $# -gt 0 ]]; do
 case "$1" in
  --node) NODE="$2"; shift 2;; --page) PAGE="$2"; shift 2;; --kv) KV="$2"; shift 2;;
  --cap) CAP="$2"; shift 2;; --ctx) CTX="$2"; shift 2;; --obs) OBS="$2"; shift 2;;
  --mode) MODE="$2"; shift 2;; --port) PORT="$2"; shift 2;; --root) RUNROOT="$2"; shift 2;;
  --hicache-ratio) RATIO="$2"; shift 2;; --storage) STORAGE="$2"; shift 2;;
  --write-policy) WRITE_POLICY="$2"; shift 2;; --prefetch-policy) PREFETCH_POLICY="$2"; shift 2;;
  --tier) TIER="$2"; shift 2;; --host-mode) HOST_MODE="$2"; shift 2;;
  --io-backend) IO="$2"; shift 2;; --layout) LAYOUT="$2"; shift 2;;
  --help|-h) echo 'serve.sh --mode plan|run --tier device|host|storage --storage none|file --obs 0|1 --cap 8192 --hicache-ratio 2 --write-policy write_through|write_back|write_through_selective --prefetch-policy wait_complete|best_effort|timeout'; exit 0;;
  *) echo "unknown argument: $1" >&2; exit 2;;
 esac
done
[[ "$MODE" == plan || "$MODE" == run ]] || exit 2
[[ "$TIER" == device || "$TIER" == host || "$TIER" == storage ]] || exit 2
[[ "$STORAGE" == none || "$STORAGE" == file ]] || exit 2
[[ "$OBS" == 0 || "$OBS" == 1 ]] || exit 2
[[ "$HOST_MODE" == cache || "$HOST_MODE" == buffer_only ]] || exit 2
[[ "$TIER" != storage || "$STORAGE" == file ]] || { echo 'storage tier requires --storage file'; exit 2; }
[[ "$TIER" == storage || "$STORAGE" == none ]] || { echo 'file backend requires --tier storage'; exit 2; }
[[ "$HOST_MODE" != buffer_only || "$TIER" == storage ]] || exit 2
# This file only exports platform variables; unlike radix-tree/serve.sh it does not
# install .pth files or update radix-tree/.state or /tmp/m1-radix handoff.
source "$(dirname "$ROOT")/radix-tree/lib/env.sh" "$NODE"
SGLANG_SRC="$SGLANG_REPO"
case "$KV" in bf16) KV_DTYPE=bfloat16;; fp8) KV_DTYPE=fp8_e4m3;; *) exit 2;; esac
export PYTHONPATH="$HERE/obs:$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
export M1_RADIX_OBSERVE=0 M0_OBSERVE=0 HICACHE_OBSERVE="$OBS" PYTHONDONTWRITEBYTECODE=1
unset M0_TRACE_DIR HICACHE_TRACE_DIR
args=(--model-path "$MODEL_PATH" --trust-remote-code --host "$HOST" --port "$PORT"
 --dist-init-addr "$HOST:$((PORT + 1000))" --nnodes 1 --node-rank 0 --tp-size 8
 --context-length "$CTX" --kv-cache-dtype "$KV_DTYPE" --dtype bfloat16
 --mem-fraction-static 0.8 --max-total-tokens "$CAP" --chunked-prefill-size "$CAP"
 --max-prefill-tokens "$CAP" --page-size "$PAGE" --nsa-prefill-backend flashmla_sparse
 --nsa-decode-backend flashmla_kv --quantization slimquant_w4a8_marlin
 --enable-cache-report --enable-metrics --disable-cuda-graph --tokenizer-worker-num 8
 --max-running-requests 16 --random-seed 148156088 --radix-eviction-policy lru)
if [[ "$TIER" != device ]]; then
 args+=(--enable-hierarchical-cache --hicache-ratio "$RATIO"
  --hicache-write-policy "$WRITE_POLICY" --hicache-io-backend "$IO" --hicache-mem-layout "$LAYOUT")
fi
if [[ "$TIER" == storage ]]; then
 args+=(--hicache-storage-backend "$STORAGE" --hicache-storage-prefetch-policy "$PREFETCH_POLICY"
 --hicache-storage-backend-extra-config '{"prefetch_threshold":64}')
fi
[[ "$HOST_MODE" == buffer_only ]] && args+=(--hicache-host-memory-mode buffer_only)
printf 'source=%s\ntier=%s obs=%s\n' "$SGLANG_SRC" "$TIER" "$OBS"
printf '%q ' python3 -m sglang.launch_server "${args[@]}"; printf '\n'
[[ "$MODE" == plan ]] && exit 0
# Inspect allocation without sending tensors or stopping any existing service.
python3 - "$HOST" "$PORT" <<'CHECK'
import socket, sys, torch
with socket.socket() as s:
 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0:
  sys.exit('Port already in use; refusing to attach to an existing service')
if torch.cuda.device_count() != 8: sys.exit('Expected 8 GPUs')
for i in range(8):
 free, total = torch.cuda.mem_get_info(i)
 if free / total < .90: sys.exit('GPU %d busy: %.1f%% free' % (i, 100 * free / total))
print('Eight GPUs have >=90% free memory')
CHECK
TAG="${TIER}_${WRITE_POLICY}_${PREFETCH_POLICY}_cap${CAP}_p${PAGE}_obs${OBS}"
RUN="$RUNROOT/runs/$(date +%Y%m%d_%H%M%S)_${TAG}_$$"
mkdir -p "$RUN" "$ROOT/.state"
export HICACHE_TRACE_DIR="$RUN"
# Unique store per run; do not reuse storage left by another model/config.
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$RUN/storage"
printf '%s\n' "$RUN" > "$ROOT/.state/last_run"
printf 'host=%s\nport=%s\ncap=%s\npage=%s\ntier=%s\nobs=%s\nsglang=%s\n' "$HOST" "$PORT" "$CAP" "$PAGE" "$TIER" "$OBS" "$SGLANG_SRC" > "$RUN/axes.env"
printf '%q ' python3 -m sglang.launch_server "${args[@]}" > "$RUN/command.txt"
sha256sum "$SGLANG_SRC/sglang/srt/mem_cache/registry.py" "$SGLANG_SRC/sglang/srt/mem_cache/unified_radix_cache.py" > "$RUN/source.sha256"
cp "$0" "$RUN/serve.sh"
echo "RUN=$RUN"
python3 -m sglang.launch_server "${args[@]}" > >(tee "$RUN/server.log") 2>&1 &
child=$!; printf '%s\n' "$child" > "$RUN/server.pid"
trap 'kill -TERM "$child" 2>/dev/null || true' INT TERM
set +e
wait "$child"; rc=$?
printf '%s\n' "$rc" > "$RUN/exit_code"
exit "$rc"
