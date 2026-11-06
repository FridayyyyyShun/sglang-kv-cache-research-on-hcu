#!/usr/bin/env bash
# M1 radix-tree 实验：唯一启动入口。base = TP8 / 无 MTP / 无 EP / 无 CP / 无 layer split。
# radix 全程 ON（除非 --radix off 做对照）。
#
# 用法：
#   bash lib/serve.sh --node example                 # 起服务（默认基座）
#   bash lib/serve.sh --mode plan               # 只打印命令行，不起
#   bash lib/serve.sh --kv fp8 --page 64        # 换 KV 量化
#   bash lib/serve.sh --radix off               # 对照组（仅用于证明命中来自 radix）
#   bash lib/serve.sh --cap 8192                # 缩小池子，强制频繁驱逐
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

NODE=example; RADIX=on; PAGE=64; KV=bf16; CAP=65536; CTX=65536; POLICY=lru
EAGLE=0; CHUNK_PREFILL=32768; OBS=1; SRC=repo; MODE=run; PORT=31000
RUNROOT=/tmp/m1-radix

while [[ $# -gt 0 ]]; do
 case "$1" in
  --node) NODE="$2"; shift 2;;
  --radix) RADIX="$2"; shift 2;;            # on|off
  --page) PAGE="$2"; shift 2;;
  --kv) KV="$2"; shift 2;;                  # bf16|fp8
  --cap) CAP="$2"; shift 2;;                # max-total-tokens
  --ctx) CTX="$2"; shift 2;;
  --policy) POLICY="$2"; shift 2;;          # lru|lfu|slru|priority
  --eagle) EAGLE="$2"; shift 2;;            # 0|1（默认不开 MTP）
  --chunk) CHUNK_PREFILL="$2"; shift 2;;
  --obs) OBS="$2"; shift 2;;                # 1|0 观测钩子
  --src) SRC="$2"; shift 2;;                # repo|pkg
  --mode) MODE="$2"; shift 2;;              # plan|run
  --port) PORT="$2"; shift 2;;
  --root) RUNROOT="$2"; shift 2;;
  -h|--help) sed -n '2,12p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;;
 esac
done

# shellcheck source=./env.sh
source "$HERE/env.sh" "$NODE"

case "$SRC" in
 repo) SGLANG_SRC="$SGLANG_REPO";;
 pkg)  SGLANG_SRC="$SGLANG_PKG";;
 *) echo "--src must be repo|pkg" >&2; exit 2;;
esac
case "$KV" in
 bf16) KV_DTYPE=bfloat16;;
 fp8)  KV_DTYPE=fp8_e4m3;;
 *) echo "--kv must be bf16|fp8" >&2; exit 2;;
esac

OBS_DIR="$ROOT/lib/obs"
mkdir -p "$OBS_DIR"
# 关键：SGLang 用 spawn 起 scheduler 子进程，PYTHONPATH/环境变量不一定传过去。
# 用 .pth 把 obs 目录写进 site-packages，让解释器一启动就 import sitecustomize —— 对 spawn 也生效。
SITE="$(python3 -c 'import site,sys; print(next((p for p in site.getsitepackages() if "dist-packages" in p), site.getsitepackages()[0]))' 2>/dev/null || echo /usr/local/lib/python3.10/dist-packages)"
PTH="$SITE/zz_m1_radix_obs.pth"
if [[ "$MODE" == plan ]]; then
 export PYTHONPATH="$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
elif [[ "$OBS" == 1 ]]; then
 # SGLang 用 spawn 起 scheduler 子进程，PYTHONPATH/环境变量不保证传过去。
 # .pth 行在每个 Python 解释器启动早期执行：把 obs 目录放进 sys.path 并 import m1_bootstrap，
 # 由它安装观测钩子 —— 对主进程和 spawn 子进程都生效。
 printf 'import sys; sys.path.insert(0, %s); import m1_bootstrap\n' "\"$OBS_DIR\"" > "$PTH" 2>/dev/null \
   && echo "installed obs .pth -> $PTH" \
   || echo "WARN 无法写 ${PTH}，退回 PYTHONPATH（spawn 子进程可能观测不到）"
 export PYTHONPATH="$OBS_DIR:$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
 export M1_RADIX_OBSERVE=1
 export M1_SNAPSHOT_EVERY="${M1_SNAPSHOT_EVERY:-0}"
else
 rm -f "$PTH" 2>/dev/null || true
 export PYTHONPATH="$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
 export M1_RADIX_OBSERVE=0
fi
export M1_RADIX_TIME="${M1_RADIX_TIME:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${M1_CHUNK_THRESH:-0}"

# ---------------- 参数组装：base 只到 TP8 ----------------
args=(--model-path "$MODEL_PATH" --trust-remote-code --host "$HOST" --port "$PORT"
 --dist-init-addr "$HOST:6100" --nnodes 1 --node-rank 0 --tp-size 8
 --context-length "$CTX" --kv-cache-dtype "$KV_DTYPE" --dtype bfloat16
 --mem-fraction-static 0.8 --max-total-tokens "$CAP"
 --chunked-prefill-size "$CHUNK_PREFILL" --max-prefill-tokens "$CHUNK_PREFILL"
 --page-size "$PAGE"
 --nsa-prefill-backend flashmla_sparse --nsa-decode-backend flashmla_kv
 --quantization slimquant_w4a8_marlin --enable-cache-report
 --disable-cuda-graph --tokenizer-worker-num 8
 --max-running-requests 16 --random-seed 148156088)

# radix：默认全程 ON
if [[ "$RADIX" == off ]]; then
 args+=(--disable-radix-cache)
else
 args+=(--radix-eviction-policy "$POLICY")
fi

# MTP/EAGLE：默认不开，开了会变成 bigram，命中长度少 1 再对齐
if [[ "$EAGLE" == 1 ]]; then
 args+=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --json-model-override-args '{"index_share_for_mtp_iteration": false}')
fi

TAG="r${RADIX}_kv${KV}_p${PAGE}_ev${POLICY}_eg${EAGLE}_cap${CAP}_ctx${CTX}_ck${M1_CHUNK_THRESH:-0}_${SRC}"
printf 'tag=%s\n' "$TAG"
printf 'sglang=%s\n' "$SGLANG_SRC"
printf '%q ' python3 -m sglang.launch_server "${args[@]}"; printf '\n'
[[ "$MODE" == plan ]] && exit 0

RUN="$RUNROOT/runs/$(date +%Y%m%d_%H%M%S)_${TAG}_$$"
mkdir -p "$RUN"
export M0_TRACE_DIR="$RUN"
# handoff：spawn 出来的 scheduler 子进程可能丢 M0_TRACE_DIR，用固定文件兜底
mkdir -p "$RUNROOT"
printf '%s\n' "$RUN" > "$RUNROOT/.trace_dir"
mkdir -p "$ROOT/.state"
printf '%s\n' "$RUN" > "$ROOT/.state/last_run"
printf 'host=%s\nport=%s\nnode=%s\nradix=%s\nkv=%s\npage=%s\npolicy=%s\neagle=%s\ncap=%s\nctx=%s\nchunk_prefill=%s\nobs=%s\nsrc=%s\nsglang=%s\n' \
 "$HOST" "$PORT" "$NODE" "$RADIX" "$KV" "$PAGE" "$POLICY" "$EAGLE" "$CAP" "$CTX" "$CHUNK_PREFILL" "$OBS" "$SRC" "$SGLANG_SRC" \
 > "$RUN/axes.env"
cp "$0" "$HERE/env.sh" "$RUN/" 2>/dev/null || true
cp "$OBS_DIR/m1_radix_observe.py" "$OBS_DIR/sitecustomize.py" "$RUN/" 2>/dev/null || true
sha256sum "$SGLANG_SRC/sglang/srt/mem_cache/radix_cache.py" >  "$RUN/source.sha256" 2>/dev/null || true
sha256sum "$SGLANG_SRC/sglang/srt/mem_cache/evict_policy.py" >> "$RUN/source.sha256" 2>/dev/null || true
sha256sum "$SGLANG_SRC/sglang/srt/mem_cache/utils.py"       >> "$RUN/source.sha256" 2>/dev/null || true

echo "RUN=$RUN"
[[ -f "$RUN/axes.env" ]] && sed 's/^/  /' "$RUN/axes.env"

# 只占用全闲卡，不打断别人的服务
python3 - <<'CHECK'
import torch,os,sys
if torch.cuda.device_count()!=8: sys.exit('expected 8 GPUs, got %d'%torch.cuda.device_count())
for i in range(8):
    free,total=torch.cuda.mem_get_info(i)
    if free/total < .90: sys.exit('GPU %d busy: free %.1f%%'%(i,free/total*100))
print('all GPUs idle >=90%')
CHECK

set +e
python3 -m sglang.launch_server "${args[@]}" 2>&1 | tee "$RUN/server.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$RUN/exit_code"
exit "$rc"
