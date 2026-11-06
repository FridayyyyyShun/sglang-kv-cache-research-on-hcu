#!/usr/bin/env bash
# T1：验证 GPU→Host write_through 备份提交及完成；不验证回载和KV指纹。
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel \
#   --policy lru --radix on --kv bf16 --page 64 \
#   --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# 服务就绪后另开终端（默认1条4096-token输入、生成1个token）：
# bash run_hicache.sh --only t1 --host 127.0.0.1
# 或直接运行：
# python3 exp/t1_hicache_write_through.py --host 127.0.0.1 --port 31000 \
#   --run-dir "$(cat .state/last_run)" --page 64 --prompt-len 4096 --wait-seconds 60
# 详细说明：README.md。保留本机env.sh，核对模型、IP和网卡。
# ---------------- T2 / T3 服务启动示例 ----------------
# 在实验目录（lib 的上一级）执行。先停止旧服务，等待 GPU 释放。
# 保留本机 env.sh 的模型、IP、网卡配置；末尾加 --mode plan 可只预览。
#
# T2：Host 回载及 KV 指纹正确性；指纹模式会同步 GPU、复制数据到 CPU。
# M1_HICACHE_VERIFY=1 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel \
#   --policy lru --radix on --kv bf16 --page 64 \
#   --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# 服务就绪后另开终端：
# bash run_hicache.sh --only t2 --host 127.0.0.1
#
# T4 性能对比：关闭两种额外校验，保留基础 observer。
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel \
#   --policy lru --radix on --kv bf16 --page 64 \
#   --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# 服务就绪后另开终端（会清缓存，仅用于空闲专用测试服务）：
# bash run_hicache.sh --only t4 --host 127.0.0.1 --policy lru --cycles 10 --repeats 3
#
# T3 淘汰顺序审计：开启候选节点审计，关闭 KV 指纹模式。
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=1 bash lib/serve.sh \
#   --hicache on --hc-ratio 4 --hc-policy write_through \
#   --hc-layout page_first --hc-io kernel \
#   --policy lru --radix on --kv bf16 --page 64 \
#   --cap 16384 --ctx 32768 --chunk 8192 \
#   --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# 服务就绪后另开终端：
# bash run_hicache.sh --only t3 --host 127.0.0.1 --policy lru
#
# 对比其他策略时，启动与测试命令的 --policy 同时改为 lfu/slru/priority，
# 并重启服务；测试脚本不会自动切换策略或修改服务环境变量。
# 不要混用带指纹/候选审计的请求耗时与关闭校验后的性能结果。
# ------------------------------------------------------
# HiCache: --hicache on|off --hc-ratio 1.5 --hc-policy write_through --hc-layout page_first --hc-io kernel
# M1 radix-tree 实验：唯一启动入口。base = TP8 / 无 MTP / 无 EP / 无 CP / 无 layer split。
# radix 全程 ON（除非 --radix off 做对照）。
#
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

NODE=example; RADIX=on; PAGE=64; KV=bf16; CAP=65536; CTX=65536; POLICY=lru
EAGLE=0; CHUNK_PREFILL=32768; OBS=1; SRC=repo; MODE=run; PORT=31000
HICACHE=on; HC_RATIO=1.5; HC_POLICY=write_through; HC_LAYOUT=page_first; HC_IO=kernel
RUNROOT="${M1_RUNROOT:-$ROOT/.runtime}"
OBS_PTH=0

while [[ $# -gt 0 ]]; do
 case "$1" in
  --obs-pth) OBS_PTH="$2"; shift 2;;
  --hicache) HICACHE="$2"; shift 2;;
  --hc-ratio) HC_RATIO="$2"; shift 2;;
  --hc-policy) HC_POLICY="$2"; shift 2;;
  --hc-layout) HC_LAYOUT="$2"; shift 2;;
  --hc-io) HC_IO="$2"; shift 2;;
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

case "$HICACHE" in on|off) ;; *) echo "--hicache must be on|off" >&2; exit 2;; esac
case "$HC_POLICY" in write_through|write_through_selective|write_back) ;; *) echo "invalid --hc-policy" >&2; exit 2;; esac
case "$HC_IO" in kernel|direct) ;; *) echo "invalid --hc-io" >&2; exit 2;; esac
for n in "$PAGE" "$CAP" "$CTX" "$CHUNK_PREFILL" "$PORT"; do
 [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "positive integer required: $n" >&2; exit 2; }
done
[[ "$HICACHE" != on || "$RADIX" == on ]] || { echo "HiCache requires --radix on" >&2; exit 2; }
(( CAP >= PAGE && CTX >= PAGE )) || { echo "cap/ctx must be >= page" >&2; exit 2; }

case "$MODE" in plan|run) ;; *) echo "--mode must be plan|run" >&2; exit 2;; esac
case "$RADIX" in on|off) ;; *) echo "--radix must be on|off" >&2; exit 2;; esac
case "$OBS:$OBS_PTH:$EAGLE" in [01]:[01]:[01]) ;; *) echo "obs/obs-pth/eagle must be 0|1" >&2; exit 2;; esac
[[ "$HC_RATIO" =~ ^[0-9]+([.][0-9]+)?$ && "$HC_RATIO" != 0 ]] || { echo "invalid --hc-ratio" >&2; exit 2; }
(( PORT <= 65535 )) || { echo "invalid port" >&2; exit 2; }

OBS_DIR="$ROOT/lib/obs"
export M1_TRACE_HANDOFF="$RUNROOT/.trace_dir"
export M1_RADIX_OBSERVE="$OBS"
if [[ "$OBS" == 1 ]]; then
 export PYTHONPATH="$OBS_DIR:$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
else
 export PYTHONPATH="$SGLANG_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
export M1_SNAPSHOT_EVERY="${M1_SNAPSHOT_EVERY:-0}"

# Standard Python spawn inherits PYTHONPATH. System .pth installation is opt-in.
if [[ "$MODE" == run ]]; then
 mkdir -p "$RUNROOT"
 RUNROOT="$(cd -- "$RUNROOT" && pwd)"
 export M1_TRACE_HANDOFF="$RUNROOT/.trace_dir"
 export TMPDIR="${TMPDIR:-$RUNROOT/tmp}"
 export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNROOT/cache}"
 export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
 export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
 for d in "$RUNROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"; do
   mkdir -p "$d"
   probe="$d/.m1-write-check-$$"
   printf 'check\n' > "$probe" || { echo "cannot write: $d" >&2; exit 1; }
   rm -- "$probe"
 done
 df -h "$RUNROOT" "$TMPDIR" "$XDG_CACHE_HOME"
 if [[ "$OBS" == 1 && "$OBS_PTH" == 1 ]]; then
   SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
   printf 'import sys; sys.path.insert(0, %s); import m1_bootstrap\n' "\"$OBS_DIR\"" > "$SITE/zz_m1_radix_obs.pth"
 fi
fi
export M1_RADIX_TIME="${M1_RADIX_TIME:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${M1_CHUNK_THRESH:-0}"

# ---------------- 参数组装：base 只到 TP8 ----------------
args=(--model-path "$MODEL_PATH" --trust-remote-code --host "$HOST" --port "$PORT"
 --dist-init-addr "$HOST:6100" --nnodes 1 --node-rank 0 --tp-size 8
 --context-length "$CTX" --kv-cache-dtype "$KV_DTYPE" --dtype bfloat16
 --mem-fraction-static 0.85 --max-total-tokens "$CAP"
 --chunked-prefill-size "$CHUNK_PREFILL" --max-prefill-tokens "$CHUNK_PREFILL"
 --page-size "$PAGE"
 --nsa-prefill-backend flashmla_sparse --nsa-decode-backend flashmla_kv
 --quantization slimquant_w4a8_marlin --enable-cache-report
 --disable-cuda-graph --tokenizer-worker-num 8
 --max-running-requests 16 --random-seed 148156088)

if [[ "$HICACHE" == on ]]; then
 args+=(--enable-hierarchical-cache --hicache-ratio "$HC_RATIO"
        --hicache-mem-layout "$HC_LAYOUT" --hicache-io-backend "$HC_IO"
        --hicache-write-policy "$HC_POLICY")
fi

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

TAG="r${RADIX}_kv${KV}_p${PAGE}_ev${POLICY}_eg${EAGLE}_cap${CAP}_ctx${CTX}_ck${M1_CHUNK_THRESH:-0}_${SRC}_hc${HICACHE}_${HC_POLICY}_${HC_LAYOUT}_${HC_IO}"
printf 'tag=%s\n' "$TAG"
printf 'sglang=%s\nrunroot=%s\nobs_pth=%s\n' "$SGLANG_SRC" "$RUNROOT" "$OBS_PTH"
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
printf 'hicache=%s\nhc_ratio=%s\nhc_policy=%s\nhc_layout=%s\nhc_io=%s\n' \
 "$HICACHE" "$HC_RATIO" "$HC_POLICY" "$HC_LAYOUT" "$HC_IO" >> "$RUN/axes.env"
cp "$0" "$HERE/env.sh" "$RUN/" 2>/dev/null || true
cp "$OBS_DIR/m1_radix_observe.py" "$OBS_DIR/sitecustomize.py" "$OBS_DIR/m1_bootstrap.py" "$RUN/" 2>/dev/null || true
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
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [[ "$rc" == 0 && "${statuses[1]}" != 0 ]]; then rc=${statuses[1]}; fi
printf '%s\n' "$rc" > "$RUN/exit_code" || echo "WARN cannot record exit_code=$rc" >&2
exit "$rc"
