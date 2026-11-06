#!/usr/bin/env bash
# T1/T2/T3/T4 测试入口：必须先手动启动服务器，本脚本不会启动、重启或停止服务。
# T1：验证冷请求的 GPU→Host write_through 备份提交及完成。
# 服务需 --hicache on --hc-policy write_through --obs 1，启动时设置：
# M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0
# 运行：bash run_hicache.sh --only t1 --host 127.0.0.1
# T1默认发送1条4096-token输入，生成1个token，不验证回载或KV指纹。
# 以下均为使用示例，注释本身不会执行；完整启动说明见 README.md。
#
# 1. T2：验证目标 KV 从 Host 回载到 GPU，且与原始 KV 指纹一致。
#    服务需开启 HiCache、--obs 1，并在启动时设置：
#    M1_HICACHE_VERIFY=1 M1_EVICTION_VERIFY=0
#    运行：bash run_hicache.sh --only t2 --host 127.0.0.1
#
# 2. T4：运行五种访问场景，对比不同淘汰策略的命中和请求延迟。
#    服务需 --obs 1；启动时关闭额外校验，避免影响性能：
#    M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0
#    先用 --policy lru 启动服务，再执行：
#    bash run_hicache.sh --only t4 --policy lru --cycles 10 --repeats 3
#    cycles=10：每个场景连续执行10个访问周期，周期之间不清缓存。
#    repeats=3：每个场景独立运行3次，每次开始前清缓存并重新预热。
#    注意：性能测试会清缓存，仅在空闲专用测试服务运行。
#
# 3. T3审计：验证实际被选的GPU淘汰节点符合LRU/LFU/SLRU/Priority排序。
#    服务需 --obs 1；启动时设置：
#    M1_EVICTION_VERIFY=1 M1_HICACHE_VERIFY=0
#    先用 --policy lru 启动服务，再执行：
#    bash run_hicache.sh --only t3 --policy lru
#
# --policy 仅指定测试对应的策略，不会修改已运行服务器的策略。
# 测试其他策略时，先重启对应策略服务，再修改测试命令中的 --policy。
# 三种模式的服务环境要求不同，因此一次只选择一种 --only 模式。
# --host 未指定时从当前 RUN/axes.env 读取；RUN默认读取 .state/last_run。
# 所有命令行参数解析后才确定RUN，可通过 --run-dir 显式覆盖。
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(cat "$HERE/.state/last_run" 2>/dev/null || true)"
HOST=""; PORT=31000; ONLY=""; POLICY=""; PAGE=64; PROMPT=4096
PRESSURE=8192; ROUNDS=6; RANKS=8; WAIT=60; CYCLES=10; REPEATS=3
while [[ $# -gt 0 ]]; do
 case "$1" in
  --host) HOST="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  --run-dir) RUN="$2"; shift 2;;
  --only) ONLY="$2"; shift 2;;
  --policy) POLICY="$2"; shift 2;;
  --page) PAGE="$2"; shift 2;;
  --prompt-len) PROMPT="$2"; shift 2;;
  --pressure-len) PRESSURE="$2"; shift 2;;
  --max-rounds) ROUNDS="$2"; shift 2;;
  --expected-ranks) RANKS="$2"; shift 2;;
  --wait-seconds) WAIT="$2"; shift 2;;
  --cycles) CYCLES="$2"; shift 2;;
  --repeats) REPEATS="$2"; shift 2;;
  -h|--help) sed -n '2,5p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;;
 esac
done
[[ -d "$RUN" && -f "$RUN/axes.env" ]] || { echo 'Missing run directory or axes.env' >&2; exit 2; }
# Read data, never source axes.env as executable shell code.
axis() { sed -n "s/^$1=//p" "$RUN/axes.env" | head -n 1; }
HOST="${HOST:-$(axis host)}"
POLICY="${POLICY:-$(axis policy)}"
case "$ONLY" in t1|t2|t3|t4) ;; *) echo 'Select --only t1, t2, t4 (benchmark), or t3 (audit); startup modes differ.' >&2; exit 2;; esac
[[ -n "$HOST" ]] || { echo 'Specify --host' >&2; exit 2; }
cd "$HERE"
common=(--host "$HOST" --port "$PORT" --run-dir "$RUN" --page "$PAGE" --prompt-len "$PROMPT" --expected-ranks "$RANKS")
case "$ONLY" in
 t1)
  echo 'T1 requires HiCache on, write_through and --obs 1; verifies backup, not restore or KV bytes.'
  python3 exp/t1_hicache_write_through.py --host "$HOST" --port "$PORT" --run-dir "$RUN" --page "$PAGE" --prompt-len "$PROMPT" --wait-seconds "$WAIT"
  ;;
 t2)
  echo 'T2 requires server started with M1_HICACHE_VERIFY=1 and HiCache on.'
  python3 exp/t2_hicache_load_back.py "${common[@]}" --pressure-len "$PRESSURE" --max-rounds "$ROUNDS" --wait-seconds "$WAIT"
  ;;
 t4)
  echo 'T4 benchmark flushes cache: dedicated idle server only. Disable both verification modes at server startup.'
  python3 exp/t4_policy_benchmark.py "${common[@]}" --policy "$POLICY" --cycles "$CYCLES" --repeats "$REPEATS"
  ;;
 t3)
  echo 'T3 audit requires server started with M1_EVICTION_VERIFY=1.'
  python3 exp/t3_eviction_policy.py "${common[@]}" --policy "$POLICY" --pressure-len "$PRESSURE" --max-rounds "$ROUNDS" --wait-seconds "$WAIT"
  ;;
esac
rc=$?
printf '%s=%s\n' "$ONLY" "$rc" | tee "$RUN/run_${ONLY}_summary.txt"
echo "Evidence: $RUN"
exit "$rc"
