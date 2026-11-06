#!/usr/bin/env bash
# HiCache M1-M6 机制实验一键跑（对应 radix-tree/run_m1.sh）。
# 前提：已 bash lib/serve.sh --obs 1 --tier host|storage 起服务，.state/last_run 指向本次 RUN。
# 用法：bash run_m1.sh [--host H] [--port P] [--only m1,m3] [其余参数原样透传给每个实验]
# 退出码：0 全部 PASS；3 有 NOT_EXERCISED 无 FAIL；1 有 FAIL。
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export M1_RADIX_OBSERVE=0 M0_OBSERVE=0 PYTHONDONTWRITEBYTECODE=1

RUN="$(cat "$HERE/.state/last_run" 2>/dev/null || true)"
if [ -z "$RUN" ]; then echo "找不到 RUN 目录：先 bash lib/serve.sh 起服务" >&2; exit 2; fi

HOST="${M1_HOST:-127.0.0.1}"
PORT=32000
ONLY=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --run-dir) RUN="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

NAMES=(m1_match_split m2_share_lock m3_page_boundary m4_multiturn_restore m5_miss_attribution m6_cancel_release)
DESC=("匹配与分裂" "共享引用与锁" "页边界" "多轮与恢复" "miss 归因" "ready/取消/释放")

want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",${1%%_*},"* ]]; }

cd "$HERE/exp"
CODES=()
for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"
  want "$n" || continue
  echo
  echo "=================== $n  ${DESC[$i]} ==================="
  timeout 3600 python3 -B "${n}.py" --host "$HOST" --port "$PORT" --run-dir "$RUN" ${EXTRA[@]+"${EXTRA[@]}"}
  CODES+=("$n=$?")
done

fails=0
skips=0
{
  echo "==================== SUMMARY ===================="
  for c in "${CODES[@]}"; do
    rc="${c#*=}"
    case "$rc" in
      0) s=PASS ;;
      3) s=NOT_EXERCISED ;;
      *) s="FAIL(rc=$rc)" ;;
    esac
    echo "  ${c%%=*}  $s"
  done
} | tee "$RUN/run_m1_summary.txt"
# 管道里是子 shell，计数放在外面
for c in "${CODES[@]}"; do
  case "${c#*=}" in 0) ;; 3) skips=$((skips+1)) ;; *) fails=$((fails+1)) ;; esac
done

echo
echo "失败项：$fails  未覆盖项：$skips  （均为 0 表示全部真实通过）"
echo "证据目录：$RUN"
ls "$RUN"
if [ "$fails" -gt 0 ]; then exit 1; elif [ "$skips" -gt 0 ]; then exit 3; else exit 0; fi
