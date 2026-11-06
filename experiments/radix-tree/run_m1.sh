#!/usr/bin/env bash
# 依次跑完 M1 六个任务项，汇总成一份 summary。
# 前提：服务已经用 lib/serve.sh 起好（RUN 目录在 .state/last_run）。
#
# 用法：
#   bash run_m1.sh                       # 用 .state/last_run 里的 RUN
#   bash run_m1.sh --host 127.0.0.1 --port 31000
#   bash run_m1.sh --only t1,t3          # 只跑指定项
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(cat "$HERE/.state/last_run" 2>/dev/null || true)"
[[ -z "$RUN" ]] && { echo "没有 RUN：先 bash lib/serve.sh --node example" >&2; exit 2; }

HOST=127.0.0.1; PORT=31000; ONLY=""
while [[ $# -gt 0 ]]; do
 case "$1" in
  --host) HOST="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  --only) ONLY="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
 esac
done

declare -a NAMES=(t1_match_split t2_share_lock t3_page_boundary t4_multiturn t5_miss_attr t6_ready_cancel)
declare -a DESC=("匹配与分裂" "共享引用与锁" "页边界" "多轮与恢复" "miss 归因" "ready/取消/释放")

want() { [[ -z "$ONLY" ]] && return 0; [[ ",$ONLY," == *",$1,"* ]]; }

cd "$HERE/exp"
declare -a CODES=()
for i in "${!NAMES[@]}"; do
 n="${NAMES[$i]}"
 want "$n" || continue
 echo ""
 echo "=================================================================="
 echo "== ${DESC[$i]}  ($n)"
 echo "=================================================================="
 # T6 压力实验参数小一点，避免占机过久；其余用默认
 if [[ "$n" == t6_ready_cancel ]]; then
   timeout 2400 python3 "${n}.py" --host "$HOST" --port "$PORT" --run-dir "$RUN" \
     --rounds 3 --per-round 6 --prompt-len 16384
 else
   timeout 3600 python3 "${n}.py" --host "$HOST" --port "$PORT" --run-dir "$RUN"
 fi
 rc=$?
 CODES+=("$n=$rc")
 [[ $rc -ne 0 ]] && echo ">>> $n 退出码 $rc"
done

echo ""
echo "==================== SUMMARY ===================="
printf '%s\n' "${CODES[@]}" | tee "$RUN/run_m1_summary.txt"
fails=$(printf '%s\n' "${CODES[@]}" | grep -cv '=0$' || true)
echo "失败项：$fails  （0 表示全部通过）"
echo "证据目录：$RUN"
ls -1 "$RUN" | sed 's/^/  /'
exit "$fails"
