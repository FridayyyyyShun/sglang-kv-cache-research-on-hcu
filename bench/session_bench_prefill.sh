#!/bin/bash
# GLM-5.3 prefill-only session benchmark (max_tokens=1)
#
# 数据集: hy4-preview 2.3k sessions，按"session 首条记录时间"顺序下发，跑完全部 2300 条
# 复现要点（缺一不可）:
#   1. --session_order start_time : session 之间的顺序显式固定，不再依赖数据文件的排布
#   2. --tail_threshold 0         : 不在活跃并发下降时中断长尾 → 实际发送条数也固定
#   3. --dump_plan                : 下发计划落盘，两次运行 diff 即可确认顺序一致
#   4. --rounds 1 --duration 0    : 不重复轮次、不循环复用队列
# 环境要求: 需要 ijson（deps 目录），否则加载 1.2GB JSON 数组会直接报错
set -u

REPLAY_ID="${REPLAY_ID:-lru-c64-r1}"
PLAN_DIR="./plans"
# plan 文件名带时间戳：即使重复使用同一个 REPLAY_ID 也不会互相覆盖
RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$PLAN_DIR" ./logs

export PYTHONPATH="${PYTHONPATH:-}"
# 只在当前解释器缺 ijson 时，才把只读的 deps 目录补进 PYTHONPATH
if ! python3 -c 'import ijson' >/dev/null 2>&1; then
    export PYTHONPATH="${DEPS_DIR:-/public/home/apply_test/juyh/glm5.3_int4/suiyz/deps}${PYTHONPATH:+:$PYTHONPATH}"
fi

setsid python3 -u ./session_concurrency_benchmark.py \
    --target_model /model/GLM-5.3-Channel-INT4-w4a8 \
    --base_url http://0.0.0.0:30016/v1 \
    --data_files ./hy4-preview-0829-1930-cache94.77_2.3k-by-session.json \
    --session_key session_id \
    --concurrency_levels 64 \
    --max_tokens 1 \
    --session_id_header x-session-id \
    --session_order start_time \
    --order_seed 0 \
    --replay_id "$REPLAY_ID" \
    --dump_plan "$PLAN_DIR/${REPLAY_ID}_${RUN_TS}.json" \
    --pre_sample 2300 \
    --tail_threshold 0 \
    --rounds 1 \
    --duration 0 \
    2>&1 | tee "./logs/glm53_prefill_${REPLAY_ID}_$(date +%Y%m%d_%H%M%S).log"
