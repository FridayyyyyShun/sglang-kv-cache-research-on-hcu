#!/usr/bin/env python3
"""M5 miss 归因（对应 radix-tree T5，并叠加 L2/L3）。

做法：同一 key 依次制造并记录各类 miss 原因——冷命名空间、查询上限截掉的页尾、
L1 容量不足由 L2 补救；不同 cache_salt 不得复用内存中的 device/host 树。
L3 配置下清空 L1/L2 后应由存储补救；host 配置下清空后应 0 命中。
跨 salt 的 L3 行为不属于本地 salt 隔离契约，只记录不断言。

用法：python3 exp/m5_miss_attribution.py --run-dir <RUN>
"""
import sys

import common as C


def body(e):
    key = C.tokens(8 * e.p + 1, e.seed)
    # --- 冷命名空间 ---
    e.flush("M5_baseline")
    row = e.seed_prefix(key, "cold")
    e.expect_hit(row, 0)
    e.observation("miss attribution", cause="cold_namespace", req_id=row["req_id"])
    # --- 查询上限 / 页尾排除 ---
    warm = e.gen(key, "warm_page_tail")
    e.expect_hit(warm, C.aligned(len(key) - 1, e.p), "device")
    e.observation("query/page exclusion", input_len=len(key), query_limit=len(key) - 1,
                  reusable_tokens=warm["cached"], excluded_tokens=len(key) - warm["cached"])
    # --- L1 容量 miss，由 L2 补救 ---
    e.idle("M5_warm_done")
    e.to_host(key)
    host = e.gen(key, "device_capacity_miss_host_hit")
    e.expect_hit(host, C.aligned(len(key) - 1, e.p), "host")
    e.observation("miss attribution", cause="L1_capacity_with_L2_rescue", req_id=host["req_id"])
    # --- salt 隔离：同一合成 prompt，不同本地 salt；L3 明确不在契约内 ---
    salt_key = C.tokens(8 * e.p + 1, e.seed + 99)
    e.seed_prefix(salt_key, "salt_A_seed", salt="A")
    local = e.gen(salt_key, "salt_B_query", salt="B")
    e.check((local["details"].get("device") or 0) == 0 and (local["details"].get("host") or 0) == 0,
            "different salt cannot reuse in-memory device/host tree", details=local["details"])
    # --- 清空 L1/L2：L3 补救，或彻底丢失 ---
    if e.axes["tier"] == "storage":
        e.observation("cross-salt L3 behavior outside local salt contract",
                      storage_tokens=local["details"].get("storage"),
                      note="Token-only L3 keys may reuse these tokens; this is not a local salt isolation failure")
        e.flush("M5_L3")
        replay = e.gen(key, "local_memory_miss_storage_hit")
        e.expect_hit(replay, C.aligned(len(key) - 1, e.p), "storage")
        e.observation("miss attribution", cause="L1_L2_cleared_L3_rescue", req_id=replay["req_id"])
    else:
        e.flush("M5_memory_only_loss")
        lost = e.gen(key, "all_memory_cleared")
        e.expect_hit(lost, 0)
        e.observation("miss attribution", cause="explicit_L1_L2_flush_no_L3", req_id=lost["req_id"])
    e.idle("M5_done")


if __name__ == "__main__":
    sys.exit(C.run(body, "m5_miss_attribution"))
