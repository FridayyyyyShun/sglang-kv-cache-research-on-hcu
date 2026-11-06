#!/usr/bin/env python3
"""M1 匹配与分裂（对应 radix-tree T1，并叠加 L2/L3）。

做法：A 冷、热请求确认 L1 命中；把 A 压到 L2 后，B 只共享页对齐的前缀并分叉，
再用树快照确认分裂点落在页边界上且分叉 >=2 支。L3 配置下 flush 后从外部存储恢复，
同样检查分叉。原路径 A 的命中在分裂后保持不变。

用法：python3 exp/m1_match_split.py --run-dir <RUN>
"""
import sys

import common as C


def body(e):
    p = e.p
    common = 5 * p
    a = C.tokens(8 * p + 1, e.seed)
    shared = a[:common + p // 2]
    b = shared + C.tokens(3 * p + 1, e.seed + 1)
    e.flush("M1_baseline")
    cold = e.seed_prefix(a, "A_cold")
    e.expect_hit(cold, 0)
    warm = e.gen(a, "A_L1")
    e.expect_hit(warm, 8 * p, "device")
    e.idle("A_warm_done")
    e.to_host(a)
    branch = e.gen(b, "B_shared_partial_page_L2")
    e.expect_hit(branch, common, "host")
    snaps = e.idle("after_L2_branch")
    e.check(all(C.branching(s, a[:common]) for s in snaps),
            "L2 split at aligned shared prefix on every rank",
            shared_tokens=len(shared), split_tokens=common)
    original = e.gen(a, "A_after_split")
    e.expect_hit(original, 8 * p)
    if e.axes["tier"] == "storage":
        e.flush("M1_L3_restore")
        c = shared + C.tokens(3 * p + 1, e.seed + 2)
        restored = e.gen(c, "C_new_branch_L3")
        e.expect_hit(restored, common, "storage")
        e.gen(a, "A_after_L3_branch")
        snaps = e.idle("after_L3_branch")
        e.check(all(C.branching(s, a[:common]) for s in snaps),
                "L3 restored prefix branches at same page boundary")


if __name__ == "__main__":
    sys.exit(C.run(body, "m1_match_split"))
