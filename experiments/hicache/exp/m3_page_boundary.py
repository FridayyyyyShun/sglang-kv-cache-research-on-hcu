#!/usr/bin/env python3
"""M3 页边界（对应 radix-tree T3，并叠加 L2/L3）。

做法：长度取 p-1, p, p+1, 2p-1, 2p, 2p+1, 4p+1；每个长度冷、热各一次，期望命中
aligned(n-1, p)（最后一个 token 必须重算）。有完整页时压到 L2 再查一次，期望值不变。
L3 配置下 flush 后从存储恢复，并检查 device/host/storage 各层命中都按页对齐。

用法：python3 exp/m3_page_boundary.py --run-dir <RUN>
"""
import sys

import common as C


def body(e):
    p = e.p
    for i, n in enumerate([p - 1, p, p + 1, 2 * p - 1, 2 * p, 2 * p + 1, 4 * p + 1]):
        key = C.tokens(n, e.seed + i)
        expected = C.aligned(n - 1, p)
        # --- L1：冷 / 热 ---
        e.flush("M3_len%d" % n)
        cold = e.seed_prefix(key, "L%d_cold" % n)
        e.expect_hit(cold, 0)
        warm = e.gen(key, "L%d_L1" % n)
        e.expect_hit(warm, expected)
        # --- L2：只剩 host 副本 ---
        if expected:
            e.idle("boundary_ready")
            e.to_host(key)
            host = e.gen(key, "L%d_L2" % n)
            e.expect_hit(host, expected, "host")
        else:
            e.observation("no full page lies within query limit", input_len=n, query_limit=n - 1)
        # --- L3：L1/L2 清空，从存储恢复 ---
        if e.axes["tier"] == "storage":
            e.flush("M3_storage_len%d" % n)
            storage = e.gen(key, "L%d_L3" % n)
            e.expect_hit(storage, expected, "storage" if expected else None)
            for name in ("device", "host", "storage"):
                value = storage["details"].get(name) or 0
                e.check(value % p == 0, "layer hit is page aligned", length=n, layer=name, value=value)
        e.idle("M3_len%d_done" % n)


if __name__ == "__main__":
    sys.exit(C.run(body, "m3_page_boundary"))
