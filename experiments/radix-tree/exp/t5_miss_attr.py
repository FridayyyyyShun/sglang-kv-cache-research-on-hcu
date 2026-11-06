#!/usr/bin/env python3
"""T5 每次 miss 有原因（对应 M1：③ 每次 miss 有原因）。

对每个请求算「理论应命中」，与实际 cached_tokens 比对，把差值归到四类：
  cold            冷启动，树里没有这条路径
  page_tail       非整页尾段，本来就不该命中
  salt_mismatch   cache_salt/命名空间不同，必须 0 命中（这是"正确行为"，不是 miss）
  capacity        容量不足被驱逐（用 delete_leaf 事件证明）
任何一条归不进去，就是"无法解释的 miss"，直接判失败。

用法：python3 exp/t5_miss_attr.py
"""
import sys

import common as C


def main():
    a = C.setup(C.parse_args(__doc__), "t5_miss_attr")
    P = a.page
    r = C.Report(a, "t5_miss_attr")
    C.flush(a)

    # 整页长度查询时末页压在边界上，线上会 MATCH_minus1（命中少一页），
    # 所以热命中的期望写成一个集合 {aligned(L), aligned(L-1)}。
    def warm_set(L):
        return {C.aligned(L, P), C.aligned(L - 1, P)}

    cases = []

    # 1) 冷启动
    x1 = C.ids(2048, seed=41)
    cases.append(("cold", x1, None, {0}, "cold"))

    # 2) 热命中（整页）
    cases.append(("warm-page", x1, None, warm_set(2048), None))

    # 3) 非整页尾段：尾段本来就不该命中
    x3 = C.ids(2000, seed=42)
    C.gen(a, x3)                                    # 先入 2000
    cases.append(("tail-partial", x3, None, warm_set(2000), "page_tail"))

    # 4) salt 不同：必须 0 命中，属于"正确的 miss"
    cases.append(("salt-A", x1, "tenant-A", {0}, "salt_mismatch"))
    cases.append(("salt-A-warm", x1, "tenant-A", warm_set(2048), None))
    cases.append(("salt-B-cold", x1, "tenant-B", {0}, "salt_mismatch"))

    # 5) 容量压力：同一长 prompt 连发两次——首发冷(0)，次发应热命中；
    #    若池子被挤爆导致次发命中下降，必须有 evict 事件解释，否则归 cold。
    pressure = C.ids(8192, seed=50)
    cases.append(("pressure-cold", pressure, None, {0}, "cold"))
    cases.append(("pressure-warm", pressure, None, None, "capacity_or_warm"))

    rows = []
    n0 = C.mark_events(a)
    unexplained = 0
    for name, x, salt, exp, reason in cases:
        before = len(C.events(a, "evict"))
        got = C.gen(a, x, salt=salt)
        after = len(C.events(a, "evict"))
        evicted = after - before
        al = C.aligned(len(x), P)
        c = got["cached"]
        if exp is not None:
            ok = (c in exp)
            why = reason if not ok else ""
        else:
            # 没给期望值：允许命中 al/al-page，低于这些必须有驱逐事件解释
            ok = (c in warm_set(len(x))) or (c is not None and c < al - P and evicted > 0)
            why = "" if ok else reason
        if not ok:
            unexplained += 1
        rows.append(dict(case=name, L=len(x), salt=salt, aligned=al, cached=c,
                         evict_events=evicted, expect=sorted(exp) if exp else None,
                         reason=why or "-", ok=ok))
        r.row(**rows[-1])

    for x in rows:
        r.check(x["ok"], "用例 %s 命中可解释" % x["case"],
                "cached=%s aligned=%s evict=%s reason=%s"
                % (x["cached"], x["aligned"], x["evict_events"], x["reason"]))
    r.check(unexplained == 0, "没有无法归因的 miss", unexplained)


    # observer 侧交叉验证：每条 match 都带 key_len/aligned_len/miss_tokens
    ev = C.events_since(a, n0, "match")
    bad = [e for e in ev if e.get("aligned_len") is None or e.get("hit_tokens") is None]
    r.check(not bad, "match 事件全部带 aligned_len/hit_tokens", len(bad))
    mism = [e for e in ev if e.get("miss_tokens") not in (0, None)
            and e["hit_tokens"] == e["aligned_len"]]
    r.check(not mism, "hit==aligned 时 miss_tokens==0（账目自洽）", len(mism))

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
