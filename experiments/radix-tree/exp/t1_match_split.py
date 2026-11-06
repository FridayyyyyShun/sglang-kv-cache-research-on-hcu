#!/usr/bin/env python3
"""T1 匹配与分裂（对应 M1：①原理摸清 / 树图 / 调用链 / ②命中页数）。

做法：构造确定性的共享前缀，用 cached_tokens 断言命中长度，再用结构化树快照断言
"分裂点确实出现在页边界上"，并把树形状落盘作为树图证据。

用法：python3 exp/t1_match_split.py
"""
import sys

import common as C


def main():
    a = C.setup(C.parse_args(__doc__), "t1_match_split")
    page = a.page
    r = C.Report(a, "t1_match_split")
    C.flush(a)

    # --- 全量插入：A 一次冷、一次热 ---
    A = C.ids(256, 1)
    cold = C.gen(a, A)
    warm = C.gen(a, A)
    r.row(case="A-256-cold", L=256, expect=0, cached=cold["cached"])
    r.row(case="A-256-warm", L=256, expect=C.aligned(256, page), cached=warm["cached"])
    r.check(cold["cached"] == 0, "冷请求必须 0 命中", cold["cached"])
    # 末 token 可能不入树（要留 1 个 token 才能出首 token），允许差一页
    ok_full = warm["cached"] in (C.aligned(256, page), C.aligned(256, page) - page)
    r.check(ok_full, "热请求命中全前缀（允许末页不进树）",
            "%s ∈ {%d,%d}" % (warm["cached"], C.aligned(256, page), C.aligned(256, page) - page))

    # --- 分裂：B 只共享前 160（页对齐 128），后段全新 ---
    n0 = C.mark_events(a)
    B = A[:160] + C.ids(192, 2)
    hb = C.gen(a, B)
    exp_b = C.aligned(160, page)
    r.row(case="B-share160", L=len(B), expect=exp_b, cached=hb["cached"])
    r.check(hb["cached"] == exp_b, "共享 160 token 应命中 %d（页对齐取整）" % exp_b, hb["cached"])

    # --- 树结构断言 ---
    # 线上 UnifiedRadixCache 按「页」切节点：每个节点 klen 多为 page_size（64），
    # 不是旧 RadixCache 的「一整段一节点」。所以"分裂点在 128"= 从 root 到某节点
    # 累计 klen==exp_b(128) 且该节点有 >=2 个孩子（前缀在此分叉）。
    snap = C.snapshot(a, "t1_after_B")
    r.check(snap is not None, "拿到树快照", a.run_dir)
    if snap:
        nodes = snap.get("nodes_detail") or []
        by_id = {n["id"]: n for n in nodes}

        def depth_tokens(n):
            """从 root 累加到 n（含）的 klen 之和 = 该节点代表的前缀长度。"""
            tot, cur = 0, n
            seen = set()
            while cur is not None and cur["id"] not in seen:
                seen.add(cur["id"])
                tot += cur["klen"]
                cur = by_id.get(cur["parent"])
            return tot

        split = [n for n in nodes
                 if depth_tokens(n) == exp_b
                 and len([c for c in nodes if c["parent"] == n["id"]]) >= 2]
        r.check(bool(split),
                "分裂点落在页边界 %d（累计前缀长）且分叉 >=2 支" % exp_b,
                "找到 %d 个分叉节点" % len(split))
        # tokens 来自 total_size()，统计 FULL 组件槽数之和 = sum(val)
        r.check(snap["tokens"] == sum(n["val"] for n in nodes),
                "树内 token 账本自洽 sum(val)==total_size",
                "%s vs %s" % (snap["tokens"], sum(n["val"] for n in nodes)))
        r.row(case="snapshot", L=snap["tokens"], expect=None, cached=None,
              nodes=snap["nodes"], leaves=snap["leaves"])

    # --- 更短前缀：C 只共享 64 ---
    Cc = A[:64] + C.ids(56, 9)          # 共享 64，再带 56 个自己的 token
    hc = C.gen(a, Cc)
    r.row(case="C-share64", L=len(Cc), expect=64, cached=hc["cached"])
    r.check(hc["cached"] == 64, "共享 64 token 应命中 64", hc["cached"])

    # --- 新分支插入后，A 的全前缀命中不能被破坏 ---
    ha = C.gen(a, A)
    r.row(case="A-again-after-split", L=256, expect=warm["cached"], cached=ha["cached"])
    r.check(ha["cached"] == warm["cached"], "分裂后原路径命中不变", ha["cached"])

    # --- 调用链证据：match/insert 事件成对出现且带 key 元信息 ---
    ev = C.events_since(a, n0)
    kinds = {}
    for e in ev:
        kinds[e["event"]] = kinds.get(e["event"], 0) + 1
    r.row(case="events", **{"L": None, "expect": None, "cached": None})
    r.check(kinds.get("match", 0) >= 2 and kinds.get("insert", 0) >= 1,
            "match/insert 事件成对齐全", kinds)
    lack = [e for e in ev if e["event"] == "match" and "aligned_len" not in e]
    r.check(not lack, "每条 match 事件都带 aligned_len/hit_tokens（miss 可归因的前提）", len(lack))

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
