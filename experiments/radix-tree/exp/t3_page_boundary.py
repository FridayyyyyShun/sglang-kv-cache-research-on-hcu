#!/usr/bin/env python3
"""T3 页边界回归（对应 M1：页边界 / ②命中页数符合预期）。

期望值不写死成一条公式，而是同时算两种理论，命中哪种就说明走哪种语义：
  page   = L//P*P             （RadixKey.match 页对齐取整）
  minus1 = (L-1)//P*P         （is_bigram / EAGLE 分支再 -1）

用法：python3 exp/t3_page_boundary.py [--page 64]
"""
import sys

import common as C

CASES = [1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257,
         1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 16383, 16384, 32767, 32768]


def theory(L, page):
    return L // page * page, (L - 1) // page * page


def main():
    a = C.setup(C.parse_args(__doc__), "t3_page_boundary")
    P = a.page
    r = C.Report(a, "t3_page_boundary")

    for i, L in enumerate(CASES):
        C.flush(a)
        x = C.ids(L, seed=200 + i)
        cold = C.gen(a, x)
        warm = C.gen(a, x)
        exp_p, exp_m1 = theory(L, P)
        got = warm["cached"]
        if got == exp_p:
            verdict = "MATCH_page"
        elif got == exp_m1:
            verdict = "MATCH_minus1"
        else:
            verdict = "MISMATCH(exp %d|%d)" % (exp_p, exp_m1)
        r.row(L=L, pages=L // P, expect_page=exp_p, expect_minus1=exp_m1,
              cold=cold["cached"], warm=got, tail=L - (got or 0), verdict=verdict)
        r.check(cold["cached"] == 0, "L=%d 冷请求 0 命中" % L, cold["cached"])
        r.check(verdict.startswith("MATCH"), "L=%d 命中符合页对齐语义" % L,
                "%s (cached=%s)" % (verdict, got))

    # --- 分裂点落在页边界：先入 384，再查前 320 ---
    # 注意：320 是 64 的整数倍，查询本身末尾正好压在页边界上，和上面 L=256/1024 一样
    # 会触发 MATCH_minus1（命中少一页）。两种都算对，关键是不得跨页多给。
    C.flush(a)
    base = C.ids(384, seed=999)
    C.gen(a, base)
    h = C.gen(a, base[:320])
    exp_p, exp_m1 = theory(320, P)
    got = h["cached"]
    r.row(L=320, pages=5, expect_page=exp_p, expect_minus1=exp_m1,
          cold=None, warm=got, tail=320 - (got or 0),
          verdict="split-probe")
    r.check(got in (exp_p, exp_m1),
            "共享 320 前缀命中 %d 或 %d（页边界整除触发 minus1）" % (exp_p, exp_m1), got)

    # --- 非整页前缀不能被"四舍五入"命中 ---
    C.flush(a)
    q = C.ids(1024, seed=777)
    C.gen(a, q)
    h2 = C.gen(a, q[:1023])
    r.row(L=1023, pages=15, expect_page=C.aligned(1023, P), expect_minus1=None,
          cold=None, warm=h2["cached"], tail=1023 - (h2["cached"] or 0),
          verdict="non-aligned")
    r.check(h2["cached"] == C.aligned(1023, P),
            "1023 长度只能命中 %d（不得跨页多给）" % C.aligned(1023, P), h2["cached"])

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
