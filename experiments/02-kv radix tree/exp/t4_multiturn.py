#!/usr/bin/env python3
"""T4 多轮与恢复（对应 M1：多轮 / 未完成请求增量 / 恢复正确性）。

三个子项：
  A. 多轮累积：同一 session 每轮把历史整体重发，命中必须等于"上一轮总长的页对齐值"；
  B. 未完成请求：长请求还在 decode 时发一个共享其前缀的新请求，必须能命中已算好的部分
     （走 cache_unfinished_req 路径）；
  C. 恢复：/flush_cache 之后树必须真正清空，随后从头累积。

用法：python3 exp/t4_multiturn.py
"""
import sys
import threading
import time

import common as C


def main():
    a = C.setup(C.parse_args(__doc__), "t4_multiturn")
    P = a.page
    r = C.Report(a, "t4_multiturn")

    # ---------------- A. 多轮累积 ----------------
    C.flush(a)
    hist = C.ids(512, seed=11)
    prev = 0
    for turn in range(1, 6):
        g = C.gen(a, hist)
        exp = C.aligned(prev, P) if prev else 0
        r.row(case="turn%d" % turn, L=len(hist), expect=exp, cached=g["cached"])
        r.check(g["cached"] == exp, "第 %d 轮命中 %d" % (turn, exp), g["cached"])
        prev = len(hist)
        hist = hist + C.ids(512, seed=20 + turn)   # 追加下一轮用户输入

    # ---------------- B. 未完成请求的增量命中 ----------------
    C.flush(a)
    long_ids = C.ids(8192, seed=31)
    box = {}

    def slow():
        try:
            box["r"] = C.gen(a, long_ids, max_new=128)
        except Exception as e:
            box["err"] = repr(e)[:80]

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(2.0)                     # 让 prefill 至少算完一部分
    probe = C.gen(a, long_ids[:4096])   # 共享它已算好的前 4096
    t.join()
    got = probe["cached"]
    r.row(case="unfinished-overlap", L=4096, expect=C.aligned(4096, P), cached=got)
    r.check(isinstance(got, int) and got > 0,
            "长请求未完成时，新请求能命中已算前缀（>0）", got)

    # ---------------- C. 恢复：flush 后必须真清空 ----------------
    C.flush(a)
    s0 = C.snapshot(a, "t4_after_flush")
    r.check(s0 is not None, "拿到 flush 后快照", a.run_dir)
    if s0:
        r.row(case="after_flush", L=s0["tokens"], expect=0, cached=None,
              nodes=s0["nodes"], locks=s0["sum_lock_ref"], free_pages=s0.get("free_pages"))
        r.check(s0["tokens"] == 0, "flush 后树内 token 归零", s0["tokens"])
        r.check(s0["sum_lock_ref"] == 0, "flush 后无残留锁引用", s0["sum_lock_ref"])

    fresh = C.gen(a, long_ids[:4096])
    r.row(case="after_flush_cold", L=4096, expect=0, cached=fresh["cached"])
    r.check(fresh["cached"] == 0, "flush 后首个请求必须冷启动", fresh["cached"])

    again = C.gen(a, long_ids[:4096])
    # 4096 是整页，查询末尾压在页边界 → 可能 MATCH_minus1（4032）或 MATCH_page（4096）
    exp_p, exp_m1 = C.aligned(4096, P), C.aligned(4096 - 1, P)
    r.check(again["cached"] in (exp_p, exp_m1),
            "重新累积后能再次命中 %d 或 %d" % (exp_p, exp_m1), again["cached"])

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
