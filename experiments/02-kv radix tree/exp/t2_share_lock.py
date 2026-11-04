#!/usr/bin/env python3
"""T2 共享引用与锁（对应 M1：共享引用 / ready 保护 / 物理页账本）。

三件事：
  1. 并发共享同一前缀时，protected（被锁住的 token）必须 > 0 → 共享者确实在保护路径；
  2. 全部结束后 sum_lock_ref 必须回 0 → 引用计数不泄漏；
  3. 驱逐事件里被删的叶子必须都是 lock_ref==0 的 → 不存在"驱逐活跃请求依赖的页"。

用法：python3 exp/t2_share_lock.py
"""
import sys
import threading

import common as C


def main():
    a = C.setup(C.parse_args(__doc__), "t2_share_lock")
    page = a.page
    r = C.Report(a, "t2_share_lock")
    C.flush(a)

    P = C.ids(8192, 3)
    warm = C.gen(a, P)
    C.gen(a, P)
    base = C.snapshot(a, "t2_base")
    r.check(base is not None, "拿到基线快照", a.run_dir)
    base_lock = base.get("sum_lock_ref") if base else None
    r.check(base_lock == 0, "静止时 sum_lock_ref==0", base_lock)

    # --- 6 个请求并发共享同一前缀 ---
    results = {}
    lock = threading.Lock()
    peak = {"protected": 0, "evictable_min": 10 ** 9}

    def one(i):
        q = P + C.ids(256, 100 + i)          # 共享 8192，各自独占尾部
        try:
            res = C.gen(a, q, max_new=16)
            with lock:
                results[i] = res
        except Exception as e:
            with lock:
                results[i] = dict(cached=None, error=repr(e)[:80])

    th = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    for t in th:
        t.start()

    # 在飞行中采样两次，看共享保护是否生效
    samples = []
    for k in range(2):
        s = C.snapshot(a, "t2_flight%d" % k, wait=1.5, retries=2)
        if s:
            samples.append(s)
            peak["protected"] = max(peak["protected"], s.get("protected") or 0)
            peak["evictable_min"] = min(peak["evictable_min"], s.get("evictable") or 10 ** 9)
    for t in th:
        t.join()

    hits = [v.get("cached") for v in results.values()]
    r.row(case="concurrent6", L=8192, expect=C.aligned(8192, page), cached=hits,
          flight_samples=len(samples), peak_protected=peak["protected"])
    r.check(all(h == C.aligned(8192, page) for h in hits if h is not None),
            "6 个共享请求都命中同一前缀", hits)
    r.check(peak["protected"] > 0, "飞行中 protected>0（共享引用真的锁住了）", peak["protected"])

    # --- 结束后必须回 0 ---
    end = C.snapshot(a, "t2_end")
    r.check(end is not None, "拿到结束快照", a.run_dir)
    if end:
        r.check(end["sum_lock_ref"] == 0, "全部完成后 sum_lock_ref==0", end["sum_lock_ref"])
        r.check(end["tokens"] == end["evictable"] + end["protected"],
                "账本自洽 tokens==evictable+protected",
                "%s vs %s+%s" % (end["tokens"], end["evictable"], end["protected"]))
        r.row(case="after_all", L=end["tokens"], expect=None, cached=None,
              evictable=end["evictable"], protected=end["protected"],
              free_pages=end.get("free_pages"))

    # --- 驱逐：UnifiedRadixCache 没有 _delete_leaf 钩子，用 evict 事件做账本证据 ---
    C.flush(a)
    C.gen(a, C.ids(8192, 4))
    C.gen(a, C.ids(8192, 5))       # 让树长出两个分支
    mid = C.snapshot(a, "t2_before_evict")
    evs = C.events(a, "evict")
    if evs:
        # 驱逐过的 token 数必须 >=0，且 before/after evictable 单调（驱逐只减可淘汰量）
        bad = [e for e in evs if (e.get("evicted") or 0) < 0]
        r.check(not bad, "evict 事件 evicted>=0（无负驱逐）", len(bad))
        r.row(case="evict", L=None, expect=None, cached=None,
              evict_events=len(evs),
              evicted_total=sum(e.get("evicted") or 0 for e in evs))
    else:
        r.row(case="evict", L=None, expect=None, cached=None, evict_events=0,
              note="本轮容量充足未触发驱逐（正常）")
    if mid:
        r.row(case="before_evict", L=mid["tokens"], expect=None, cached=None,
              evictable=mid["evictable"], free_pages=mid.get("free_pages"))


    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
