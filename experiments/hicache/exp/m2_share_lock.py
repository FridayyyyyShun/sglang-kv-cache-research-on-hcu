#!/usr/bin/env python3
"""M2 共享引用与锁（对应 radix-tree T2，并叠加 L2/L3）。

做法：同一长前缀先写入并压到 L2，再并发 3 个只在尾部不同的请求；在飞行中抓快照，
要求每个 rank running>=2 且共享节点 device_lock>=2，排空后 protected 归零。
L3 配置下 flush 后再做一遍。全程检查 host 锁确实被取得、驱逐没有回收被锁页。
最后单独制造 host 驱逐：host 池（ratio x device）比 device 大，自然压力填不满它，evict_host
根本不会被调用。所以先用无关请求把 host 池写满，再把目标压到 L2；先发写入请求 W，
它一进调度就发目标 T：T 的 L2 载回持 host 锁期间，调度器处理 W 的 prefill 结果并 write-through，
host 已满就必须 evict_host。要求 evict_host 发生时确有 host 锁节点，且这些节点的 host 页一个不少。
没抓到飞行窗口记 NOT_EXERCISED；几次尝试都没让 host 驱逐与 host 锁重叠也记 NOT_EXERCISED。

用法：python3 exp/m2_share_lock.py --run-dir <RUN>
"""
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import common as C


def concurrent_phase(e, key, label):
    """并发共享同一前缀；必须在飞行中观测到 >=2 个引用，否则不下结论。"""
    workers = 3
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(e.gen, key + C.tokens(e.p, i + e.seed), label + "_%d" % i, None, 512)
                   for i in range(workers)]
        flight = None
        deadline = time.monotonic() + e.a.timeout
        while time.monotonic() < deadline and not all(f.done() for f in futures):
            snapshots = e.snaps(label + "_flight")
            if all(s["running"] >= 2 and any(n["components"]["0"]["device_lock"] >= 2 for n in s["nodes"])
                   for s in snapshots):
                flight = snapshots
                break
            time.sleep(0.1)
        rows = [future.result() for future in futures]
    if flight is None:
        raise C.NotExercised("No protected live interval captured for " + label)
    e.check(all(any(n["components"]["0"]["device_lock"] > 0 for n in s["nodes"]) for s in flight),
            "shared device references protected during flight", phase=label)
    after = e.idle(label + "_drained")
    e.check(all(s["protected"] == 0 for s in after),
            "request protection released after concurrent requests", phase=label)
    return rows


def host_eviction_phase(e, key, attempts=5):
    """host 池写满后，让 evict_host 落在目标 L2 载回（持 host 锁）期间。

    host 配置里 host 锁只在载回期间持有：load_back 加锁，载回完成后下一轮调度的 loading_check 放锁；
    目标自己的 write-through 在放锁之后，碰不上。overlap 调度下，发出目标 batch 之后才处理上一个
    batch 的结果，所以先发写入请求 W，看到它被调度（match_prefix）立刻发目标 T：W 的 prefill
    结果在 T 载回期间处理，W 的 write-through 在 host 已满时必须 evict_host。
    """
    e.flush("M2_host_pressure")
    host_cap = min(pool["logical_size"] for s in e.snaps("host_capacity") for pool in s["host_pools"].values())
    filler_len = min(e.cap // 4, 2048)
    for i in range(host_cap // filler_len + 2):
        e.gen(C.tokens(filler_len, e.seed + 300000 + i), "host_fill_%d" % i)
    e.idle("host_filled")
    e.seed_prefix(key, "host_pressure_seed")
    for attempt in range(attempts):
        e.to_host(key)
        since = time.time()
        wid = "hicache_host_evict_writer_" + uuid.uuid4().hex
        offsets = {}
        e.tail_events(offsets)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(e.gen, C.tokens(filler_len, e.seed + 320000 + attempt),
                                 "host_evict_writer_%d" % attempt, None, 1, wid)
            deadline = time.monotonic() + e.a.timeout
            while time.monotonic() < deadline and not any(
                    x["event"] == "match_prefix" and x.get("req_id") == wid for x in e.tail_events(offsets)):
                time.sleep(0.002)
            row = pool.submit(e.gen, key + C.tokens(e.p, e.seed + 310000 + attempt),
                              "host_lock_target_%d" % attempt).result()
            writer.result()
        e.idle("host_pressure_%d_drained" % attempt)
        recent = [x for x in e.events() if x["time"] >= since and x["event"] == "evict_host"]
        overlap = [x for x in recent if x.get("protected_before")]
        e.observation("host eviction attempt", attempt=attempt, target_host_hit=row["details"].get("host"),
                      evict_host_calls=len(recent), overlapping_calls=len(overlap))
        if (row["details"].get("host") or 0) > 0 and overlap:
            e.check(not any(x.get("locked_page_violations") for x in overlap),
                    "host eviction competing with active host lock", attempt=attempt, req_id=row["req_id"],
                    evictions=[dict(pid=x["pid"], protected=x["protected_before"], evicted=x["evicted_tokens"])
                               for x in overlap])
            return
    e.not_exercised("host eviction competing with active host lock",
                    note="evict_host never ran while a host lock was held after %d attempts" % attempts)


def body(e):
    key = C.tokens(16 * e.p + 1, e.seed)
    # --- L2：前缀只在 host 上，并发请求共享载回 ---
    e.flush("M2_baseline")
    e.seed_prefix(key)
    e.to_host(key)
    rows = concurrent_phase(e, key, "L2_shared")
    e.check(any((r["details"].get("host") or 0) > 0 for r in rows), "concurrent batch actually used L2")
    # --- L3：L1/L2 清空，并发请求共享预取 ---
    if e.axes["tier"] == "storage":
        e.flush("M2_L3")
        rows = concurrent_phase(e, key, "L3_shared")
        e.check(any((r["details"].get("storage") or 0) > 0 for r in rows), "concurrent batch actually used L3")
    # --- 锁与驱逐 ---
    events = e.events()
    e.check(any(x["event"] == "inc_host_lock_ref" and x.get("host_lock", 0) > 0 for x in events),
            "host references acquired for transfers")
    violations = [x for x in events if x.get("locked_page_violations")]
    e.check(not violations, "no observed eviction reclaimed locked device/host pages", violations=violations)
    # --- host 驱逐 vs host 锁 ---
    try:
        host_eviction_phase(e, C.tokens(C.aligned(min(e.cap // 2, 64 * e.p), e.p) + 1, e.seed + 1))
    except C.NotExercised as exc:
        e.not_exercised("host eviction competing with active host lock", note=str(exc))
    violations = [x for x in e.events() if x.get("locked_page_violations")]
    e.check(not violations, "no eviction reclaimed locked pages after host pressure", violations=violations)


if __name__ == "__main__":
    sys.exit(C.run(body, "m2_share_lock"))
