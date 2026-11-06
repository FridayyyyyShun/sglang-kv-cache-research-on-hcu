#!/usr/bin/env python3
"""M6 ready/取消/释放（对应 radix-tree T6，并叠加 L2/L3）。

做法：先取空树基线（device 空闲量 + 所有 host 池）。每个循环：冷参考 -> 同 key 热参考（命中
整页前缀、未经取消）-> 压到 L2 -> 发长请求，出第一个 token 后定向 abort -> 必须真的以 abort
结束且未到 token 上限 -> 重试命中长度与热参考相同、输出逐 token 一致 -> flush 后池恢复到基线、
无累计泄漏。请求在 abort 之前已完成记 NOT_EXERCISED。L3 配置下看到预取发起立刻 abort，
以每个 rank 的 release_aborted_request 调用前仍挂着该预取为证据；几次都没落在窗口里记 NOT_EXERCISED。

为什么对比热参考而不是冷参考：复用前缀时只对尾部 token 做 prefill，和全量 prefill 走不同的
kernel 形状，bf16 下贪心输出可以合法地不同（s3 实测：冷/冷一致、热/热一致、热/冷 4/4 不同）。
取消如果污染了被复用的 KV，会表现为重试和同命中长度的热参考不一致；冷参考差异只记录不断言。

用法：python3 exp/m6_cancel_release.py --run-dir <RUN> [--cycles 3]
"""
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import common as C


def prefetch_cancel_phase(e, signature, attempts=5):
    """看到 prefetch_from_storage 立刻定向 abort；证据是每个 rank 的 release_aborted_request
    在调用前仍持有该请求的 ongoing_prefetch 条目（不是快照里碰巧看到 ongoing>0）。"""
    # 前缀尽量长（不超过 device 容量），把预取窗口从 ~60ms 拉长
    n = C.aligned(min(e.cap - 2 * e.p, 96 * e.p), e.p) + 1
    for attempt in range(attempts):
        key = C.tokens(n, e.seed + 90000 + attempt)
        e.seed_prefix(key, "prefetch_cancel_seed_%d" % attempt)
        e.flush("prefetch_cancel_clear_memory_%d" % attempt)
        rid = "hicache_prefetch_cancel_" + uuid.uuid4().hex
        offsets = {}
        e.tail_events(offsets)
        sent = False
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(e.gen, key, "prefetch_cancel_%d" % attempt, None, 1024, rid)
            deadline = time.monotonic() + min(e.a.timeout, 30)
            while not future.done() and time.monotonic() < deadline:
                if any(x["event"] == "prefetch_from_storage" and x.get("req_id") == rid
                       for x in e.tail_events(offsets)):
                    e.http("/abort_request", {"rid": rid, "abort_all": False})
                    sent = True
                    break
                time.sleep(0.002)
            if not sent and not future.done():
                e.http("/abort_request", {"rid": rid, "abort_all": False})
            row = future.result()
        e.idle("prefetch_cancel_%d_drained" % attempt)
        events = [x for x in e.events() if x.get("req_id") == rid]
        released = {x["pid"] for x in events if x["event"] == "release_aborted_request" and x.get("ongoing_before")}
        loaded = {x["pid"] for x in events if x["event"] == "_handle_prefetch_result"
                  and x.get("storage_loaded_tokens", 0) > 0}
        e.observation("prefetch cancel attempt", attempt=attempt, abort_sent_on_prefetch=sent,
                      finish_reason=row["meta_info"].get("finish_reason"),
                      ranks_released_during_prefetch=len(released), ranks_completed_prefetch=len(loaded))
        e.flush("prefetch_cancel_final_%d" % attempt)
        e.check(C.baseline_signature(e.idle("prefetch_cancel_empty_%d" % attempt)) == signature,
                "prefetch cancel leaves no pool leak", attempt=attempt)
        if C.abort_reason(row) and len(released) == e.a.ranks:
            e.check(True, "cancellation during active L3 prefetch", attempt=attempt, req_id=rid,
                    ranks_released_during_prefetch=sorted(released))
            return
    e.not_exercised("cancellation during active L3 prefetch",
                    note="abort never landed while every rank still had the prefetch pending (%d attempts)" % attempts)


def body(e):
    baseline = e.flush("M6_empty_baseline")
    signature = C.baseline_signature(baseline)
    for cycle in range(e.a.cycles):
        # --- 冷参考 ---
        key = C.tokens(16 * e.p + 1, e.seed + cycle)
        reference = e.gen(key, "cycle%d_cold_reference" % cycle, max_new=8)
        e.expect_hit(reference, 0)
        e.idle("cycle%d_reference_committed" % cycle)
        warm = e.gen(key, "cycle%d_warm_reference" % cycle, max_new=8)
        e.expect_hit(warm, C.aligned(len(key) - 1, e.p), "device")
        e.idle("cycle%d_warm_committed" % cycle)
        e.to_host(key)
        e.idle("cycle%d_before_cancel" % cycle)

        def cancel(rid):
            """出第一个 token 后定向 abort（不是 abort_all）。"""
            e.snaps("cycle%d_cancel_flight" % cycle)
            response = e.http("/abort_request", {"rid": rid, "abort_all": False})
            e.observation("targeted abort submitted", req_id=rid, response=response)

        # --- 飞行中取消 ---
        row = e.gen(key, "cycle%d_cancel" % cycle, max_new=1024, on_first=cancel)
        if not C.abort_reason(row) and row["meta_info"].get("completion_tokens", 0) >= 1024:
            raise C.NotExercised("Request completed before abort injection; no cancellation safety claim")
        e.check(C.abort_reason(row) and row["meta_info"].get("completion_tokens", 1024) < 1024,
                "request terminated by real abort before token limit", req_id=row["req_id"], meta_info=row["meta_info"])
        e.idle("cycle%d_cancel_drained" % cycle)
        # --- 重试 + 释放 ---
        retry = e.gen(key, "cycle%d_retry" % cycle, max_new=8)
        e.flush("cycle%d_cleanup" % cycle)
        after = e.idle("cycle%d_empty" % cycle)
        e.check(C.baseline_signature(after) == signature, "device and every host/sidecar pool restored to baseline",
                cycle=cycle, before=str(signature), after=str(C.baseline_signature(after)))
        # 同一前缀、同样的命中长度，只差中间经历过一次取消
        e.check(retry["cached"] == warm["cached"], "retry reuses the same prefix length as warm reference",
                cycle=cycle, retry=retry["details"], warm=warm["details"])
        left = retry["output_ids"] or retry["text"]
        right = warm["output_ids"] or warm["text"]
        e.check(left == right, "retry output agrees with uncancelled warm reference", cycle=cycle,
                retry=left, warm=right, retry_details=retry["details"], warm_details=warm["details"])
        cold = reference["output_ids"] or reference["text"]
        e.observation("cold full-prefill reference (numerics differ from prefix reuse; not asserted)",
                      cycle=cycle, equal_to_retry=cold == left, cold=cold)
        e.flush("cycle%d_final_flush" % cycle)
        e.check(C.baseline_signature(e.idle("cycle%d_final_empty" % cycle)) == signature,
                "no cumulative pool leak", cycle=cycle)
    # --- L3：abort 必须在 L3 预取进行中落地 ---
    if e.axes["tier"] == "storage":
        prefetch_cancel_phase(e, signature)

if __name__ == "__main__":
    sys.exit(C.run(body, "m6_cancel_release"))
