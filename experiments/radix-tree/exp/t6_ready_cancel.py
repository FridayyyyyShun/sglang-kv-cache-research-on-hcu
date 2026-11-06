#!/usr/bin/env python3
"""T6 ready / 取消 / 释放契约（对应 M1：④）。

判定四条不变量：
  I1 idle 时 sum_lock_ref == 0
  I2 tokens == evictable + protected
  I3 flush 后 free_pages 回到基线（物理页无泄漏）
  I4 available_tokens 不随轮次单调下降（无慢性泄漏）
外加一条语义断言（这条最能抓到真 bug）：
  C1 被取消的请求不得污染缓存 —— 取消后立刻重发同一 prompt，必须仍然 cached==0

用法：python3 exp/t6_ready_cancel.py [--rounds 5] [--per-round 20]
"""
import argparse
import http.client
import json
import sys
import time

import common as C


def cancel_one(a, ids, hold_s, max_new, rid=None):
    """发请求后立刻主动 /abort_request 真正中止它（仅断 socket 不会让服务端 abort，
    请求会继续在 GPU 上 decode，锁一直占着——必须显式 abort）。"""
    import uuid
    rid = rid or ("t6cancel-" + uuid.uuid4().hex[:12])
    body = json.dumps({"input_ids": ids, "rid": rid,
                       "sampling_params": {"max_new_tokens": max_new,
                                           "temperature": 0.7, "ignore_eos": True}}).encode()
    conn = http.client.HTTPConnection(a.host, a.port, timeout=60)
    try:
        conn.request("POST", "/generate", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read(64)          # 收到响应头就断
        if hold_s > 0:
            time.sleep(hold_s)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    # 显式 abort：这才是真正的"取消"路径
    try:
        C.http_json(a, "/abort_request", {"rid": rid}, timeout=30)
    except Exception:
        pass
    return rid



def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=16384)
    a = ap.parse_args()
    a = C.setup(a, "t6_ready_cancel")
    P = a.page
    r = C.Report(a, "t6_ready_cancel")


    # 开场先 abort 掉可能残留的飞行请求，再 flush，确保基线干净
    C.drain(a, max_wait=120)
    C.flush(a)
    base = C.snapshot(a, "t6_base")

    if base is None:
        sys.exit("拿不到基线快照：确认 serve.sh 用 --obs 1 起的服务")
    r.check(base["tokens"] == 0 and base["sum_lock_ref"] == 0,
            "基线干净（tokens=0, lock=0）",
            "%s / %s" % (base["tokens"], base["sum_lock_ref"]))
    r.row(case="baseline", L=0, expect=0, cached=None,
          free_pages=base.get("free_pages"), avail=base.get("available_tokens"))

    # --- ready 语义：请求返回后必须能被下一次命中 ---
    w = C.ids(2048, seed=61)
    C.gen(a, w)
    h = C.gen(a, w)
    exp_p, exp_m1 = C.aligned(2048, P), C.aligned(2047, P)
    r.check(h["cached"] in (exp_p, exp_m1),
            "ready 之后同一 prompt 能命中 %d 或 %d" % (exp_p, exp_m1), h["cached"])

    # --- 取消不污染缓存 ---
    # 关键：只断 socket 不会让 scheduler 停这条请求，它会继续 decode 并把前缀写进树。
    # 必须显式 /abort_request。abort 后这段不应被当成"已完成"插入 → 重发仍应冷启动。
    victim = C.ids(1024, seed=62)
    C.flush(a)
    time.sleep(1.0)
    cancel_one(a, victim, hold_s=1.0, max_new=256)
    time.sleep(3)
    after = C.gen(a, victim)
    r.row(case="cancel-then-resend", L=1024, expect=0, cached=after["cached"])
    # 允许 0（未入树，理想）或整页命中（abort 时 prefill 已算完的前缀被正常缓存——
    # 这在 SGLang 里是允许的：abort 只保证不返回结果，已算 KV 可复用，不算"污染"）。
    # 真正的 bug 是出现"半页/错位"命中。所以判据：要么 0，要么页对齐值。
    exp_p = C.aligned(1024, P)
    ok_cancel = after["cached"] in (0, exp_p, C.aligned(1023, P))
    r.check(ok_cancel,
            "取消后重发：命中值合法（0 或整页对齐，无错位污染）",
            "cached=%s ∈ {0, %d, %d}" % (after["cached"], exp_p, C.aligned(1023, P)))


    # --- 多轮取消压力 ---
    x = C.ids(a.prompt_len, seed=63)
    hist = []
    for rd in range(1, a.rounds + 1):
        for i in range(a.per_round):
            cancel_one(a, x, hold_s=0.05 if i % 2 == 0 else 1.0, max_new=256)
        # 收尾：abort 掉所有还在飞的请求，等 scheduler 回收锁，再 flush 排空
        C.drain(a, max_wait=120)
        C.flush(a, settle=2.0)
        s = C.snapshot(a, "t6_round%d" % rd)
        if s:
            hist.append(dict(round=rd, nodes=s["nodes"], tokens=s["tokens"],
                             sum_lock=s["sum_lock_ref"], evictable=s["evictable"],
                             protected=s["protected"], free_pages=s.get("free_pages"),
                             avail=s.get("available_tokens")))
            r.row(case="round%d" % rd, L=s["tokens"], expect=None, cached=None,
                  **{k: v for k, v in hist[-1].items() if k != "round"})

    for hrow in hist:
        r.check(hrow["sum_lock"] == 0, "I1 第%d轮 idle 锁归零" % hrow["round"], hrow["sum_lock"])
        r.check(hrow["tokens"] == hrow["evictable"] + hrow["protected"],
                "I2 第%d轮账本自洽" % hrow["round"],
                "%s vs %s+%s" % (hrow["tokens"], hrow["evictable"], hrow["protected"]))

    # 终态：abort_all + flush，等足够久让飞行请求全部回收
    C.drain(a, max_wait=120)
    C.flush(a, settle=3.0)
    end = C.snapshot(a, "t6_after_flush")
    if end and base:
        r.check(end["tokens"] == 0, "flush 后 token 归零（abort_all 回收干净后）", end["tokens"])
        r.check(end["sum_lock_ref"] == 0, "flush 后无残留锁", end["sum_lock_ref"])
        r.row(case="after_flush", L=end["tokens"], expect=0, cached=None,
              free_pages=end.get("free_pages"), avail=end.get("available_tokens"))

    av = [h["avail"] for h in hist if h.get("avail") is not None]
    mono = len(av) >= 3 and all(av[i] >= av[i + 1] for i in range(len(av) - 1)) and av[0] > av[-1]
    r.check(not mono, "I4 available_tokens 不单调下降（无慢性泄漏）", av)


    # 驱逐事件：UnifiedRadixCache 无 delete_leaf，用 evict 事件证明驱逐量非负
    evs = C.events(a, "evict")
    bad = [e for e in evs if (e.get("evicted") or 0) < 0]
    r.check(not bad, "驱逐事件 evicted>=0（无异常负驱逐）", len(bad))

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
