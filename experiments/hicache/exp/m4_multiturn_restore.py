#!/usr/bin/env python3
"""M4 多轮与恢复（对应 radix-tree T4，并叠加 L2/L3）。

做法：4 轮对话，每轮把上一轮完整输入 + 原生 output_ids + 新 2 页拼成下一轮，期望命中
上一轮输入的页对齐长度。随后把整段历史压到 L2 回放；flush 后 L3 配置应从存储恢复，
host 配置应 0 命中。最后起一个长请求，在它出 token 后用重叠前缀探测未完成请求的增量发布：
同一 pid 内必须按 cache_unfinished_req -> 探测 match_prefix -> cache_finished_req 的顺序出现，
且所有 rank 都成立才算 PASS，否则记 NOT_EXERCISED。

用法：python3 exp/m4_multiturn_restore.py --run-dir <RUN>
"""
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import common as C


def body(e):
    # --- 多轮：每轮复用上一轮完整输入 ---
    key = C.tokens(16 * e.p + 1, e.seed)
    old = None
    e.flush("M4_baseline")
    for turn in range(4):
        row = e.gen(key, "turn_%d" % turn)
        e.expect_hit(row, 0 if old is None else C.aligned(old, e.p))
        e.idle("turn_%d_committed" % turn)
        if not row["output_ids"]:
            raise C.NotExercised("Native output_ids unavailable for exact multiturn token history")
        old = len(key)
        if turn < 3:
            key = key + row["output_ids"] + C.tokens(2 * e.p, e.seed + turn + 1)
    # --- 历史从 L2 回放 ---
    e.to_host(key)
    replay = e.gen(key, "history_from_L2")
    e.expect_hit(replay, C.aligned(len(key) - 1, e.p), "host")
    # --- 清空内存后恢复：L3 有，host-only 没有 ---
    e.flush("M4_drop_memory")
    recovered = e.gen(key, "history_after_flush")
    if e.axes["tier"] == "storage":
        e.expect_hit(recovered, C.aligned(len(key) - 1, e.p), "storage")
    else:
        e.expect_hit(recovered, 0)
    e.idle("M4_recovered")
    # --- 未完成请求的增量发布 ---
    live_key = C.tokens(24 * e.p + 1, e.seed + 70000)
    rid = "hicache_unfinished_" + uuid.uuid4().hex
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(e.gen, live_key, "unfinished_source", None, 512, rid, lambda _: started.set())
        if not started.wait(e.a.timeout):
            raise RuntimeError("unfinished source never produced a token")
        probe = e.gen(live_key + C.tokens(e.p, e.seed + 70001), "unfinished_overlap_probe")
        future.result()
    e.check(probe["cached"] <= C.aligned(len(live_key), e.p) and probe["cached"] % e.p == 0,
            "unfinished overlap cannot reuse beyond completed page prefix", cached=probe["cached"])
    events = e.events()
    ranks = set(x["pid"] for x in events if x.get("req_id") == probe["req_id"])
    proven = []
    for pid in ranks:
        matches = [x["time"] for x in events if x["pid"] == pid and x["event"] == "match_prefix"
                   and x.get("req_id") == probe["req_id"]]
        publications = [x["time"] for x in events if x["pid"] == pid and x["event"] == "cache_unfinished_req"
                        and x.get("req_id") == rid]
        finishes = [x["time"] for x in events if x["pid"] == pid and x["event"] == "cache_finished_req"
                    and x.get("req_id") == rid]
        proven.append(bool(matches and publications and finishes
                           and min(publications) <= min(matches) < min(finishes)))
    e.result["checks"].append(dict(
        item="unfinished-request incremental publication before completion",
        status="PASS" if len(proven) == e.a.ranks and all(proven) and probe["cached"] > 0 else "NOT_EXERCISED",
        evidence={"ranks_proven": sum(proven), "probe_req_id": probe["req_id"], "source_req_id": rid,
                  "note": "Probe after source completion does not prove unfinished reuse"}))
    e.idle("M4_unfinished_drained")


if __name__ == "__main__":
    sys.exit(C.run(body, "m4_multiturn_restore"))
