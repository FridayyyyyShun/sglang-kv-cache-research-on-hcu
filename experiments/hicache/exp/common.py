#!/usr/bin/env python3
"""HiCache M1-M6 机制实验：公共库（纯标准库，容器内 python3 直接可用）。

约定：
  * RUN 目录（观测产物 + axes.env）从 <hicache>/.state/last_run 读，可 --run-dir 覆盖；
  * 只做机制断言，不测 TTFT / 吞吐；
  * 每个实验结束落一份 <name>_<随机>.json 到 RUN 目录，退出码 0=PASS / 1=FAIL / 3=NOT_EXERCISED；
  * 窗口没抓到（并发共享、未完成请求复用、预取中取消）记 NOT_EXERCISED，不算通过。
"""
import argparse
import hashlib
import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotExercised(RuntimeError):
    """目标行为没有真正发生：不能判 PASS，也不是 FAIL。"""


# ---------------------------------------------------------------- 构造输入
def tokens(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 50000) for _ in range(n)]


def aligned(n, p):
    return max(0, n) // p * p


# ---------------------------------------------------------------- 树快照解析
def state_of(snap, key, salt=None):
    """沿树走 key，返回匹配长度以及其中落在 device / host 上的 token 数。"""
    rows = snap["nodes"]
    by_parent = {}
    for row in rows:
        by_parent.setdefault(row["parent"], []).append(row)
    ids = {row["id"] for row in rows}
    roots = {row["parent"] for row in rows if row["parent"] not in ids}
    parent = next(iter(roots), None)
    offset = device = host = 0
    while parent is not None:
        found = None
        for row in by_parent.get(parent, []):
            if row.get("salt") != salt:
                continue
            node_key = row["key_tokens"]
            take = min(len(node_key), len(key) - offset)
            if take and key[offset:offset + take] == node_key[:take]:
                found = row
                break
        if found is None:
            break
        take = min(found["key_len"], len(key) - offset)
        cd = found["components"]["0"]
        if cd["device_slots"]:
            device += take
        if cd["host_slots"]:
            host += take
        offset += take
        if offset == len(key):
            break
        parent = found["id"]
    return dict(matched=offset, device=device, host=host)


def branching(snap, prefix):
    """存在一个节点：从 root 累计的前缀恰为 prefix，且有 >=2 个孩子（前缀在此分叉）。"""
    rows = {n["id"]: n for n in snap["nodes"]}

    def path(n):
        return (path(rows[n["parent"]]) if n["parent"] in rows else []) + n["key_tokens"]

    return any(path(n) == prefix and len(n["children"]) >= 2 for n in rows.values())


def invariants(snap, empty=False):
    nodes = snap["nodes"]
    page = snap["page_size"]
    assert snap["device_tree_tokens"] == sum(n["components"]["0"]["device_slots"] for n in nodes), "tree/token sum differs"
    assert snap["device_tree_tokens"] == snap["evictable"] + snap["protected"], "device accounting differs"
    for node in nodes:
        assert node["key_len"] % page == 0, "node crosses partial page"
        for cd in node["components"].values():
            assert cd["device_lock"] >= 0 and cd["host_lock"] >= 0, "negative lock"
    if snap["fully_idle"] and not any(snap["ongoing"].values()):
        assert all(cd["device_lock"] == cd["host_lock"] == 0
                   for n in nodes for cd in n["components"].values()), "idle locks retained"
    if empty:
        assert not nodes and snap["device_tree_tokens"] == 0, "flush left nodes"
        assert not any(snap["ongoing"].values()), "flush left async operations"


def abort_reason(row):
    reason = row["meta_info"].get("finish_reason")
    return isinstance(reason, dict) and reason.get("type") in ("abort", "aborted")


def baseline_signature(snaps):
    """每个 rank 的 device 空闲量 + 所有 host 池（含 indexer 等辅助池），用来判定无泄漏。"""
    return {(s["pp_rank"], s["tp_rank"]): {"device_available": s["device_available"], "host_pools": s["host_pools"]}
            for s in snaps}


# ---------------------------------------------------------------- 实验驱动
class Experiment:
    def __init__(self, args, name):
        self.a = args
        self.name = name
        self.run = args.run_dir or Path((ROOT / ".state/last_run").read_text().strip())
        self.axes = dict(line.split("=", 1) for line in (self.run / "axes.env").read_text().splitlines() if "=" in line)
        if self.axes.get("obs") != "1":
            raise RuntimeError("Mechanism suite requires --obs 1")
        if self.axes.get("tier") not in ("host", "storage"):
            raise RuntimeError("Use --tier host or storage")
        if self.axes.get("host") != args.host or int(self.axes["port"]) != args.port:
            raise RuntimeError("run/server mismatch")
        self.base = "http://%s:%d" % (args.host, args.port)
        self.p = int(self.axes["page"])
        self.cap = int(self.axes["cap"])
        self.seed = args.seed + uuid.uuid4().int % 100000000
        self.result = dict(experiment=name, part="mechanism", namespace_seed=self.seed, axes=self.axes,
                           checks=[], rows=[], snapshots=[], status="running", started_at=time.time())
        self.out = self.run / (name + "_" + uuid.uuid4().hex[:8] + ".json")

    # --- HTTP ---
    def http(self, path, payload=None, timeout=120):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()

    # --- 断言 ---
    def check(self, ok, item, **evidence):
        self.result["checks"].append(dict(item=item, status="PASS" if ok else "FAIL", evidence=evidence))
        if not ok:
            raise AssertionError(item + " " + str(evidence))

    def observation(self, item, **evidence):
        self.result["checks"].append(dict(item=item, status="OBSERVED", evidence=evidence))

    def not_exercised(self, item, **evidence):
        self.result["checks"].append(dict(item=item, status="NOT_EXERCISED", evidence=evidence))

    # --- 观测 ---
    def events(self):
        out = []
        for path in self.run.glob("hicache_events_*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if value.get("time", 0) >= self.result["started_at"]:
                    out.append(value)
        return out

    def tail_events(self, offsets):
        """只读各事件文件新追加的部分（offsets 记各文件读到的字节位置），给毫秒级窗口轮询用。"""
        out = []
        for path in self.run.glob("hicache_events_*.jsonl"):
            with path.open("rb") as f:
                f.seek(offsets.get(path, 0))
                data = f.read()
            end = data.rfind(b"\n") + 1
            offsets[path] = offsets.get(path, 0) + end
            for line in data[:end].splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def snaps(self, label):
        """写 mailbox，等每个 rank 在 scheduler 线程里各落一份快照（不发请求，不碰 LRU）。"""
        tag = self.name + "_" + label + "_" + uuid.uuid4().hex
        target = self.run / "snapshot.request.json"
        tmp = self.run / ("snapshot." + tag + ".tmp")
        tmp.write_text(json.dumps({"tag": tag}))
        tmp.replace(target)
        deadline = time.monotonic() + self.a.timeout
        found = {}
        while time.monotonic() < deadline:
            found = {}
            for path in self.run.glob("hicache_snapshots_*.jsonl"):
                for line in path.read_text().splitlines():
                    try:
                        s = json.loads(line)
                    except ValueError:
                        continue
                    if s.get("tag") == tag:
                        found[(s["pp_rank"], s["tp_rank"])] = s
            if len(found) == self.a.ranks:
                values = list(found.values())
                self.check(all(s["cache_class"] == "UnifiedRadixCache" and s["hicache_attached"] for s in values),
                           "Unified HiCache attached on all ranks")
                for s in values:
                    invariants(s)
                self.result["snapshots"].append({"label": label, "tag": tag, "ranks": values})
                return values
            time.sleep(0.2)
        raise RuntimeError("Snapshot timeout: %d/%d ranks; inspect snapshot_error events" % (len(found), self.a.ranks))

    def idle(self, label):
        """请求队列、异步回写 / 载回 / 预取、控制队列全部清空才返回。"""
        deadline = time.monotonic() + self.a.timeout
        while time.monotonic() < deadline:
            values = self.snaps(label)
            if all(s["fully_idle"] and not any(s["ongoing"].values())
                   and not any(s.get("control_queues", {}).values())
                   and s.get("prefetch_tokens_occupied", 0) == 0 for s in values):
                return values
            time.sleep(0.2)
        raise RuntimeError("Async/request queues did not drain: " + label)

    def flush(self, label="flush"):
        self.idle(label + "_before")
        text = self.http("/flush_cache?timeout=60", {}, 90)
        self.check("flushed" in text.lower() and "fail" not in text.lower(), "flush acknowledged", response=text)
        values = self.idle(label + "_after")
        for s in values:
            invariants(s, empty=True)
        return values

    # --- 请求 ---
    def gen(self, key, label, salt=None, max_new=1, request_id=None, on_first=None):
        rid = request_id or "hicache_" + self.name + "_" + uuid.uuid4().hex
        payload = {"rid": rid, "input_ids": key, "stream": True, "sampling_params": {
            "max_new_tokens": max_new, "temperature": 0, "ignore_eos": True, "top_k": 1}}
        if salt is not None:
            payload["cache_salt"] = salt
        req = urllib.request.Request(self.base + "/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        final = None
        first = False
        with urllib.request.urlopen(req, timeout=self.a.timeout + 600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                final = json.loads(data)
                if "error" in final:
                    raise RuntimeError(str(final["error"]))
                meta = final.get("meta_info", {})
                if not first and (meta.get("completion_tokens", 0) > 0 or final.get("output_ids") or final.get("text")):
                    first = True
                    if on_first:
                        on_first(rid)
        if final is None:
            raise RuntimeError("No native SSE result")
        meta = final.get("meta_info", {})
        cached = meta.get("cached_tokens")
        if cached is None and abort_reason(final):
            # prefill 之前就被取消（例如 L3 预取中 abort）：服务端不报命中，按 0 记并标明
            cached, meta = 0, dict(meta, cached_tokens_reported=False)
        if cached is None:
            raise RuntimeError("Missing cached_tokens")
        details = meta.get("cached_tokens_details") or ({"device": 0, "host": 0} if cached == 0 else None)
        if details is None:
            raise NotExercised("cached_tokens_details missing; cannot attribute layers")
        self.check(cached == sum(details.get(k, 0) or 0 for k in ("device", "host", "storage")),
                   "response layer accounting", req_id=rid, cached=cached, details=details)
        row = dict(label=label, req_id=rid, input_len=len(key),
                   input_sha256=hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                   salt=salt, cached=cached, details=details, output_ids=final.get("output_ids"),
                   text=final.get("text"), meta_info=meta)
        self.result["rows"].append(row)
        return row

    def expect_hit(self, row, expected, layer=None):
        self.check(row["cached"] == expected, "page-aligned reusable prefix",
                   label=row["label"], expected=expected, actual=row["cached"])
        if layer and expected:
            self.check((row["details"].get(layer) or 0) > 0, "positive " + layer + " hit", row=row)

    def seed_prefix(self, key, label="seed", salt=None):
        row = self.gen(key, label, salt=salt)
        self.idle(label + "_backups_drained")
        return row

    def to_host(self, key, salt=None):
        """用无关请求自然挤压 L1，直到目标前缀在每个 rank 上都只剩 host 副本。"""
        limit = aligned(len(key) - 1, self.p)
        if not limit:
            return self.idle("no_complete_reusable_page")
        # 快照不碰 LRU；压力预算随 device 容量缩放
        pressure_len = min(self.cap // 4, 2048)
        pressure_limit = self.a.pressure_limit
        if pressure_limit is None:
            pressure_limit = (self.cap + pressure_len - 1) // pressure_len + 4
        for i in range(pressure_limit):
            values = self.idle("l2_pressure_%d" % i)
            states = [state_of(s, key[:limit], salt) for s in values]
            if all(v["matched"] >= limit and v["device"] == 0 and v["host"] >= limit for v in states):
                self.observation("target is host-only on all ranks", states=states)
                return values
            self.gen(tokens(pressure_len, self.seed + 500000 + i), "pressure_%d" % i)
        raise NotExercised("Target never became host-only within pressure limit; no L2 claim")

    # --- 收尾 ---
    def finish(self, status="PASS", error=None):
        events = self.events()
        self.result["events"] = events
        self.result.update(status=status, error=error)
        # 响应里报了 L2/L3 命中，就必须在每个 rank 上看到真实的搬运完成事件
        for row in self.result["rows"]:
            for layer, event_name, field in (("host", "init_load_back", "loaded_device_tokens"),
                                             ("storage", "_handle_prefetch_result", "storage_loaded_tokens")):
                if (row["details"].get(layer) or 0) <= 0:
                    continue
                pids = {ev["pid"] for ev in events if ev["event"] == event_name
                        and ev.get("req_id") == row["req_id"] and ev.get(field, 0) > 0}
                ok = len(pids) == self.a.ranks
                self.result["checks"].append(dict(
                    item="positive " + layer + " response has completed per-rank transfer evidence",
                    status="PASS" if ok else "FAIL",
                    evidence={"req_id": row["req_id"], "rank_processes": sorted(pids)}))
                if not ok:
                    self.result["status"] = "FAIL"
        bad = [ev for ev in events if ev["event"] in ("observer_error", "snapshot_error")]
        if bad:
            self.result["status"] = "FAIL"
            self.result["observation_errors"] = bad
        protected_errors = [ev for ev in events if ev.get("locked_page_violations")]
        if protected_errors:
            self.result["status"] = "FAIL"
            self.result["locked_page_violations"] = protected_errors
        self.out.write_text(json.dumps(self.result, indent=2, ensure_ascii=False))
        for c in self.result["checks"]:
            print("  %-13s %s" % (c["status"], c["item"]))
        print("\n[%s] %s -> %s" % (self.name, self.result["status"], self.out))
        return 0 if self.result["status"] == "PASS" else (3 if self.result["status"] == "NOT_EXERCISED" else 1)


def run(body, name):
    """统一入口：前置检查 -> body(e) -> 落盘；把 NotExercised / 异常映射成状态。"""
    e = Experiment(parser().parse_args(), name)
    try:
        e.check(e.cap >= 4096, "mechanism fixture has sufficient device capacity", cap=e.cap)
        e.check(e.p == 64, "first-pass HCU mechanism suite uses verified page64", page=e.p)
        body(e)
        incomplete = any(c["status"] == "NOT_EXERCISED" for c in e.result["checks"])
        return e.finish("NOT_EXERCISED" if incomplete else "PASS")
    except NotExercised as exc:
        return e.finish("NOT_EXERCISED", str(exc))
    except Exception as exc:
        return e.finish("FAIL", repr(exc))


def parser():
    ap = argparse.ArgumentParser(description="L1/L2/L3 mechanism suite; no performance metrics")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=32000)
    ap.add_argument("--run-dir", type=Path, help="观测产物目录，默认读 .state/last_run")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--pressure-limit", type=int, default=None, help="覆盖按容量估算的自然驱逐请求数")
    ap.add_argument("--cycles", type=int, default=3, help="M6 取消-释放循环次数")
    return ap
