#!/usr/bin/env python3
"""M1 radix-tree 实验：公共库（纯标准库，容器内 python3 直接可用）。

约定：
  * 服务地址从 <radix-tree>/.state/last_run 里读不到，就用 --host/--port；
  * RUN 目录（观测产物）同样从 .state/last_run 读，可 --run-dir 覆盖；
  * 每个实验结束都落一份 <name>.json + <name>.csv 到 RUN 目录，并返回退出码。
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

PAGE_DEFAULT = 64


# ---------------------------------------------------------------- 基础设施
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def last_run():
    p = os.path.join(repo_root(), ".state", "last_run")
    try:
        return open(p).read().strip()
    except Exception:
        return None


def base_parser(desc):
    """返回未解析的 ArgumentParser，供需要追加参数的实验（如 T6）使用。"""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--run-dir", default=None, help="观测产物目录，默认读 .state/last_run")
    ap.add_argument("--page", type=int, default=PAGE_DEFAULT)
    ap.add_argument("--out", default=None, help="结果前缀，默认 <run-dir>/<脚本名>")
    return ap


def parse_args(desc):
    return base_parser(desc).parse_args()



def setup(a, name):
    run = a.run_dir or last_run()
    if not run:
        sys.exit("找不到 RUN 目录：先 bash lib/serve.sh 起服务，或传 --run-dir")
    a.run_dir = run
    if not a.out:
        a.out = os.path.join(run, name)
    return a


# ---------------------------------------------------------------- HTTP
def _url(a, path):
    return "http://%s:%d%s" % (a.host, a.port, path)


def http_json(a, path, payload=None, timeout=1800):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_url(a, path), data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def wait_ready(a, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            http_json(a, "/health", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def flush(a, settle=0.5):
    """清空 radix 树，保证每个用例的冷/热基线干净。

    SGLang 的 /flush_cache 在有 running/waiting 请求时会拒绝执行（返回提示文本，
    不是 JSON）。所以带 timeout 参数让它先等飞行请求排空，再清。
    """
    try:
        http_json(a, "/flush_cache?timeout=30", timeout=60)
    except Exception:
        # 返回的是纯文本（"Cache flushed." 之类），不是 JSON——正常，忽略
        pass
    time.sleep(settle)


def abort_all(a, settle=6.0):
    """主动中止所有飞行请求（取消/压力实验收尾必备）。"""
    try:
        http_json(a, "/abort_request", {"abort_all": True}, timeout=30)
    except Exception:
        pass
    time.sleep(settle)


def running_reqs(a):
    """从 /get_server_info 读当前在跑/排队请求数；拿不到返回 None。"""
    try:
        d = http_json(a, "/get_server_info", timeout=10)
    except Exception:
        return None
    for key in ("internal_states", "scheduler_info"):
        sub = d.get(key)
        if isinstance(sub, list) and sub:
            sub = sub[0]
        if isinstance(sub, dict):
            nr = sub.get("num_running_reqs")
            nq = sub.get("num_waiting_reqs") or sub.get("num_queue_reqs") or 0
            if nr is not None:
                return int(nr) + int(nq or 0)
    nr = d.get("num_running_reqs")
    return int(nr) if nr is not None else None


def drain(a, max_wait=90, poll=3.0):
    """abort_all 后轮询到服务端没有在跑/排队的请求，再返回。

    大请求在 GPU 上停下需要时间；仅 sleep 固定秒数不可靠。无法读到计数时退回
    固定等待。返回最终读到的请求数（或 None）。
    """
    abort_all(a, settle=2.0)
    t0 = time.time()
    last = None
    while time.time() - t0 < max_wait:
        n = running_reqs(a)
        last = n
        if n is None:
            time.sleep(poll)
            # 读不到就再 abort 一次兜底
            abort_all(a, settle=0.0)
            if time.time() - t0 > 20:
                break
            continue
        if n <= 0:
            break
        # 还有在飞的，继续 abort + 等
        abort_all(a, settle=0.0)
        time.sleep(poll)
    return last




def ids(n, seed=1):
    """确定性 token 序列；不同 seed 前缀不同，用例之间不会互相命中。"""
    return [1000 + ((i * 37 + seed * 7919) % 50000) for i in range(n)]


def gen(a, input_ids, salt=None, max_new=1, timeout=1800):
    p = {"input_ids": input_ids,
         "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0,
                             "ignore_eos": True, "top_k": 1}}
    if salt is not None:
        p["cache_salt"] = salt
    t0 = time.time()
    r = http_json(a, "/generate", p, timeout=timeout)
    mi = r.get("meta_info") or {}
    c = mi.get("cached_tokens")
    if c is None:
        c = (mi.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(cached=c, prompt_tokens=mi.get("prompt_tokens"),
                wall=round(time.time() - t0, 3), raw=mi)


def aligned(n, page):
    return n // page * page


# ---------------------------------------------------------------- 观测产物
def snapshot(a, tag, wait=3.0, retries=6):
    """让 scheduler 打一份结构化树快照，读回最大的那份。

    竞态处理：sentinel 由客户端写在 RUN 目录，scheduler 在下一次 match_prefix 时
    消费它并把快照追加到 radix_snapshots_<pid>.jsonl。所以这里：
      1) 写 sentinel；
      2) 主动发一个极短请求逼 scheduler 走一次 match_prefix（否则空闲时永不触发）；
      3) 轮询读回 tag 匹配、nodes 最多的那条。
    """
    sentinel = os.path.join(a.run_dir, "snapshot.request")
    with open(sentinel, "w") as f:
        f.write(tag)
    # 逼一次 match_prefix：发个 1-token 请求，scheduler 处理时会消费 sentinel
    try:
        gen(a, ids(1, 777), max_new=1, timeout=30)
    except Exception:
        pass
    best = None
    for _ in range(retries):
        time.sleep(wait)
        for fn in os.listdir(a.run_dir):
            if not (fn.startswith("radix_snapshots_") and fn.endswith(".jsonl")):
                continue
            with open(os.path.join(a.run_dir, fn)) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("tag") == tag and (best is None or d.get("nodes", 0) > best.get("nodes", 0)):
                        best = d
        if best is not None:
            return best
        # 还没落地：sentinel 可能已被别的 rank 消费，重写一次并再逼一发
        if not os.path.exists(sentinel):
            with open(sentinel, "w") as f:
                f.write(tag)
            try:
                gen(a, ids(1, 777), max_new=1, timeout=30)
            except Exception:
                pass
    return best



def events(a, kind=None):
    out = []
    for fn in os.listdir(a.run_dir):
        if not (fn.startswith("radix_events_") and fn.endswith(".jsonl")):
            continue
        with open(os.path.join(a.run_dir, fn)) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if kind is None or d.get("event") == kind:
                    out.append(d)
    out.sort(key=lambda d: d.get("time", 0))
    return out


def mark_events(a):
    """返回当前事件条数，用于只看之后新增的事件。"""
    return len(events(a))


def events_since(a, n0, kind=None):
    return events(a)[n0:] if kind is None else [d for d in events(a)[n0:] if d.get("event") == kind]


# ---------------------------------------------------------------- 报告
class Report:
    def __init__(self, a, name):
        self.a = a
        self.name = name
        self.checks = []
        self.rows = []

    def check(self, ok, key, detail=""):
        self.checks.append(dict(ok=bool(ok), item=key, detail=str(detail)))
        print(("  PASS  " if ok else "  FAIL  ") + key + (("  | " + str(detail)) if detail else ""))
        return ok

    def row(self, **kw):
        self.rows.append(kw)
        return kw

    def finish(self):
        bad = [c for c in self.checks if not c["ok"]]
        out = dict(experiment=self.name, run_dir=self.a.run_dir,
                   passed=len(self.checks) - len(bad), failed=len(bad),
                   checks=self.checks, rows=self.rows)
        jp = self.a.out + ".json"
        with open(jp, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if self.rows:
            keys = sorted({k for r in self.rows for k in r})
            with open(self.a.out + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.rows:
                    w.writerow({k: r.get(k) for k in keys})
        print("\n[%s] %d passed / %d failed -> %s.json" % (self.name, out["passed"], out["failed"], self.a.out))
        return 0 if not bad else 1
